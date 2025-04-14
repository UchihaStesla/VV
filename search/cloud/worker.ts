// 调试模式配置
const ALLOW_DEBUG = false; // 开启调试模式方便排查问题
const EXTERNAL_API_URL = 'https://vvapi.cicada000.work/search'; // 外部 API 地址

// AI 服务相关类型定义
interface SFEmbeddingResponse {
    object: 'list';
    model: string;
    data: Array<{
        embedding: number[];
        index: number;
    }>;
    usage: {
        prompt_tokens: number;
        total_tokens: number;
    };
}

interface AIService {
    run(model: string, input: { text: string[] }): Promise<{ data: number[][] }>;
    SF_API_KEY?: string;
}

// 向量数据库相关类型定义
interface VectorMetadata {
    text: string;
    timestamp: string;
    filename: string;
    image_similarity: number;
}

// 向量数据库查询结果的类型定义
interface VectorMatch {
    id: string;
    score: number;
    metadata: VectorMetadata;
}

interface VectorQueryResponse {
    count: number;
    matches: VectorMatch[];
}

interface VectorIndex {
    query(vector: number[], options: {
        topK: number;
        filter?: {
            image_similarity?: { $gte: number };
        };
    }): Promise<VectorQueryResponse>;
}

interface Env {
    AI: AIService;
    SUBTITLE_INDEX: VectorIndex;
    SF_API_KEY?: string;  // Silicon Flow API key
}

interface SearchResult {
    filename: string;
    timestamp: string;
    match_ratio: number;
    text: string;
    similarity: number;
}

// 自定义错误类型
interface SearchError extends Error {
    code?: number;
}

// 响应对象类型定义
interface ErrorResponse {
    error: string;
    debug?: string[];
}

// 搜索处理参数及相似度过滤逻辑
async function handleSearch(
    query: string,
    minRatio: number | undefined,
    minSimilarity: number,
    maxResults: number,
    env: Env,
    debugLogs: string[]
): Promise<SearchResult[]> {
    let embeddingResult;
    
    if (env.SF_API_KEY) {
        // 使用 Silicon Flow API
        debugLogs.push("Using Silicon Flow API for embeddings");
        const response = await fetch('https://api.siliconflow.cn/v1/embeddings', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${env.SF_API_KEY}`,
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model: 'BAAI/bge-m3',
                input: [query],
                encoding_format: 'float'
            })
        });

        if (!response.ok) {
            const error = await response.text();
            throw new Error(`Silicon Flow API failed: ${error}`);
        }

        const sfResult = await response.json() as SFEmbeddingResponse;
        embeddingResult = { data: [sfResult.data[0].embedding] };
    } else {
        // 使用 Cloudflare AI API
        debugLogs.push("Using Cloudflare AI API for embeddings");
        embeddingResult = await env.AI.run('@cf/baai/bge-m3', {
            text: [query]
        });
    }
    
    debugLogs.push("Embedding result: " + JSON.stringify(embeddingResult));
    
    if (!embeddingResult?.data?.[0] || embeddingResult.data[0].length !== 1024) {
        throw new Error('Invalid embedding vector generated');
    }
    
    // 修改 filterCriteria，当 minRatio 存在且大于 0 时设置过滤条件，否则不设置
    const filterCriteria = (minRatio !== undefined && !isNaN(minRatio) && minRatio > 0)
        ? { image_similarity: { $gte: minRatio / 100 } }
        : undefined;
    
    const queryOptions = {
        topK: maxResults,
        returnMetadata: "all",
        filter: filterCriteria
    };
    debugLogs.push("Query options: " + JSON.stringify(queryOptions));
    
    const queryResult = await env.SUBTITLE_INDEX.query(embeddingResult.data[0], queryOptions);
    debugLogs.push("Vector query result: " + JSON.stringify(queryResult));
    
    return queryResult.matches
        .map(v => ({
            text: v.metadata.text,
            timestamp: v.metadata.timestamp,
            filename: v.metadata.filename,
            match_ratio: v.score,
            similarity: v.metadata.image_similarity
        }))
        .filter(v => v.match_ratio >= minSimilarity)
        .sort((a, b) => b.match_ratio - a.match_ratio)
        .slice(0, maxResults)
        .map(v => ({
            ...v,
            match_ratio: Math.round(v.match_ratio * 1000) / 1000,
            similarity: Math.round(v.similarity * 1000) / 1000
        }));
}

// 转发请求到外部 API
async function forwardToExternalAPI(
    url: URL, 
    request: Request, 
    debugLogs: string[]
): Promise<Response> {
    // 构建外部 API URL
    const externalUrl = new URL(EXTERNAL_API_URL);
    
    // 复制所有查询参数，但移除 rag 参数
    url.searchParams.forEach((value, key) => {
        if (key !== 'rag') {
            externalUrl.searchParams.append(key, value);
        }
    });
    
    debugLogs.push(`Forwarding request to: ${externalUrl.toString()}`);
    
    // 创建新的请求，保留原始请求的方法和头信息
    const forwardRequest = new Request(externalUrl.toString(), {
        method: request.method,
        headers: request.headers,
    });
    
    debugLogs.push(`Forward request created with URL: ${forwardRequest.url}`);
    
    // 发送请求并返回结果
    const response = await fetch(forwardRequest);
    
    debugLogs.push(`Received response from external API with status: ${response.status}`);
    
    // 如果内容是文本，获取并记录它
    try {
        const clonedResponse = response.clone();
        const responseText = await clonedResponse.text();
        debugLogs.push(`Response body (first 200 chars): ${responseText.substring(0, 200)}`);
        
        // 构建新的响应，保留原始响应的状态和内容
        return new Response(responseText, {
            status: response.status,
            statusText: response.statusText,
            headers: {
                'Content-Type': response.headers.get('Content-Type') || 'application/json; charset=utf-8',
                'Access-Control-Allow-Origin': '*'
            }
        });
    } catch (error) {
        debugLogs.push(`Error reading response: ${error}`);
        // 构建新的响应，保留原始响应的状态和内容
        return new Response(response.body, {
            status: response.status,
            statusText: response.statusText,
            headers: {
                'Content-Type': response.headers.get('Content-Type') || 'application/json; charset=utf-8',
                'Access-Control-Allow-Origin': '*'
            }
        });
    }
}

export default {
    async fetch(request: Request, env: Env): Promise<Response> {
        // CORS 预检请求处理
        if (request.method === 'OPTIONS') {
            return new Response(null, {
                headers: {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Content-Type': 'application/json; charset=utf-8'
                }
            });
        }

        // 仅允许 GET 请求
        if (request.method !== 'GET') {
            return new Response('Method not allowed', { status: 405 });
        }

        const debugLogs: string[] = [];
        const url = new URL(request.url);
        
        // 调试模式配置
        const debugMode = url.searchParams.get('debug') === 'true'; // 永远允许调试模式
        debugLogs.push(`Request URL: ${url.toString()}`);
        debugLogs.push(`Query Parameters: ${JSON.stringify(Object.fromEntries(url.searchParams.entries()))}`);
        
        // 检查是否需要将请求转发到外部 API
        const ragParam = url.searchParams.get('rag');
        debugLogs.push(`RAG parameter value: ${ragParam}`);
        
        const ragMode = ragParam !== 'false';
        debugLogs.push(`RAG mode enabled: ${ragMode}`);
        
        if (!ragMode) {
            // rag=false，将请求转发到外部 API
            debugLogs.push("Forwarding mode activated, preparing to forward request");
            try {
                const response = await forwardToExternalAPI(url, request, debugLogs);
                
                // 如果启用了调试模式，附加调试日志
                if (debugMode) {
                    // 尝试解析响应内容
                    const originalBody = await response.text();
                    let responseBody;
                    
                    try {
                        // 尝试解析为 JSON
                        responseBody = JSON.parse(originalBody);
                        responseBody.debug = debugLogs;
                        responseBody.forwarded = true;
                        
                        return new Response(
                            JSON.stringify(responseBody),
                            {
                                status: response.status,
                                headers: {
                                    'Content-Type': 'application/json; charset=utf-8',
                                    'Access-Control-Allow-Origin': '*'
                                }
                            }
                        );
                    } catch (e) {
                        // 如果不是 JSON，返回原始内容加上调试信息
                        return new Response(
                            JSON.stringify({
                                original_response: originalBody,
                                debug: debugLogs,
                                forwarded: true
                            }),
                            {
                                status: response.status,
                                headers: {
                                    'Content-Type': response.headers.get('Content-Type') || 'application/json; charset=utf-8',
                                    'Access-Control-Allow-Origin': '*'
                                }
                            }
                        );
                    }
                }
                
                return response;
            } catch (error: unknown) {
                const searchError = error as SearchError;
                debugLogs.push("Forwarding error: " + JSON.stringify({
                    message: searchError.message,
                    stack: searchError.stack,
                    error: error
                }));
                
                return new Response(
                    JSON.stringify({ 
                        error: "Failed to forward request to external API", 
                        message: searchError.message,
                        debug: debugLogs,
                        forwarded_attempt: true
                    }),
                    {
                        status: 502,
                        headers: {
                            'Content-Type': 'application/json; charset=utf-8',
                            'Access-Control-Allow-Origin': '*'
                        }
                    }
                );
            }
        }

        try {
            const query = url.searchParams.get('query');
            const minRatioStr = url.searchParams.get('min_ratio');
            const minRatio = minRatioStr ? parseFloat(minRatioStr) : undefined;
            const minSimilarity = parseFloat(url.searchParams.get('min_similarity') || '0.5');
            const maxResults = parseInt(url.searchParams.get('max_results') || '10');

            if (!query) {
                const resp: ErrorResponse = { error: 'Missing query parameter' };
                if (debugMode) { resp.debug = debugLogs; }
                return new Response(
                    JSON.stringify(resp),
                    {
                        status: 400,
                        headers: {
                            'Content-Type': 'application/json; charset=utf-8',
                            'Access-Control-Allow-Origin': '*'
                        }
                    }
                );
            }

            // maxResults 限制
            if (maxResults > 20) {
                const resp: ErrorResponse = { error: 'max_results cannot exceed 20' };
                if (debugMode) { resp.debug = debugLogs; }
                return new Response(
                    JSON.stringify(resp),
                    {
                        status: 400,
                        headers: {
                            'Content-Type': 'application/json; charset=utf-8',
                            'Access-Control-Allow-Origin': '*'
                        }
                    }
                );
            }

            // 调用 handleSearch 时传入 minRatio 参数
            const results = await handleSearch(query, minRatio, minSimilarity, maxResults, env, debugLogs);
            
            if (debugMode) {
                // 调试模式下保持原格式
                const responseBody = {
                    query,
                    results,
                    total: results.length,
                    debug: debugLogs
                };
                
                return new Response(
                    JSON.stringify(responseBody),
                    {
                        headers: {
                            'Content-Type': 'application/json; charset=utf-8',
                            'Access-Control-Allow-Origin': '*'
                        }
                    }
                );
            } else {
                // 非调试模式下使用 NDJSON 格式
                const ndjson = results
                    .map(result => JSON.stringify(result))
                    .join('\n');
                
                return new Response(
                    ndjson,
                    {
                        headers: {
                            'Content-Type': 'text/plain; charset=utf-8',
                            'Access-Control-Allow-Origin': '*'
                        }
                    }
                );
            }
        } catch (error: unknown) {
            const searchError = error as SearchError;
            debugLogs.push("Error occurred: " + JSON.stringify({
                message: searchError.message,
                code: searchError.code,
                stack: searchError.stack,
                error: error
            }));
            if (debugMode) {
                return new Response(
                    JSON.stringify({ 
                        error: 'Search failed: ' + (searchError.message || 'Unknown error'),
                        details: searchError.code ? `Error code: ${searchError.code}` : undefined,
                        debug: debugLogs
                    }),
                    {
                        status: searchError.code === 40006 ? 400 : 500,
                        headers: {
                            'Content-Type': 'application/json; charset=utf-8',
                            'Access-Control-Allow-Origin': '*'
                        }
                    }
                );
            } else {
                return new Response(
                    JSON.stringify({ error: "Internal Server Error" }),
                    {
                        status: 500,
                        headers: {
                            'Content-Type': 'application/json; charset=utf-8',
                            'Access-Control-Allow-Origin': '*'
                        }
                    }
                );
            }
        }
    }
};
