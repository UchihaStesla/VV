// 调试模式配置
const ALLOW_DEBUG = false; // 控制是否允许调试模式

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
    text: string;
    timestamp: string;
    filename: string;
    text_similarity: number;
    image_similarity: number;
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
                model: 'BAAI/bge-large-zh-v1.5',
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
        embeddingResult = await env.AI.run('@cf/baai/bge-large-en-v1.5', {
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
            text_similarity: v.score,
            image_similarity: v.metadata.image_similarity
        }))
        .filter(v => v.text_similarity >= minSimilarity)
        .sort((a, b) => b.text_similarity - a.text_similarity)
        .slice(0, maxResults)
        .map(v => ({
            ...v,
            text_similarity: Math.round(v.text_similarity * 1000) / 1000,
            image_similarity: Math.round(v.image_similarity * 1000) / 1000
        }));
}

export default {
    async fetch(request: Request, env: Env): Promise<Response> {
        // CORS 预检请求处理
        if (request.method === 'OPTIONS') {
            return new Response(null, {
                headers: {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type'
                }
            });
        }

        // 仅允许 GET 请求
        if (request.method !== 'GET') {
            return new Response('Method not allowed', { status: 405 });
        }

        const debugLogs: string[] = [];
        const url = new URL(request.url);
    // 调试模式配置（根据URL参数控制）
        const debugMode = ALLOW_DEBUG && url.searchParams.get('debug') === 'true';

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
                            'Content-Type': 'application/json',
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
                            'Content-Type': 'application/json',
                            'Access-Control-Allow-Origin': '*'
                        }
                    }
                );
            }

            // 调用 handleSearch 时传入 minRatio 参数
            const results = await handleSearch(query, minRatio, minSimilarity, maxResults, env, debugLogs);
            
            // 根据 debug 模式返回不同格式
            const responseBody = debugMode ? {
                query,
                results,
                total: results.length,
                debug: debugLogs
            } : results;

            return new Response(
                JSON.stringify(responseBody),
                {
                    headers: {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    }
                }
            );
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
                            'Content-Type': 'application/json',
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
                            'Content-Type': 'application/json',
                            'Access-Control-Allow-Origin': '*'
                        }
                    }
                );
            }
        }
    }
};
