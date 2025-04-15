const ALLOW_DEBUG = false; 
const EXTERNAL_API_URL = 'https://vv-api.vercel.app/search';

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

interface VectorMetadata {
    text: string;
    timestamp: string;
    filename: string;
    image_similarity: number;
}

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
    SF_API_KEY?: string; 
}

interface SearchResult {
    filename: string;
    timestamp: string;
    match_ratio: number;
    text: string;
    similarity: number;
}

interface SearchError extends Error {
    code?: number;
}

interface ErrorResponse {
    error: string;
    debug?: string[];
}

interface AIProcessResult {
    type: string;
    keywords: string[];
    answer: string;
    search_keywords: string;
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
        debugLogs.push("Using Cloudflare AI API for embeddings");
        embeddingResult = await env.AI.run('@cf/baai/bge-m3', {
            text: [query]
        });
    }
    
    debugLogs.push("Embedding result: " + JSON.stringify(embeddingResult));
    
    if (!embeddingResult?.data?.[0] || embeddingResult.data[0].length !== 1024) {
        throw new Error('Invalid embedding vector generated');
    }
    
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

async function processWithAI(query: string, env: Env, debugLogs: string[], webSearchResults?: any[]): Promise<AIProcessResult> {
    debugLogs.push("开始调用 AI 处理搜索结果");
    
    if (!env.SF_API_KEY) {
        debugLogs.push("未配置 SF_API_KEY，无法调用 AI 服务");
        return {
            type: "keywords",
            keywords: [query],
            answer: "AI 服务未配置，无法处理",
            search_keywords: query
        };
    }
    
    try {
        let systemMessage = `你是一个专业的问答助手和关键词提取器。你有两个任务：
            1. 根据用户的问题，生成一个简洁明了的回答。
            2. 从问题中提取5-8个最核心的关键词（中文），用于概括主题。

            回答问题时：
            - 基于事实和提供的信息进行总结。
            - 保持回答简洁、信息丰富。
            - 如果信息不足或不确定，请说明。

            提取关键词时：
            - 关键词必须是中文。
            - 提取5-8个最能代表核心主题，最好带有评价性的词语。
            - 用顿号"、"分隔关键词。

            你的回复格式必须严格遵循：
            1. 先回答问题（一段文字）。
            2. 然后必须空一行。
            3. 最后一行以"关键词："开头，后跟用顿号分隔的关键词列表。

            例如：
            这是一个关于某事的回答总结...\n
            关键词：关键词1、关键词2、关键词3`;
        
        let userMessage = `请根据以下问题，回答问题并提取关键词：问题: ${query}`;
        
        if (webSearchResults && webSearchResults.length > 0) {
            debugLogs.push(`提供 ${webSearchResults.length} 条联网搜索结果作为上下文`);
            
            const webSearchText = webSearchResults.map((result, index) => {
                return `[${index + 1}] ${result.title}\n${result.body}\n链接: ${result.href}\n`;
            }).join('\n');
            
            systemMessage = `你是一个专业的问答助手和关键词提取器。你有两个任务：
                1. 根据用户的问题和提供的联网搜索结果，生成一个简洁明了的回答。
                2. 从问题中提取5-8个最核心的关键词（中文），用于概括主题。

                回答问题时：
                - 基于提供的联网搜索结果和你的知识进行总结。
                - 保持回答简洁、信息丰富。
                - 如果信息不足或不确定，请说明。

                提取关键词时：
                - 关键词必须是中文。
                - 提取5-8个最能代表核心主题，最好带有评价性的词语。
                - 用顿号"、"分隔关键词。

                你的回复格式必须严格遵循：
                1. 先回答问题（一段文字）。
                2. 然后必须空一行。
                3. 最后一行以"关键词："开头，后跟用顿号分隔的关键词列表。

                例如：
                这是一个关于某事的回答总结...\n
                关键词：关键词1、关键词2、关键词3`;
            
            userMessage = `请根据以下问题和联网搜索结果，回答问题并提取关键词：

问题: ${query}

联网搜索结果:
${webSearchText}`;
        }
        
        const messages = [
            { role: "system", content: systemMessage },
            { role: "user", content: userMessage }
        ];
        
        const payload = {
            model: "Qwen/Qwen2.5-7B-Instruct",
            messages: messages,
            temperature: 0.4,
            max_tokens: 800,
            stream: false
        };
        
        debugLogs.push(`向 AI API 发送请求: ${JSON.stringify(payload)}`);
        
        const response = await fetch('https://api.siliconflow.cn/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${env.SF_API_KEY}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            debugLogs.push(`AI API 返回错误: ${response.status}, ${errorText}`);
            throw new Error(`AI API 请求失败: ${response.status} ${errorText}`);
        }
        
        const result = await response.json();
        debugLogs.push(`AI API 响应: ${JSON.stringify(result)}`);
        
        // 检查响应格式
        if (!result.choices || !Array.isArray(result.choices) || result.choices.length === 0) {
            debugLogs.push("AI API 响应格式不正确，缺少 choices 数组");
            throw new Error("AI API 响应格式不正确");
        }
        
        const message = result.choices[0].message;
        if (!message || !message.content) {
            debugLogs.push("AI API 响应格式不正确，缺少 message.content");
            throw new Error("AI API 响应格式不正确，缺少内容");
        }
        
        const fullResponse = message.content;
        debugLogs.push(`AI 原始响应 (前300字符): ${fullResponse.substring(0, 300)}...`);
        
        let answer = "";
        let keywordsPartRaw = "";
        
        try {
            const keywordsRegex = /关键词：([^\\n]+)/;
            const keywordsMatch = fullResponse.match(keywordsRegex);
            
            if (keywordsMatch && keywordsMatch[1]) {
                // 提取关键词部分
                keywordsPartRaw = keywordsMatch[1].trim();
                debugLogs.push(`使用正则表达式找到关键词部分: ${keywordsPartRaw}`);
                
                // 提取回答部分（关键词之前的所有内容）
                answer = fullResponse.substring(0, fullResponse.indexOf("关键词：")).trim();
                debugLogs.push(`提取的回答: ${answer.substring(0, 50)}...`);
            } else {
                // 如果没有找到关键词部分，尝试其他方法
                debugLogs.push("正则表达式未找到关键词部分，尝试其他方法");
                
                // 尝试按双换行符分割
                const parts = fullResponse.split("\n\n");
                debugLogs.push(`分割后的部分数量: ${parts.length}`);
                
                // 第一部分是回答
                if (parts.length > 0 && parts[0]) {
                    answer = parts[0].trim();
                    debugLogs.push(`提取的回答: ${answer.substring(0, 50)}...`);
                } else {
                    debugLogs.push("无法提取回答部分");
                    throw new Error("无法提取回答部分");
                }
                
                // 查找包含"关键词："的部分
                let foundKeywords = false;
                for (let i = 1; i < parts.length; i++) {
                    if (!parts[i]) continue;
                    
                    const lines = parts[i].split('\n');
                    for (const line of lines) {
                        if (line && line.trim().startsWith("关键词：")) {
                            keywordsPartRaw = line.trim().split("关键词：", 1)[1].trim();
                            foundKeywords = true;
                            debugLogs.push(`找到关键词部分: ${keywordsPartRaw}`);
                            break;
                        }
                    }
                    if (foundKeywords) break;
                }
                
                // 如果没有找到关键词部分，尝试在整个响应中查找
                if (!foundKeywords) {
                    debugLogs.push("在分割部分中未找到关键词，尝试在整个响应中查找");
                    const allLines = fullResponse.split('\n');
                    for (const line of allLines) {
                        if (line && line.trim().startsWith("关键词：")) {
                            keywordsPartRaw = line.trim().split("关键词：", 1)[1].trim();
                            foundKeywords = true;
                            debugLogs.push(`在整个响应中找到关键词部分: ${keywordsPartRaw}`);
                            break;
                        }
                    }
                }
                
                // 如果仍然没有找到关键词，使用默认值
                if (!foundKeywords) {
                    debugLogs.push("无法从AI响应解析出关键词部分，使用原始查询作为后备");
                    keywordsPartRaw = query;
                }
            }
        } catch (parseError) {
            debugLogs.push(`解析AI响应时出错: ${parseError}`);
            answer = fullResponse.trim();
            keywordsPartRaw = query;
        }
        
        const keywordsList = keywordsPartRaw.split("、").map(kw => kw.trim()).filter(kw => kw);
        debugLogs.push(`提取的关键词列表: ${JSON.stringify(keywordsList)}`);
        
        const searchKeywords = keywordsList.slice(0, 3).join(" ");
        
        debugLogs.push(`AI 处理完成。提取的关键词: '${keywordsPartRaw}', 用于搜索的关键词: '${searchKeywords}'`);
        
        return {
            type: "keywords",
            keywords: keywordsList,
            answer: answer,
            search_keywords: searchKeywords || query
        };
    } catch (error) {
        debugLogs.push(`AI 处理出错: ${error}`);
        return {
            type: "keywords",
            keywords: [query],
            answer: `AI 处理出错: ${error}`,
            search_keywords: query
        };
    }
}

// 转发请求到外部 API
async function forwardToExternalAPI(
    url: URL, 
    request: Request, 
    debugLogs: string[],
    aiSearch: boolean = false
): Promise<Response> {
    const externalUrl = new URL(EXTERNAL_API_URL);
    
    const query = url.searchParams.get('query');
    
    if (query) {
        externalUrl.searchParams.append('query', query);
    }
    
    if (aiSearch) {
        externalUrl.searchParams.append('ai_search', 'true');
    }
    
    debugLogs.push(`Forwarding request to: ${externalUrl.toString()}`);
    
    const forwardRequest = new Request(externalUrl.toString(), {
        method: request.method,
        headers: request.headers,
    });
    
    debugLogs.push(`Forward request created with URL: ${forwardRequest.url}`);
    
    const response = await fetch(forwardRequest);
    
    debugLogs.push(`Received response from external API with status: ${response.status}`);
    
    try {
        const clonedResponse = response.clone();
        const responseText = await clonedResponse.text();
        debugLogs.push(`Response body (first 200 chars): ${responseText.substring(0, 200)}`);
        
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

        if (request.method !== 'GET') {
            return new Response('Method not allowed', { status: 405 });
        }

        const debugLogs: string[] = [];
        const url = new URL(request.url);
        
        // 调试模式配置
        const debugMode = url.searchParams.get('debug') === 'true';
        debugLogs.push(`Request URL: ${url.toString()}`);
        debugLogs.push(`Query Parameters: ${JSON.stringify(Object.fromEntries(url.searchParams.entries()))}`);
        
        // 检查是否需要将请求转发到外部 API
        const ragParam = url.searchParams.get('rag');
        debugLogs.push(`RAG parameter value: ${ragParam}`);
        
        const ragMode = ragParam === 'true';
        debugLogs.push(`RAG mode enabled: ${ragMode}`);
        
        const query = url.searchParams.get('query');
        
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
        
        if (!ragMode) {
            debugLogs.push("Forwarding mode activated, preparing to forward request");
            try {
                const externalUrl = new URL(EXTERNAL_API_URL);
                
                for (const [key, value] of url.searchParams.entries()) {
                    if (key !== 'rag') {
                        externalUrl.searchParams.append(key, value);
                    }
                }
                
                debugLogs.push(`Forwarding request to: ${externalUrl.toString()}`);
                
                const forwardRequest = new Request(externalUrl.toString(), {
                    method: request.method,
                    headers: request.headers,
                });
                
                const response = await fetch(forwardRequest);
                
                debugLogs.push(`Received response from external API with status: ${response.status}`);
                
                if (debugMode) {
                    const originalBody = await response.text();
                    let responseBody;
                    
                    try {
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
                
                return new Response(response.body, {
                    status: response.status,
                    statusText: response.statusText,
                    headers: {
                        'Content-Type': response.headers.get('Content-Type') || 'application/json; charset=utf-8',
                        'Access-Control-Allow-Origin': '*'
                    }
                });
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
        } else {
            debugLogs.push("RAG mode activated, fetching web search results first");
            try {
                if (query) {
                    // 先向外部API发送请求获取联网搜索结果
                    const externalUrl = new URL(EXTERNAL_API_URL);
                    externalUrl.searchParams.append('query', query);
                    externalUrl.searchParams.append('ai_search', 'true');
                    
                    debugLogs.push(`Fetching web search results from: ${externalUrl.toString()}`);
                    
                    const webSearchResponse = await fetch(externalUrl.toString(), {
                        method: 'GET',
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    });
                    
                    if (!webSearchResponse.ok) {
                        const errorText = await webSearchResponse.text();
                        debugLogs.push(`Web search API returned error: ${webSearchResponse.status}, ${errorText}`);
                        throw new Error(`Web search API request failed: ${webSearchResponse.status} ${errorText}`);
                    }
                    
                    // 解析联网搜索结果
                    let webSearchResults;
                    try {
                        webSearchResults = await webSearchResponse.json();
                        debugLogs.push(`Received ${webSearchResults.length} web search results`);
                    } catch (parseError) {
                        debugLogs.push(`Failed to parse web search results: ${parseError}`);
                        webSearchResults = [];
                    }
                    
                    // 调用AI处理，传入联网搜索结果
                    const aiResult = await processWithAI(query, env, debugLogs, webSearchResults);
                    debugLogs.push(`AI processing result: ${JSON.stringify(aiResult)}`);
                    
                    // 构建最终响应
                    const finalResponse = {
                        type: "keywords",
                        keywords: aiResult.keywords,
                        answer: aiResult.answer,
                        search_keywords: aiResult.search_keywords
                    };
                    
                    // 根据关键词进行向量搜索 - 使用并发查询
                    debugLogs.push(`Starting concurrent vector search for ${aiResult.keywords.length} keywords`);
                    
                    // 创建一个函数来处理单个关键词的向量搜索
                    const searchKeyword = async (keyword: string): Promise<SearchResult[]> => {
                        try {
                            debugLogs.push(`Performing vector search for keyword: ${keyword}`);
                            
                            let embeddingResult;
                            
                            if (env.SF_API_KEY) {
                                debugLogs.push(`Using Silicon Flow API for keyword: ${keyword}`);
                                const response = await fetch('https://api.siliconflow.cn/v1/embeddings', {
                                    method: 'POST',
                                    headers: {
                                        'Authorization': `Bearer ${env.SF_API_KEY}`,
                                        'Content-Type': 'application/json',
                                    },
                                    body: JSON.stringify({
                                        model: 'BAAI/bge-m3',
                                        input: [keyword],
                                        encoding_format: 'float'
                                    })
                                });

                                if (!response.ok) {
                                    const error = await response.text();
                                    debugLogs.push(`Silicon Flow API failed for keyword ${keyword}: ${error}`);
                                    return [];
                                }

                                const sfResult = await response.json() as SFEmbeddingResponse;
                                embeddingResult = { data: [sfResult.data[0].embedding] };
                            } else {
                                debugLogs.push(`Using Cloudflare AI API for keyword: ${keyword}`);
                                embeddingResult = await env.AI.run('@cf/baai/bge-m3', {
                                    text: [keyword]
                                });
                            }
                            
                            if (!embeddingResult?.data?.[0] || embeddingResult.data[0].length !== 1024) {
                                debugLogs.push(`Invalid embedding vector generated for keyword ${keyword}`);
                                return [];
                            }
                            
                            // 查询向量数据库
                            const queryOptions = {
                                topK: 5, // 每个关键词最多返回5条结果
                                returnMetadata: "all"
                            };
                            
                            const queryResult = await env.SUBTITLE_INDEX.query(embeddingResult.data[0], queryOptions);
                            debugLogs.push(`Vector query result for keyword ${keyword}: ${JSON.stringify(queryResult)}`);
                            
                            return queryResult.matches
                                .map(v => ({
                                    text: v.metadata.text,
                                    timestamp: v.metadata.timestamp,
                                    filename: v.metadata.filename,
                                    match_ratio: v.score,
                                    similarity: v.metadata.image_similarity
                                }))
                                .filter(v => v.match_ratio >= 0.5)
                                .sort((a, b) => b.match_ratio - a.match_ratio)
                                .slice(0, 5) // 确保每个关键词最多返回5条结果
                                .map(v => ({
                                    ...v,
                                    match_ratio: Math.round(v.match_ratio * 1000) / 1000,
                                    similarity: Math.round(v.similarity * 1000) / 1000
                                }));
                        } catch (error) {
                            debugLogs.push(`Error processing keyword ${keyword}: ${error}`);
                            return [];
                        }
                    };
                    
                    // 并发执行所有关键词的向量搜索
                    const searchResultsArrays = await Promise.all(
                        aiResult.keywords.map(keyword => searchKeyword(keyword))
                    );
                    
                    // 合并所有结果
                    const searchResults: SearchResult[] = searchResultsArrays.flat();
                    debugLogs.push(`Concurrent search completed, found ${searchResults.length} total results`);
                    
                    // 将搜索结果添加到响应中
                    if (debugMode) {
                        (finalResponse as any).debug = debugLogs;
                        (finalResponse as any).search_results = searchResults;
                        (finalResponse as any).web_search_results = webSearchResults;
                    }
                    
                    const ndjson = [
                        JSON.stringify(finalResponse),
                        ...searchResults.map(result => JSON.stringify(result))
                    ].join('\n');
                    
                    return new Response(
                        ndjson,
                        {
                            headers: {
                                'Content-Type': 'text/plain; charset=utf-8',
                                'Access-Control-Allow-Origin': '*'
                            }
                        }
                    );
                } else {
                    throw new Error("Query parameter is required for RAG mode");
                }
            } catch (error: unknown) {
                const searchError = error as SearchError;
                debugLogs.push("RAG processing error: " + JSON.stringify({
                    message: searchError.message,
                    stack: searchError.stack,
                    error: error
                }));
                
                return new Response(
                    JSON.stringify({ 
                        error: "Failed to process with AI", 
                        message: searchError.message,
                        debug: debugLogs,
                        rag_mode: true
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
    }
};
