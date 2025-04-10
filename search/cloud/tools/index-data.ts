import { readFile, readdir, writeFile } from 'fs/promises';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { createInterface } from 'readline';
import { EventEmitter } from 'events';

// 设置事件监听器上限防止内存泄漏
EventEmitter.defaultMaxListeners = 20;

const __dirname = dirname(fileURLToPath(import.meta.url));

// 全局配置参数集中管理
const CONFIG = {
    // API相关配置
    CF_ACCOUNT_ID: process.env.CLOUDFLARE_ACCOUNT_ID,
    CLOUDFLARE_API_TOKEN: process.env.CLOUDFLARE_API_TOKEN,
    SF_API_KEY: process.env.SF_API_KEY,
    USE_SF_API: false,
    
    // 并发和批处理配置
    MAX_CONCURRENT: 16,         // embedding最大并发请求数
    UPLOAD_CONCURRENT: 8,       // 上传并发数
    MAX_RETRIES: 3,            // 最大重试次数
    RETRY_DELAY: 1000,         // 重试延迟（毫秒）
    
    // 批处理大小配置
    SF_BATCH_SIZE: 32,         // Silicon Flow API批量大小
    CF_BATCH_SIZE: 100,        // Cloudflare API批量大小
    UPLOAD_BATCH_SIZE: 100,    // 向量上传批量大小
    SAVE_INTERVAL: 100,        // 保存进度间隔
    
    // 调试配置
    DEBUG: process.env.DEBUG === '1',

    // 队列长度和检查间隔配置
    MAX_QUEUE_LENGTH: 5000,    // 最大等待上传队列长度
    QUEUE_CHECK_INTERVAL: 1000, // 队列检查间隔（毫秒）
} as const;

interface SubtitleEntry {
    text: string;
    timestamp: string;
    similarity: number;
    id: string;
    filename: string; // 存储视频文件名
}

interface CloudflareResponse<T> {
    success: boolean;
    errors: any[];
    messages: any[];
    result: T;
}

interface CloudflareError {
    code: number;
    message: string;
}

// 错误类型定义
interface CloudflareAPIError extends Error {
    status?: number;
    response?: CloudflareResponse<any>;
    formatted?: string;
}

interface RetryOptions {
    maxRetries?: number;
    delayMs?: number;
    onRetry?: (error: Error, attempt: number) => void;
    context?: string;
}

// 工具函数：格式化 Cloudflare API 错误信息
function formatCloudflareErrors(errors: CloudflareError[]): string {
    return errors.map(err => `[${err.code}] ${err.message}`).join('\n');
}

// 调用 Cloudflare API 并处理响应
async function callCloudflareAPI<T>(path: string, method: string, body?: any): Promise<T> {
    const url = `https://api.cloudflare.com/client/v4${path}`;
    if (CONFIG.DEBUG) {
        console.log(`请求 API: ${method} ${url}`);
        if (body) {
            console.log('请求体:', typeof body === 'string' ? body : JSON.stringify(body, null, 2));
        }
    }

    const response = await fetch(url, {
        method,
        headers: {
            'Authorization': `Bearer ${CONFIG.CLOUDFLARE_API_TOKEN}`,
            'Content-Type': body && typeof body === 'string' 
                ? 'application/x-ndjson'
                : 'application/json',
        },
        body: typeof body === 'string' ? body : body ? JSON.stringify(body) : undefined,
    });

    // 先获取原始响应文本
    const rawText = await response.text();
    if (CONFIG.DEBUG) {
        console.log('原始响应:', rawText);
    }

    let data: CloudflareResponse<T>;
    try {
        data = JSON.parse(rawText);
    } catch (parseError) {
        console.error('解析响应失败:');
        console.error('状态码:', response.status);
        if (CONFIG.DEBUG) {
            console.error('响应头:', Object.fromEntries(response.headers.entries()));
            console.error('原始内容:', rawText);
        }
        throw new Error(`解析响应失败: ${parseError instanceof Error ? parseError.message : String(parseError)}`);
    }

    if (!response.ok || !data.success) {
        const error = new Error('API请求失败') as CloudflareAPIError;
        error.status = response.status;
        error.response = data;
        error.formatted = data.errors ? formatCloudflareErrors(data.errors) : '未知错误';
        throw error;
    }

    return data.result;
}

// 更新响应类型定义
interface EmbeddingResponse {
    data: number[][];
    shape: number[];
}

// Silicon Flow API 响应结构
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

// 并发批处理控制
async function batchConcurrent<T, R>(
    items: T[],
    batchSize: number,
    maxConcurrent: number,
    processor: (batch: T[]) => Promise<R[]>
): Promise<R[]> {
    const results: R[] = [];
    for (let i = 0; i < items.length; i += batchSize * maxConcurrent) {
        const batchPromises = [];
        for (let j = 0; j < maxConcurrent && i + j * batchSize < items.length; j++) {
            const start = i + j * batchSize;
            const batch = items.slice(start, start + batchSize);
            batchPromises.push(processor(batch));
        }
        const batchResults = await Promise.all(batchPromises);
        results.push(...batchResults.flat());
    }
    return results;
}

// 全局确认锁
let confirmationInProgress: Promise<boolean> | null = null;

// 用户操作确认
async function getSharedConfirmation(error: Error, context: string = ''): Promise<boolean> {
    // 如果已经有确认进行中，等待它完成
    if (confirmationInProgress) {
        return await confirmationInProgress;
    }
    
    // 清除所有进度显示
    process.stdout.write('\r\x1b[K');
    console.log('\n'); // 添加空行确保提示清晰可见
    
    // 创建新的确认锁
    confirmationInProgress = new Promise<boolean>((resolve) => {
        const rl = createInterface({
            input: process.stdin,
            output: process.stdout
        });

        console.error(`${context ? `[${context}] ` : ''}错误详情：`);
        if ((error as any).formatted) {
            console.error((error as any).formatted);
        } else {
            console.error(error.message);
            if (CONFIG.DEBUG && error.stack) {
                console.error('\n调用栈：');
                console.error(error.stack);
            }
        }

        rl.question('\n是否继续重试该操作？(Y/n) ', (answer: string) => {
            const normalizedAnswer = answer.trim().toLowerCase();
            rl.close();
            const result = normalizedAnswer === '' || normalizedAnswer === 'y';
            resolve(result);
            // 释放确认锁
            confirmationInProgress = null;
        });
    });
    
    return await confirmationInProgress;
}

// API 调用重试机制
async function withRetry<T>(
    operation: () => Promise<T>,
    options: RetryOptions & { context?: string } = {}
): Promise<T> {
    const maxRetries = options.maxRetries ?? CONFIG.MAX_RETRIES;
    const delay = options.delayMs ?? CONFIG.RETRY_DELAY;
    let attempt = 1;
    let totalAttempts = 0;
    let lastProgressUpdate = 0;
    const progressUpdateInterval = 500; // 限制进度更新频率（毫秒）

    while (true) {
        try {
            return await operation();
        } catch (error) {
            const lastError = error instanceof Error ? error : new Error(String(error));
            
            // 仅在首次失败或达到最大重试次数时显示详细错误
            if (attempt === 1) {
                console.error(`\n操作失败 (尝试 ${attempt}/${maxRetries})`);
            }

            if (attempt >= maxRetries) {
                // 记录当前轮次
                totalAttempts++;
                const roundInfo = totalAttempts > 1 ? ` (轮次 ${totalAttempts})` : '';
                
                // 使用全局确认锁确保不会有多个确认同时显示
                if (await getSharedConfirmation(lastError, options.context)) {
                    attempt = 1;
                    console.log(`开始新一轮尝试${roundInfo}...`);
                    continue;
                } else {
                    console.log('用户选择退出。');
                    process.exitCode = 1;
                    return process.exit();
                }
            } else {
                if (options.onRetry) {
                    options.onRetry(lastError, attempt);
                } else {
                    // 限制进度更新频率，避免过多输出
                    const now = Date.now();
                    if (now - lastProgressUpdate > progressUpdateInterval) {
                        // 使用单行输出，避免过多日志
                        process.stdout.write(`\r\x1b[K操作失败，尝试 ${attempt}/${maxRetries}`);
                        lastProgressUpdate = now;
                    }
                }
                
                attempt++;
                const waitTime = delay * attempt;
                
                // 限制进度更新频率
                await new Promise(resolve => {
                    const startTime = Date.now();
                    const interval = setInterval(() => {
                        const elapsed = Date.now() - startTime;
                        if (elapsed >= waitTime) {
                            clearInterval(interval);
                            resolve(null);
                            return;
                        }
                        
                        const now = Date.now();
                        if (now - lastProgressUpdate > progressUpdateInterval) {
                            const remaining = Math.ceil((waitTime - elapsed) / 1000);
                            process.stdout.write(`\r\x1b[K等待 ${remaining} 秒后重试... (${attempt}/${maxRetries})`);
                            lastProgressUpdate = now;
                        }
                    }, 1000);
                });
                
                // 清除等待消息
                process.stdout.write('\r\x1b[K');
            }
        }
    }
}

// 向量嵌入生成
async function generateEmbeddings(texts: string[]) {
    const SF_MAX_BATCH_SIZE = CONFIG.SF_BATCH_SIZE;  // Silicon Flow API的最大批量大小
    const CF_MAX_BATCH_SIZE = CONFIG.CF_BATCH_SIZE; // Cloudflare API的最大批量大小

    if (CONFIG.USE_SF_API) {
        if (!CONFIG.SF_API_KEY) {
            throw new Error('未设置 SF_API_KEY 环境变量。请设置：\nset SF_API_KEY=你的Silicon Flow API密钥');
        }

        const processBatch = async (batchTexts: string[]): Promise<number[][]> => {
            return await withRetry(async () => {
                const response = await fetch('https://api.siliconflow.cn/v1/embeddings', {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${CONFIG.SF_API_KEY}`,
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        model: 'BAAI/bge-m3',
                        input: batchTexts,
                        encoding_format: 'float'
                    })
                });

                if (!response.ok) {
                    const error = await response.text();
                    throw new Error(`Silicon Flow API请求失败: ${error}`);
                }

                const result = await response.json() as SFEmbeddingResponse;
                return result.data.map(d => d.embedding);
            }, {
                onRetry: (error, attempt) => {
                    console.log(`生成向量嵌入失败，正在进行第${attempt}次重试...`);
                }
            });
        };

        return await batchConcurrent(texts, SF_MAX_BATCH_SIZE, CONFIG.MAX_CONCURRENT, processBatch);
    } else {
        const processBatch = async (batchTexts: string[]): Promise<number[][]> => {
            const response = await callCloudflareAPI<EmbeddingResponse>(
                `/accounts/${CONFIG.CF_ACCOUNT_ID}/ai/run/@cf/baai/bge-m3`,
                'POST',
                { text: batchTexts }
            );
            return response.data;
        };

        return await batchConcurrent(texts, CF_MAX_BATCH_SIZE, CONFIG.MAX_CONCURRENT, processBatch);
    }
}

interface VectorData {
    id: string;
    values: number[];
    metadata: {
        text: string;
        timestamp: string;
        filename: string;
        image_similarity: number;
    };
}

// 数组转换为 NDJSON 格式
function toNDJSON(vectors: VectorData[]): string {
	return vectors.map(vector => JSON.stringify(vector)).join('\n');
}

// 向量数据更新
async function upsertVectors(vectors: VectorData[]) {
    return await withRetry(async () => {
        return await callCloudflareAPI<{ mutationId: string, ids: string[] }>(
            `/accounts/${CONFIG.CF_ACCOUNT_ID}/vectorize/v2/indexes/vv-search/upsert`,
            'POST',
            toNDJSON(vectors)
        );
    }, {
        onRetry: (error, attempt) => {
            console.log(`上传向量失败，正在进行第${attempt}次重试...`);
        }
    });
}

// 向量索引创建与配置
async function createIndex() {
    try {
        // 创建向量索引
        const indexResult = await callCloudflareAPI<any>(
            `/accounts/${CONFIG.CF_ACCOUNT_ID}/vectorize/v2/indexes`,
            'POST',
            {
                name: 'vv-search',
                description: 'Video subtitle search index',
                config: {
                    dimensions: 1024,
                    metric: 'cosine'
                }
            }
        );

        console.log('主索引创建成功，开始创建元数据索引...');

        // 数值类型元数据索引创建
        try {
            await callCloudflareAPI<{ mutationId: string }>(
                `/accounts/${CONFIG.CF_ACCOUNT_ID}/vectorize/v2/indexes/vv-search/metadata_index/create`,
                'POST',
                {
                    propertyName: "image_similarity",
                    indexType: "number"
                }
            );
            console.log('元数据索引创建成功！');
        } catch (metadataError: any) {
            // 处理已存在索引的情况
            if (metadataError?.response?.errors?.some((e: { code: number }) => e.code === 3002)) {
                console.log('元数据索引已存在，跳过创建');
            } else {
                throw metadataError;
            }
        }

        return indexResult;
    } catch (error: any) {
        // 判断是否是索引已存在错误
        if (error?.response?.errors?.some((e: { code: number }) => e.code === 3002)) {
            // 如果主索引已存在，尝试创建元数据索引
            try {
                await callCloudflareAPI<{ mutationId: string }>(
                    `/accounts/${CONFIG.CF_ACCOUNT_ID}/vectorize/v2/indexes/vv-search/metadata_index/create`,
                    'POST',
                    {
                        propertyName: "image_similarity",
                        indexType: "number"
                    }
                );
                console.log('元数据索引创建成功！');
            } catch (metadataError: any) {
                if (metadataError?.response?.errors?.some((e: { code: number }) => e.code === 3002)) {
                    console.log('元数据索引已存在，跳过创建');
                } else {
                    console.warn('创建元数据索引失败，但将继续处理：', metadataError.message);
                }
            }
            return { indexExists: true };
        }
        // 其他错误进入重试逻辑
        return await withRetry(async () => {
            const result = await callCloudflareAPI<any>(
                `/accounts/${CONFIG.CF_ACCOUNT_ID}/vectorize/v2/indexes`,
                'POST',
                {
                    name: 'vv-search',
                    description: 'Video subtitle search index',
                    config: {
                        dimensions: 1024,
                        metric: 'cosine'
                    }
                }
            );

            // 在重试成功后创建元数据索引
            try {
                await callCloudflareAPI<{ mutationId: string }>(
                    `/accounts/${CONFIG.CF_ACCOUNT_ID}/vectorize/v2/indexes/vv-search/metadata_index/create`,
                    'POST',
                    {
                        propertyName: "image_similarity",
                        indexType: "number"
                    }
                );
                console.log('元数据索引创建成功！');
            } catch (metadataError: any) {
                if (metadataError?.response?.errors?.some((e: { code: number }) => e.code === 3002)) {
                    console.log('元数据索引已存在，跳过创建');
                } else {
                    console.warn('创建元数据索引失败，但将继续处理：', metadataError.message);
                }
            }

            return result;
        }, {
            onRetry: (retryError, attempt) => {
                console.log(`创建索引失败，正在进行第${attempt}次重试...`);
            }
        });
    }
}

// API 凭据校验
async function checkApiCredentials() {
    if (!CONFIG.CF_ACCOUNT_ID) {
        throw new Error('未设置 CF_ACCOUNT_ID 环境变量。请运行：\nset CF_ACCOUNT_ID=你的账号ID');
    }
    if (!CONFIG.CLOUDFLARE_API_TOKEN) {
        throw new Error('未设置 CF_API_TOKEN 环境变量。请运行：\nset CF_API_TOKEN=你的API令牌\n\n该令牌需要具有以下权限：\n- Vectorize Write\n- AI: Read/Write');
    }

    // 仅验证令牌有效性
    try {
        await callCloudflareAPI<any>('/user/tokens/verify', 'GET');
        console.log('API Token 验证成功');
    } catch (error: any) {
        let errorMessage = '验证失败：\n';
        if (error.response?.errors) {
            errorMessage += formatCloudflareErrors(error.response.errors);
        } else if (typeof error === 'string') {
            errorMessage += error;
        } else {
            errorMessage += error.message || '未知错误';
        }

        throw new Error(
            `API Token 验证失败\n\n错误详情：\n${errorMessage}\n\n` +
            '可能的原因：\n' +
            '1. API Token 不正确或已过期\n' +
            '2. API Token 权限不足\n\n' +
            '请检查以上问题并重试。'
        );
    }
}

// 批量删除向量数据
async function deleteVectors(ids: string[]) {
    // 每批处理100个ID
    const batchSize = 100;
    console.log(`正在分批删除${ids.length}个向量...`);
    
    for (let i = 0; i < ids.length; i += batchSize) {
        const batch = ids.slice(i, i + batchSize);
        await withRetry(async () => {
            await callCloudflareAPI<{ mutationId: string }>(
                `/accounts/${CONFIG.CF_ACCOUNT_ID}/vectorize/v2/indexes/vv-search/delete_by_ids`,
                'POST',
                { 
                    ids: batch,
                    returnValues: false
                }
            );
        }, {
            onRetry: (error, attempt) => {
                console.log(`删除向量批次${Math.floor(i/batchSize)+1}/${Math.ceil(ids.length/batchSize)}失败，正在进行第${attempt}次重试...`);
            }
        });
        console.log(`已删除: ${Math.min(i + batchSize, ids.length)}/${ids.length}`);
    }
}

// 增量更新索引数据
async function updateIndex(): Promise<{ old_count: number; new_count: number; deleted: number; updated: number; added: number; diff: number; entries: SubtitleEntry[] }> {
    console.log('正在加载当前字幕文件...');
    const subtitleDir = join(__dirname, '../../../subtitle');
    const files = (await readdir(subtitleDir)).filter(f => f.endsWith('.json'));
    
    // 使用Map存储当前条目，以id为键
    const currentEntriesById = new Map<string, SubtitleEntry>();
    const currentEntriesByKey = new Map<string, SubtitleEntry>();
    
    for (const file of files) {
        const content = await readFile(join(subtitleDir, file), 'utf-8');
        const data = JSON.parse(content) as SubtitleEntry[];
        const basename = file.slice(0, -5);
        for (const entry of data) {
            const key = `${basename}-${entry.timestamp}`;
            const fullEntry = { ...entry, filename: basename };
            if (entry.id) {
                currentEntriesById.set(entry.id, fullEntry);
            }
            currentEntriesByKey.set(key, fullEntry);
        }
    }
    
    console.log('正在读取本地entries.json...');
    const localEntriesPath = join(__dirname, 'entries.json');
    let localEntries: SubtitleEntry[] = [];
    try {
        const localData = await readFile(localEntriesPath, 'utf-8');
        localEntries = JSON.parse(localData) as SubtitleEntry[];
    } catch (e) {
        console.warn('未找到entries.json或文件无法读取');
        const rl = createInterface({
            input: process.stdin,
            output: process.stdout
        });
        
        const answer = await new Promise<string>(resolve => {
            rl.question('是否继续处理？这将视为首次运行，所有条目将被视为新增。(Y/n) ', response => {
                rl.close();
                resolve(response.toLowerCase().trim());
            });
        });
        
        if (answer === '' || answer === 'y') {
            console.log('继续处理，所有条目将被视为新增...');
            localEntries = [];
        } else {
            console.log('用户选择退出。');
            process.exit(1);
        }
    }
    
    console.log('正在比较差异...');
    const existingEntriesById = new Map<string, SubtitleEntry>();
    const existingEntriesByKey = new Map<string, SubtitleEntry>();
    
    for (const entry of localEntries) {
        if (entry.id) {
            existingEntriesById.set(entry.id, entry);
        }
        const key = `${entry.filename ?? ''}-${entry.timestamp}`;
        existingEntriesByKey.set(key, entry);
    }
    
    const newEntries: SubtitleEntry[] = [];
    const changedEntries: SubtitleEntry[] = [];
    const incrementalEntries: SubtitleEntry[] = [];
    const toDeleteIds: Set<string> = new Set(existingEntriesById.keys());
    
    // 处理已有ID的条目
    for (const [id, current] of currentEntriesById) {
        toDeleteIds.delete(id);
        const existing = existingEntriesById.get(id);
        
        if (existing) {
            if (existing.text !== current.text) {
                changedEntries.push(current);
                newEntries.push(current);
            } else {
                newEntries.push(existing);
            }
        } else {
            incrementalEntries.push(current);
            newEntries.push(current);
        }
    }
    
    // 处理没有ID的新条目
    for (const [key, current] of currentEntriesByKey) {
        if (!current.id) {
            if (existingEntriesByKey.has(key)) {
                const existing = existingEntriesByKey.get(key)!;
                current.id = existing.id;
                toDeleteIds.delete(existing.id);
                if (existing.text !== current.text) {
                    changedEntries.push(current);
                    newEntries.push(current);
                } else {
                    newEntries.push(existing);
                }
            } else {
                incrementalEntries.push(current);
                newEntries.push(current);
            }
        }
    }

    console.log(`差量计算完成：新增 ${incrementalEntries.length} 条，删除 ${toDeleteIds.size} 条，更改 ${changedEntries.length} 条`);

    // 为新增条目分配ID
    if (incrementalEntries.length > 0) {
        const existingIds = new Set(Array.from(existingEntriesById.keys()));
        for (const entry of incrementalEntries) {
            let newId;
            do {
                newId = generateRandomId();
            } while (existingIds.has(newId));
            entry.id = newId;
            existingIds.add(newId);
        }
    }

    // 添加定期保存机制
    let processed = 0;
    const totalChanges = changedEntries.length + incrementalEntries.length;
    const saveInterval = CONFIG.SAVE_INTERVAL; // 每处理100条保存一次

    // 云端操作：删除已删除的条目
    if (toDeleteIds.size > 0) {
        const deleteIds = Array.from(toDeleteIds);
        console.log(`正在从云端删除${deleteIds.length}个条目...`);
        await deleteVectors(deleteIds);
    }

    // 分批处理变更条目
    if (changedEntries.length > 0) {
        console.log(`正在更新${changedEntries.length}个已变更条目...`);
        const batchSize = 100;
        for (let i = 0; i < changedEntries.length; i += batchSize) {
            const batch = changedEntries.slice(i, i + batchSize);
            const texts = batch.map(e => e.text);
            const embeddings = await generateEmbeddings(texts);
            
            const vectorDataArray = batch.map((entry, idx) => ({
                id: entry.id!,
                values: embeddings[idx],
                metadata: {
                    text: entry.text,
                    timestamp: entry.timestamp,
                    filename: entry.filename ?? '',
                    image_similarity: entry.similarity || 0
                }
            }));

            await upsertVectors(vectorDataArray);
            processed += batch.length;
            console.log(`已更新: ${Math.min(i + batchSize, changedEntries.length)}/${changedEntries.length}`);
            
            // 定期保存进度
            if (processed % saveInterval === 0) {
                await saveEntries(newEntries);
            }
        }
    }

    // 进度追踪：记录已处理和待上传的任务数量
    const batchSize = 100;
    const uploadSemaphore = new Semaphore(CONFIG.UPLOAD_CONCURRENT);
    const uploadPromises: Promise<void>[] = [];

    if (incrementalEntries.length > 0) {
        console.log(`正在处理${incrementalEntries.length}个新增条目...`);
        
        globalState.stats = {
            totalToProcess: incrementalEntries.length,
            embeddingProcessed: 0,
            pendingUploads: 0,
            uploadedCount: 0
        };
        globalState.uploadedEntries.clear(); // 清空已上传条目集合

        const progress = new ProgressDisplay();
        
        for (let i = 0; i < incrementalEntries.length; i += batchSize) {
            // 检查队列长度，如果超过限制则等待
            while (globalState.stats.pendingUploads >= CONFIG.MAX_QUEUE_LENGTH) {
                progress.update(
                    `队列已满，等待上传完成... 待上传: ${globalState.stats.pendingUploads}, ` +
                    `已完成: ${globalState.stats.uploadedCount}/${globalState.stats.totalToProcess}`
                );
                await new Promise(resolve => setTimeout(resolve, CONFIG.QUEUE_CHECK_INTERVAL));
            }

            const batch = incrementalEntries.slice(i, i + batchSize);
            const texts = batch.map(e => e.text);
            
            globalState.stats.pendingUploads += batch.length;
            
            const embeddings = await generateEmbeddings(texts);
            globalState.stats.embeddingProcessed += batch.length;
            
            const vectorDataArray = batch.map((entry, idx) => ({
                id: entry.id!,
                values: embeddings[idx],
                metadata: {
                    text: entry.text,
                    timestamp: entry.timestamp,
                    filename: entry.filename ?? '',
                    image_similarity: entry.similarity || 0
                }
            }));

            const uploadPromise = (async () => {
                await uploadSemaphore.acquire();
                try {
                    await withRetry(
                        async () => {
                            await upsertVectors(vectorDataArray);
                        },
                        {
                            maxRetries: 3,
                            delayMs: 2000,
                            context: `批次上传(${batch.length}条)`
                        }
                    );

                    globalState.stats.uploadedCount += batch.length;
                    globalState.stats.pendingUploads -= batch.length;
                    batch.forEach(entry => globalState.uploadedEntries.add(entry.id!));
                } catch (error) {
                    console.error(`批次上传失败(${batch.length}条):`, error);
                    throw error;
                } finally {
                    uploadSemaphore.release();
                }
            })();

            uploadPromises.push(uploadPromise);

            progress.update(
                `生成进度: ${globalState.stats.embeddingProcessed}/${globalState.stats.totalToProcess}, ` +
                `待上传: ${globalState.stats.pendingUploads}, ` +
                `已完成: ${globalState.stats.uploadedCount}`
            );
        }

        // 等待所有上传完成
        await Promise.all(uploadPromises);
        
        // 最后一次保存，确保包含所有成功上传的条目
        await saveEntries(newEntries);
        
        progress.clear();
        console.log('\n所有上传任务完成');
    }
    
    return {
        old_count: existingEntriesById.size,
        new_count: newEntries.length,
        deleted: toDeleteIds.size,
        updated: changedEntries.length,
        added: incrementalEntries.length,
        diff: newEntries.length - existingEntriesById.size,
        entries: newEntries
    };
}

// 添加一个生成随机ID的辅助函数
function generateRandomId(): string {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    const length = 20;
    return Array.from(crypto.getRandomValues(new Uint8Array(length)))
        .map(x => chars[x % chars.length])
        .join('');
}

// 文件写入锁实现
class FileLock {
    private isLocked = false;
    private queue: (() => void)[] = [];

    async acquire(): Promise<void> {
        if (!this.isLocked) {
            this.isLocked = true;
            return;
        }
        return new Promise<void>(resolve => {
            this.queue.push(resolve);
        });
    }

    release(): void {
        if (this.queue.length > 0) {
            const next = this.queue.shift();
            next?.();
        } else {
            this.isLocked = false;
        }
    }
}

// 持久化保存 entries 数据
async function saveEntries(entries: SubtitleEntry[]) {
    await globalState.fileLock.acquire();
    try {
        const entriesPath = join(__dirname, 'entries.json');
        await writeFile(entriesPath, JSON.stringify(entries, null, 2), 'utf-8');
    } finally {
        globalState.fileLock.release();
    }
}

// 从持久化存储加载 entries 数据
async function loadExistingEntries(): Promise<SubtitleEntry[]> {
    const entriesPath = join(__dirname, 'entries.json');
    try {
        const content = await readFile(entriesPath, 'utf-8');
        return JSON.parse(content) as SubtitleEntry[];
    } catch (e: unknown) {
        return [];
    }
}

// 信号量实现类
class Semaphore {
    private permits: number;
    private waiting: Array<() => void> = [];

    constructor(permits: number) {
        this.permits = permits;
    }

    async acquire(): Promise<void> {
        if (this.permits > 0) {
            this.permits--;
            return Promise.resolve();
        }
        return new Promise<void>(resolve => {
            this.waiting.push(resolve);
        });
    }

    release(): void {
        this.permits++;
        const next = this.waiting.shift();
        if (next) {
            this.permits--;
            next();
        }
    }
}

// 全局状态管理接口
interface IndexState {
    currentEntries: SubtitleEntry[];
    stats: {
        totalToProcess: number;
        embeddingProcessed: number;
        pendingUploads: number;
        uploadedCount: number;
    };
    uploadedEntries: Set<string>;  // 存储已成功上传的条目ID
    fileLock: FileLock;           // 文件写入锁
}

// 全局状态实例
const globalState: IndexState = {
    currentEntries: [],
    stats: {
        totalToProcess: 0,
        embeddingProcessed: 0,
        pendingUploads: 0,
        uploadedCount: 0
    },
    uploadedEntries: new Set<string>(),
    fileLock: new FileLock()
};

// 进度显示管理器
class ProgressDisplay {
    private lastUpdate: number = 0;
    private readonly updateInterval: number = 500; // 增加最小更新间隔（毫秒）以减少闪烁
    private lastMessage: string = '';

    constructor(private readonly clearLine: boolean = true) {}

    update(message: string) {
        // 如果有确认对话框正在显示，不更新进度
        if (confirmationInProgress) {
            return;
        }
        
        const now = Date.now();
        // 只有当消息变化或达到更新间隔时才更新显示
        if ((now - this.lastUpdate >= this.updateInterval) || (message !== this.lastMessage)) {
            if (this.clearLine) {
                process.stdout.write('\r\x1b[K');
            }
            process.stdout.write(`\r${message}`);
            this.lastUpdate = now;
            this.lastMessage = message;
        }
    }

    clear() {
        if (this.clearLine) {
            process.stdout.write('\r\x1b[K');
        }
        this.lastMessage = '';
    }
}

// 主程序入口：索引数据处理 
async function indexData() {
    // 初始化全局状态
    globalState.currentEntries = await loadExistingEntries();
    
    // 程序退出信号处理
    let isShuttingDown = false;
    async function handleShutdown() {
        if (isShuttingDown) return;
        isShuttingDown = true;
        
        console.log('\n正在保存进度...');
        try {
            // 持久化当前处理进度
            await saveEntries(globalState.currentEntries);
            console.log('进度已保存，程序退出');
            process.exitCode = 0;
        } catch (error) {
            console.error('保存进度时发生错误:', error);
            process.exitCode = 1;
        } finally {
            process.exit();
        }
    }

    // 注册系统信号处理函数
    process.on('SIGINT', handleShutdown);
    process.on('SIGTERM', handleShutdown);

    console.log('正在验证API凭据...');
    await checkApiCredentials();
    
    // 创建或检查索引
    console.log('正在初始化索引...');
    const indexResult = await createIndex();
    if ('indexExists' in indexResult) {
        console.log('检测到现有索引，继续处理...');
    } else {
        console.log('索引创建完成，继续处理...');
    }
    
    // 差量更新处理
    const stats = await updateIndex();
    // 更新全局状态
    globalState.currentEntries = stats.entries;
    await saveEntries(stats.entries);
    console.log(`\n索引更新完成:`);
    console.log(`- 原有条目数: ${stats.old_count}`);
    console.log(`- 当前条目数: ${stats.new_count}`);
    console.log(`- 新增条目数: ${stats.added}`);
    console.log(`- 删除条目数: ${stats.deleted}`);
    console.log(`- 更新条目数: ${stats.updated}`);
    return stats;
}

// 简化命令行处理
indexData().catch(console.error);
