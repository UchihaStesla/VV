import os
import json
import faiss
import logging
import numpy as np
import aiohttp
import asyncio
import secrets
import string
import traceback
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from rich.console import Console
from rich.prompt import Prompt, FloatPrompt, IntPrompt
from rich.panel import Panel
from rich.table import Table
from typing import Dict, Any, List, Optional

class CloudflareAPIError(Exception):
    """Cloudflare API错误"""
    def __init__(self, message: str, status: Optional[int] = None, 
                 response: Optional[Dict] = None, errors: Optional[List[Dict]] = None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.response = response or {}
        self.errors = errors or []
        
        # 如果提供了完整响应，解析其中的错误信息
        if response:
            if not response.get('success', False):
                self.errors.extend(response.get('errors', []))
                messages = response.get('messages', [])
                if not self.errors and messages:
                    self.errors.append({
                        'message': messages[0],
                        'code': 'API_ERROR'
                    })
            
            # 检查mutationId
            result = response.get('result', {})
            if result and not result.get('mutationId'):
                self.errors.append({
                    'message': 'Missing mutationId in response',
                    'code': 'MISSING_MUTATION_ID'
                })

    @classmethod
    def from_response(cls, response: Dict, status: int) -> 'CloudflareAPIError':
        """从API响应创建错误实例"""
        messages = response.get('messages', [])
        message = messages[0] if messages else 'API请求失败'
        return cls(message=message, status=status, response=response)

    def has_error_code(self, code: int) -> bool:
        """检查是否包含特定错误码"""
        return any(e.get('code') == code for e in self.errors)

    def is_rate_limit_error(self) -> bool:
        """检查是否是频率限制错误"""
        return self.status == 429

    def is_payload_too_large(self) -> bool:
        """检查是否是请求体过大错误"""
        return self.status == 413
        
    def is_special_error(self) -> bool:
        """检查是否是特殊错误(无需重试)"""
        # 检查错误代码是否在特殊错误码字典的值中
        return any(e.get('code') in CF_ERROR_CODES.values() for e in self.errors)

    def __str__(self) -> str:
        """返回格式化的错误信息"""
        parts = [self.message]
        if self.status:
            parts.append(f"状态码: {self.status}")
        if self.errors:
            parts.append("错误详情:")
            for err in self.errors:
                error_msg = err.get('message', '未知错误')
                if 'code' in err:
                    error_msg += f" (错误码: {err['code']})"
                parts.append(f"- {error_msg}")
        return "\n".join(parts)

from mapping import get_video_url

# 配置参数
USE_GPU_SEARCH = False    # 是否在SentenceTransformer中使用GPU
SEARCH_BATCH_SIZE = 256   # 索引构建时的批处理大小
MAX_RETRIES = 3          # 最大重试次数
CF_BATCH_SIZE = 100      # Cloudflare批量上传大小
CF_MAX_CONCURRENT = 8     # Cloudflare API并发连接数限制
CF_BACKOFF_BASE = 2       # 指数退避基数（秒）

# Cloudflare错误码字典
CF_ERROR_CODES = {
    'INDEX_EXISTS': 3002,         # 索引已存在
    'INDEX_DELETED': 3005,        # 索引已被删除
    'INDEX_NOT_FOUND': 10010      # 索引不存在 
}

# 配置日志和控制台
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)
console = Console()

@dataclass
class SubtitleEntry:
    text: str
    timestamp: str
    filename: str
    image_similarity: float = 0.0  # 添加图像相似度字段
    id: str = None

class SubtitleSearch:
    def __init__(self, subtitle_folder, model_name='BAAI/bge-m3'):
        self.subtitle_folder = subtitle_folder
        self.use_gpu = USE_GPU_SEARCH
        self.model = SentenceTransformer(model_name, device='cuda' if self.use_gpu else 'cpu')
        self.entries = []
        self.min_image_similarity = 0.6
        self.search_k = 5
        self.index = None
        
        # Cloudflare API配置
        self.cf_account_id = os.getenv('CLOUDFLARE_ACCOUNT_ID')
        self.cf_api_token = os.getenv('CLOUDFLARE_API_TOKEN')
        self.cf_batch_size = CF_BATCH_SIZE

    async def call_cloudflare_api(self, path: str, method: str, body: Any = None, is_ndjson: bool = False) -> Dict:
        """调用 Cloudflare API"""
        if not self.cf_api_token:
            raise ValueError("未提供 API token")

        url = f'https://api.cloudflare.com/client/v4{path}'
        headers = {
            'Authorization': f'Bearer {self.cf_api_token}',
            'Content-Type': 'application/x-ndjson' if is_ndjson else 'application/json'
        }
        
        async def make_request():
            # 记录请求详情
            request_log = {
                'url': url,
                'method': method,
                'content_type': headers['Content-Type'],
                'body_size': len(body) if isinstance(body, str) else (len(json.dumps(body)) if body else 0)
            }
            logging.debug(f"请求API: {request_log}")
            
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.request(
                        method, 
                        url, 
                        headers=headers,
                        data=body if is_ndjson else json.dumps(body) if body else None
                    ) as response:
                        status = response.status
                        raw_text = await response.text()
                        response_headers = dict(response.headers)
                        
                        # 记录响应详情（调试模式）
                        logging.debug(f"API响应状态: {status}")
                        if len(raw_text) > 1000:
                            logging.debug(f"API响应(截断): {raw_text[:1000]}...")
                        else:
                            logging.debug(f"API响应: {raw_text}")

                        if not raw_text:
                            raise CloudflareAPIError(
                                message="空响应",
                                status=status,
                                errors=[
                                    {'message': f"HTTP状态码: {status}"},
                                    {'message': f"响应头: {response_headers}"}
                                ]
                            )

                        try:
                            data = json.loads(raw_text)
                        except json.JSONDecodeError as e:
                            raise CloudflareAPIError(
                                message=f"无效的JSON响应: {str(e)}",
                                status=status,
                                errors=[
                                    {'message': f"解析错误: {str(e)}", 'code': 'JSON_PARSE_ERROR'},
                                    {'message': f"响应头: {response_headers}"},
                                    {'message': f"原始响应(截断): {raw_text[:500]}...", 'code': 'RAW_RESPONSE'}
                                ]
                            )

                        # 检查响应是否有效
                        if not data.get('success', False):
                            raise CloudflareAPIError.from_response(data, status)

                        return data.get('result', {})
                except aiohttp.ClientError as e:
                    # 捕获网络连接相关错误
                    raise CloudflareAPIError(
                        message=f"网络请求错误: {str(e)}",
                        status=0,
                        errors=[
                            {'message': f"客户端错误: {str(e)}", 'code': 'CLIENT_ERROR'},
                            {'message': f"URL: {url}", 'code': 'URL'},
                            {'message': f"方法: {method}", 'code': 'METHOD'}
                        ]
                    )
        
        async def execute_with_retry():
            attempt = 1
            while True:
                try:
                    return await make_request()
                except CloudflareAPIError as e:
                    # 使用is_special_error方法检查是否是特殊错误(无需重试)
                    if e.is_special_error():
                        raise
                    
                    # 其他错误进入重试逻辑
                    if attempt >= MAX_RETRIES:
                        error_msg = f"\n[red]操作失败 (尝试 {attempt}/{MAX_RETRIES})[/]"
                        if e.status:
                            error_msg += f"\n[yellow]HTTP状态码: {e.status}[/]"
                        if e.errors:
                            error_msg += "\n[red]错误详情:[/]"
                            for err in e.errors:
                                error_msg += f"\n[red]- {err.get('message', '未知错误')}[/]"
                                if 'code' in err:
                                    error_msg += f" [yellow](错误码: {err['code']})[/]"
                        error_msg += f"\n\n[red]{e.message}[/]"
                        
                        console.print(error_msg) 
                        if Prompt.ask("\n是否继续重试？", choices=['y', 'n'], default='y', show_choices=False) == 'y':
                            attempt = 1
                            continue
                        raise CloudflareAPIError(
                            message=f"多次重试后失败: {e.message}",
                            status=e.status,
                            errors=e.errors
                        )

                    console.print(f"\r[yellow]操作失败 ({e.message})，尝试 {attempt}/{MAX_RETRIES}...[/]")
                    # 使用CF_BACKOFF_BASE进行指数退避
                    await asyncio.sleep(CF_BACKOFF_BASE ** attempt)
                    attempt += 1

        return await execute_with_retry()

    def load_cloud_entries(self) -> Dict:
        """加载云端同步状态"""
        cloud_entries_path = os.path.join(os.path.dirname(__file__), 'cloud_entries.json')
        try:
            with open(cloud_entries_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "entries": {
                    "synced": [],
                    "pending": {
                        "update": [],
                        "delete": []
                    }
                }
            }

    def save_cloud_entries(self, data: Dict):
        """保存云端同步状态"""
        cloud_entries_path = os.path.join(os.path.dirname(__file__), 'cloud_entries.json')
        with open(cloud_entries_path, 'w') as f:
            json.dump(data, f, indent=2)

    async def delete_cloudflare_index(self):
        """删除Cloudflare向量索引"""
        if not self.cf_account_id or not self.cf_api_token:
            raise ValueError('未设置 Cloudflare API 凭据')
            
        try:
            await self.call_cloudflare_api(
                f'/accounts/{self.cf_account_id}/vectorize/v2/indexes/vv-search',
                'DELETE'
            )
            
            # 删除成功后询问是否删除cloud_entries.json
            cloud_entries_path = os.path.join(os.path.dirname(__file__), 'cloud_entries.json')
            if os.path.exists(cloud_entries_path):
                if Prompt.ask("是否同时删除本地的cloud_entries.json?", choices=['y', 'n'], default='n', show_choices=False) == 'y':
                    try:
                        os.remove(cloud_entries_path)
                        console.print("[green]✓ cloud_entries.json已删除[/]")
                    except Exception as e:
                        console.print(f"[red]cloud_entries.json删除失败: {str(e)}[/]")
                
        except CloudflareAPIError as e:
            if e.has_error_code(CF_ERROR_CODES['INDEX_DELETED']):
                console.print("[yellow]索引已经被删除[/]")
                # 虽然索引已删除，仍然询问是否要删除cloud_entries.json
                cloud_entries_path = os.path.join(os.path.dirname(__file__), 'cloud_entries.json')
                if os.path.exists(cloud_entries_path):
                    if Prompt.ask("是否同时删除本地的cloud_entries.json?", choices=['y', 'n'], default='n', show_choices=False) == 'y':
                        try:
                            os.remove(cloud_entries_path)
                            console.print("[green]✓ cloud_entries.json已删除[/]")
                        except Exception as e:
                            console.print(f"[red]cloud_entries.json删除失败: {str(e)}[/]")
            elif e.has_error_code(CF_ERROR_CODES['INDEX_NOT_FOUND']):
                console.print("[yellow]索引不存在，无需删除[/]")
            else:
                raise

    async def create_cloudflare_index(self):
        """创建或检查 Cloudflare 索引"""
        if not self.cf_account_id or not self.cf_api_token:
            raise ValueError('未设置 Cloudflare API 凭据')

        async def create_metadata_index():
            """创建元数据索引的通用函数"""
            try:
                await self.call_cloudflare_api(
                    f'/accounts/{self.cf_account_id}/vectorize/v2/indexes/vv-search/metadata_index/create',
                    'POST',
                    {
                        'propertyName': 'image_similarity',
                        'indexType': 'number'
                    }
                )
                console.print('[green]✓ 元数据索引创建成功[/]')
            except CloudflareAPIError as e:
                if e.has_error_code(3002):  # 索引已存在
                    console.print('[yellow]! 元数据索引已存在，将继续使用[/]')
                else:
                    console.print(f'[yellow]! 创建元数据索引失败，但将继续：{str(e)}[/]')
        
        # 创建主索引与元数据索引
        try:
            await self.call_cloudflare_api(
                f'/accounts/{self.cf_account_id}/vectorize/v2/indexes',
                'POST',
                {
                    'name': 'vv-search',
                    'description': 'Video subtitle search index',
                    'config': {
                        'dimensions': 1024,
                        'metric': 'cosine'
                    }
                }
            )
            console.print('[green]✓ 主索引创建成功[/]')
            await create_metadata_index()
            
        except CloudflareAPIError as e:
            if e.has_error_code(3002):  # 主索引已存在
                console.print('[yellow]主索引已存在[/]')
                if Prompt.ask("是否继续使用？", choices=['y', 'n'], default='y', show_choices=False) == 'y':
                    console.print('[yellow]继续使用现有索引[/]')
                    await create_metadata_index()
                else:
                    return
            else:
                raise


    async def upload_vectors_to_cloudflare(self):
        """上传向量到 Cloudflare Vectorize"""
        if not self.cf_account_id or not self.cf_api_token:
            raise ValueError('未设置 Cloudflare API 凭据')

        if not self.index:
            raise ValueError('本地索引未初始化')

        # 创建或检查索引
        await self.create_cloudflare_index()

        # 加载云端状态
        cloud_entries = self.load_cloud_entries()
        synced_ids = set(cloud_entries['entries']['synced'])
        pending_delete = set(cloud_entries['entries']['pending']['delete'])
        pending_update = set(cloud_entries['entries']['pending']['update'])

        # 如果是首次运行，将所有条目加入待更新列表
        if not cloud_entries['entries']['synced'] and not os.path.exists(os.path.join(os.path.dirname(__file__), 'cloud_entries.json')):
            pending_update = {str(entry.id) for entry in self.entries if entry.id is not None}

        total_updates = len(pending_update)

        # 1. 处理删除操作
        if pending_delete:
            console.print(f'[cyan]准备删除 {len(pending_delete)} 个向量...[/]')
            delete_batches = [list(pending_delete)[i:i+self.cf_batch_size] for i in range(0, len(pending_delete), self.cf_batch_size)]
            
            # 使用tqdm显示删除进度
            with tqdm(total=len(delete_batches), desc='删除向量') as pbar:
                for batch in delete_batches:
                    try:
                        await self.call_cloudflare_api(
                            f'/accounts/{self.cf_account_id}/vectorize/v2/indexes/vv-search/delete_by_ids',
                            'POST',
                            {'ids': batch, 'returnValues': False}
                        )
                        pending_delete -= set(batch)
                        cloud_entries['entries']['pending']['delete'] = list(pending_delete)
                        self.save_cloud_entries(cloud_entries)
                        pbar.update(1)
                    except Exception as e:
                        console.print(f'[red]删除批次失败，将在下次继续: {str(e)}[/]')
                        continue

        # 2. 处理更新操作
        if total_updates > 0:
            console.print(f'[cyan]准备更新 {total_updates} 个向量...[/]')

            # 直接使用字符串ID进行处理
            pending_update_ids = {id_str for id_str in pending_update}
            batch_entries = [entry for entry in self.entries if str(entry.id) in pending_update_ids]

            upload_semaphore = asyncio.Semaphore(CF_MAX_CONCURRENT)  # 限制并发API请求数
            successful_ids = set()
            failed_ids = set()
            pbar = tqdm(total=total_updates, desc='上传向量')

            async def upload_batch(batch):
                vectors_data = []
                batch_ids = set()
                
                try:
                    # 准备向量数据
                    for entry in batch:
                        vector = self.index.reconstruct(self.id_to_index[entry.id])
                        vector_data = {
                            'id': str(entry.id),
                            'values': vector.tolist(),
                            'metadata': {
                                'text': entry.text,
                                'timestamp': entry.timestamp,
                                'filename': entry.filename,
                                'image_similarity': float(entry.image_similarity)
                            }
                        }
                        vectors_data.append(json.dumps(vector_data))
                        batch_ids.add(str(entry.id))

                    if vectors_data:
                        async with upload_semaphore:  # 使用上下文管理器自动释放
                            ndjson_payload = '\n'.join(vectors_data)
                            try:
                                result = await self.call_cloudflare_api(
                                    f'/accounts/{self.cf_account_id}/vectorize/v2/indexes/vv-search/upsert',
                                    'POST',
                                    ndjson_payload,
                                    is_ndjson=True
                                )
                                
                                return {
                                    'success': True,
                                    'ids': batch_ids,
                                    'mutationId': result.get('mutationId')
                                }
                                
                            except CloudflareAPIError as e:
                                error_msg = str(e)
                                if e.is_payload_too_large():  # 使用专用方法代替 e.status == 413
                                    error_msg = "批次大小超过限制，需要减小批次大小"
                                elif e.is_rate_limit_error():  # 使用专用方法代替 e.status == 429
                                    error_msg = "请求频率过高，需要降低并发数"
                                
                                console.print(f"\n[red]API错误: {error_msg}[/]")
                                return {
                                    'success': False,
                                    'ids': batch_ids,
                                    'error': error_msg,
                                    'status': e.status,
                                    'errors': e.errors
                                }
                except Exception as e:
                    error_msg = f"发生未预期的错误: {type(e).__name__} - {str(e)}"
                    console.print(f"\n[red]{error_msg}[/]")
                    return {
                        'success': False,
                        'ids': batch_ids,
                        'error': error_msg
                    }
                
            # 使用asyncio.Queue控制并发任务创建
            async def process_batches():
                queue = asyncio.Queue()
                retry_queue = asyncio.Queue()
                in_progress = 0

                # 初始化队列 - 每个批次作为一个独立任务
                for i in range(0, len(batch_entries), self.cf_batch_size):
                    batch = batch_entries[i:min(i+self.cf_batch_size, len(batch_entries))]
                    await queue.put((batch, 0))  # (batch, retry_count)
                    in_progress += 1  # 每个批次只加1

                async def worker():
                    nonlocal in_progress
                    while True:
                        # 优先处理重试队列，然后是主队列
                        try:
                            batch, retry_count = retry_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            try:
                                batch, retry_count = await queue.get()
                            except asyncio.QueueEmpty:
                                if in_progress == 0:  # 所有任务都完成了
                                    break
                                await asyncio.sleep(0)  # 让出控制权给其他协程
                                continue

                        # 检查是否收到结束信号
                        if batch is None and retry_count == -1:
                            break

                        try:
                            result = await upload_batch(batch)
                            if result['success']:
                                successful_ids.update(result['ids'])
                                pbar.update(len(result['ids']))
                                #  成功时减少计数
                                in_progress -= 1
                            else:
                                if retry_count < MAX_RETRIES:
                                    # 计算指数退避延迟时间
                                    delay = CF_BACKOFF_BASE ** retry_count
                                    if result.get('status') == 429:  # Rate limit
                                        console.print(f"\n[yellow]速率限制，等待 {delay}秒后重试...[/]")
                                        await asyncio.sleep(delay)
                                    elif result.get('status') == 413:  # Payload too large
                                        # 将批次分成两半
                                        mid = len(batch) // 2
                                        if mid > 0:
                                            #  拆分前减少原批次计数
                                            in_progress -= 1
                                            #  为两个新批次各加一次计数
                                            in_progress += 2
                                            await queue.put((batch[:mid], retry_count))
                                            await queue.put((batch[mid:], retry_count))
                                            console.print("\n[yellow]批次过大，已拆分为两个小批次[/]")
                                            continue
                                    # 将任务放入重试队列
                                    await retry_queue.put((batch, retry_count + 1))
                                else:
                                    # 超过最大重试次数
                                    failed_ids.update(result['ids'])
                                    pbar.update(len(result['ids']))
                                    in_progress -= 1  # 任务失败，减少一个计数
                                    error_msg = result.get('error', '未知错误')
                                    console.print(f"\n[red]达到最大重试次数，放弃批次: {error_msg}[/]")
                        except Exception as e:
                            if retry_count < MAX_RETRIES:
                                # 重试
                                delay = CF_BACKOFF_BASE ** retry_count
                                console.print(f"\n[yellow]发生错误，等待 {delay}秒后重试: {str(e)}[/]")
                                await asyncio.sleep(delay)
                                await retry_queue.put((batch, retry_count + 1))
                            else:
                                # 记录失败
                                failed_ids.update(str(entry.id) for entry in batch)
                                pbar.update(len(batch))
                                in_progress -= 1  # 任务失败，减少一个计数
                                console.print(f"\n[red]达到最大重试次数，放弃批次: {str(e)}[/]")

                # 创建协程任务
                tasks = []
                for _ in range(8):  # 固定8个并发协程
                    task = asyncio.create_task(worker())
                    tasks.append(task)
                
                # 添加任务结束信号("毒丸")，确保所有worker能正确退出
                for _ in range(8):
                    await queue.put((None, -1))  # None作为结束信号

                # 等待所有协程任务完成
                await asyncio.gather(*tasks)

            # 处理所有批次
            await process_batches()
            pbar.close()

            # 更新同步状态
            synced_ids.update(successful_ids)
            pending_update -= successful_ids
            
            # 保存最终状态
            cloud_entries['entries']['synced'] = list(synced_ids)
            cloud_entries['entries']['pending']['update'] = list(pending_update)
            self.save_cloud_entries(cloud_entries)

            console.print(f'\n[green]完成! 成功上传 {len(successful_ids)}/{total_updates} 个向量[/]')

        if pending_delete or total_updates > 0:
            console.print('[green]✓ 向量同步完成![/]')
            return True
        return False
    
    def load_subtitles(self):
        json_files = [f for f in os.listdir(self.subtitle_folder) if f.endswith('.json')]
        for filename in tqdm(json_files, desc="加载字幕文件"):
            if filename.endswith('.json'):
                filepath = os.path.join(self.subtitle_folder, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    video_name = filename[:-5]  # 去除.json后缀
                    for entry in data:
                        self.entries.append(SubtitleEntry(
                            text=entry['text'],
                            timestamp=entry['timestamp'],
                            filename=video_name,
                            image_similarity=entry.get('similarity', 0.0)  # 读取图像相似度
                        ))

    def create_index(self):
        texts = [entry.text for entry in self.entries]
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=SEARCH_BATCH_SIZE
        )
        dimension = embeddings.shape[1]
        # 使用 IndexFlatIP 作为基础索引以使用内积计算余弦相似度
        base_index = faiss.IndexFlatIP(dimension)  
        # 使用 IndexIDMap2 代替 IndexIDMap 以存储原向量
        idmap = faiss.IndexIDMap2(base_index)
        # 生成随机id
        def generate_random_id():
            chars = string.ascii_letters + string.digits
            return ''.join(secrets.choice(chars) for _ in range(20))

        # 为每个条目分配随机id，同时维护一个字符串ID到整数ID的映射
        existing_ids = set()
        id_array = np.arange(len(self.entries), dtype=np.int64)  # 使用数组索引作为Faiss的ID
        self.id_to_index = {}  # 字符串ID到索引的映射
        
        for i, entry in enumerate(self.entries):
            while True:
                new_id = generate_random_id()
                if new_id not in existing_ids:
                    entry.id = new_id
                    existing_ids.add(new_id)
                    self.id_to_index[new_id] = i
                    break
                    
        idmap.add_with_ids(embeddings, id_array)
        self.index = idmap

    def save_index(self, index_dir):
        """保存索引和条目数据到文件"""
        os.makedirs(index_dir, exist_ok=True)
        index_path = os.path.join(index_dir, 'faiss.index')
        entries_path = os.path.join(index_dir, 'entries.json')
        
        faiss.write_index(self.index, index_path)
        
        entries_data = [
            {
                'text': e.text,
                'timestamp': e.timestamp,
                'filename': e.filename,
                'image_similarity': e.image_similarity,
                'id': e.id
            }
            for e in self.entries
        ]
        with open(entries_path, 'w', encoding='utf-8') as f:
            json.dump(entries_data, f, ensure_ascii=False, indent=2)

    def load_index(self, index_dir):
        """从文件加载索引和条目数据"""
        index_path = os.path.join(index_dir, 'faiss.index')
        entries_path = os.path.join(index_dir, 'entries.json')
        
        self.index = faiss.read_index(index_path)
        
        # 加载entries数据
        with open(entries_path, 'r', encoding='utf-8') as f:
            entries_data = json.load(f)
            self.entries = [
                SubtitleEntry(**entry)
                for entry in entries_data
            ]
        
        # 重建ID到索引的映射
        self.id_to_index = {
            entry.id: i
            for i, entry in enumerate(self.entries)
            if entry.id is not None
        }

    def update_index(self):
        # 读取所有当前文件的条目
        current_entries = {}
        for filename in tqdm(os.listdir(self.subtitle_folder), desc="扫描字幕文件"):
            if filename.endswith('.json'):
                video_name = filename[:-5]  # 去掉.json后缀
                with open(os.path.join(self.subtitle_folder, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for entry in data:
                        key = (video_name, entry['timestamp'])
                        current_entries[key] = entry['text']
        # 构建原有条目映射（仅包含 id 不为空的）
        existing_map = { (entry.filename, entry.timestamp): entry for entry in self.entries if entry.id is not None }
        
        new_entries = []
        changed_entries = []          # 记录文本改变的条目（需要更新向量）
        incremental_entries = []      # 记录全新的条目（需要添加向量）
        
        for key, text in current_entries.items():
            if key in existing_map:
                exist = existing_map[key]
                if exist.text != text:
                    # 文本改变，保留原有 id
                    new_entry = SubtitleEntry(
                        text=text,
                        timestamp=key[1],
                        filename=key[0],
                        image_similarity=0.0,
                        id=exist.id
                    )
                    new_entries.append(new_entry)
                    changed_entries.append(new_entry)
            else:
                # 全新条目
                new_ent = SubtitleEntry(
                    text=text,
                    timestamp=key[1],
                    filename=key[0],
                    image_similarity=0.0
                )
                new_entries.append(new_ent)
                incremental_entries.append(new_ent)
        
        # 对全新条目分配新的随机id
        existing_ids = {entry.id for entry in self.entries if entry.id is not None}
        def generate_random_id():
            chars = string.ascii_letters + string.digits
            return ''.join(secrets.choice(chars) for _ in range(20))

        for entry in incremental_entries:
            while True:
                new_id = generate_random_id()
                if new_id not in existing_ids:
                    entry.id = new_id
                    existing_ids.add(new_id)
                    break
        
        # 针对文本变化的条目，逐条更新：
        if changed_entries:
            for entry in tqdm(changed_entries, desc="更新已变更条目"):
                # 删除旧向量，使用ID到索引的映射
                idx = self.id_to_index[entry.id]
                self.index.remove_ids(np.array([idx], dtype=np.int64))
                # 计算更新后的向量并添加
                embedding = self.model.encode([entry.text])
                faiss.normalize_L2(embedding)
                self.index.add_with_ids(embedding, np.array([idx], dtype=np.int64))
        
        # 对全新条目批量添加向量
        if incremental_entries:
            new_texts = [ent.text for ent in incremental_entries]
            console.print("[cyan]正在处理新增条目...[/]")
            new_embeddings = self.model.encode(
                new_texts,
                batch_size=SEARCH_BATCH_SIZE,
                show_progress_bar=True
            )
            faiss.normalize_L2(new_embeddings)
            new_ids = np.array([ent.id for ent in incremental_entries], dtype=np.int64)
            self.index.add_with_ids(new_embeddings, new_ids)
        
        # 找出被删除的条目ID
        deleted_ids = [str(ex.id) for ex in existing_map.values() 
                      if ex.id not in [e.id for e in new_entries]]

        # 更新cloud_entries.json
        if os.path.exists(os.path.join(os.path.dirname(__file__), 'cloud_entries.json')):
            cloud_entries = self.load_cloud_entries()
            
            # 更新待删除列表
            cloud_entries['entries']['pending']['delete'].extend(deleted_ids)
            
            # 更新待更新列表
            for entry in changed_entries:
                if str(entry.id) not in cloud_entries['entries']['pending']['update']:
                    cloud_entries['entries']['pending']['update'].append(str(entry.id))
            
            # 添加新增条目到更新列表
            for entry in incremental_entries:
                if str(entry.id) not in cloud_entries['entries']['pending']['update']:
                    cloud_entries['entries']['pending']['update'].append(str(entry.id))
            
            self.save_cloud_entries(cloud_entries)

        # 计算删除的条目数        
        deleted_count = len(deleted_ids)
        
        self.entries = new_entries
        
        return {
            'old_count': len(existing_map),
            'new_count': len(self.entries),
            'diff': len(self.entries) - len(existing_map),
            'new_entries': len(incremental_entries),
            'updated': len(changed_entries),
            'deleted': deleted_count
        }

    def check_index_integrity(self):
        """检查索引完整性并修复"""
        # 获取FAISS索引中使用的数字索引
        faiss_indices = set(range(self.index.ntotal))
        
        # 比较差异 - 使用 id_to_index 映射
        missing_in_faiss = {idx for idx in self.id_to_index.items() if idx not in faiss_indices}
        extra_in_faiss = faiss_indices - set(self.id_to_index.values())
        
        if not missing_in_faiss and not extra_in_faiss:
            return {
                'status': 'ok',
                'message': '索引完整性检查通过',
                'fixed': False
            }
        
        # 修复索引
        # 1. 删除FAISS中多余的ID
        if extra_in_faiss:
            self.index.remove_ids(np.array(list(extra_in_faiss), dtype=np.int64))
        
        # 2. 为缺失的条目重新添加向量
        if missing_in_faiss:
            missing_entries = [entry for entry in self.entries if entry.id in missing_in_faiss]
            texts = [entry.text for entry in missing_entries]
            embeddings = self.model.encode(
                texts,
                batch_size=SEARCH_BATCH_SIZE,
                show_progress_bar=True
            )
            faiss.normalize_L2(embeddings)
            missing_ids = np.array([entry.id for entry in missing_entries], dtype=np.int64)
            self.index.add_with_ids(embeddings, missing_ids)
        
        return {
            'status': 'fixed',
            'message': f'索引已修复 (删除: {len(extra_in_faiss)}, 补充: {len(missing_in_faiss)})',
            'fixed': True,
            'removed': len(extra_in_faiss),
            'added': len(missing_in_faiss)
        }

    def search(self, query, k=None):
        if k is None:
            k = self.search_k
        
        search_k = min(k * 30, len(self.entries))
        query_embedding = self.model.encode([query])
        # 规范化查询向量
        faiss.normalize_L2(query_embedding)
        similarities, indices = self.index.search(query_embedding, search_k)
        
        results = []
        for i in range(search_k):
            if i >= len(indices[0]):
                break
            idx = indices[0][i]  # 这是我们使用的数组索引
            if idx >= len(self.entries):  # 确保索引有效
                continue
                
            entry = self.entries[idx]
            if entry.image_similarity < self.min_image_similarity:
                continue
            
            text_sim = float(similarities[0][i])
            img_sim = entry.image_similarity
            
            results.append({
                'text_similarity': text_sim,
                'image_similarity': img_sim,
                'timestamp': entry.timestamp,
                'text': entry.text,
                'filename': entry.filename
            })
            
            if len(results) >= k * 2:
                break
        
        # 按文本相似度排序
        results.sort(key=lambda x: x['text_similarity'], reverse=True)
        return results[:k]

if __name__ == "__main__":
    console = Console()
    subtitle_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'subtitle')
    index_dir = os.path.join(os.path.dirname(__file__), 'index')
    
    try:
        searcher = SubtitleSearch(subtitle_folder)
        
        # 检查索引文件是否存在
        if os.path.exists(index_dir):
            console.print("\n[cyan]发现已存在的索引文件[/]")
            console.print("\n1. 使用已有索引\n2. 更新索引\n3. 重新构建索引")
            choice = Prompt.ask("请选择", choices=["1", "2", "3"], default="1", show_choices=False)
            
            if choice == "1":
                console.print("[cyan]正在加载索引...[/]")
                searcher.load_index(index_dir)
                console.print("[green]索引加载完成![/]")
                input("\n按Enter继续...")
            elif choice == "2":
                console.print("[cyan]正在加载索引...[/]")
                searcher.load_index(index_dir)
                console.print("[cyan]正在更新索引...[/]")
                stats = searcher.update_index()
                console.print("[cyan]正在保存索引...[/]")
                searcher.save_index(index_dir)
                console.print(
                    f"[green]索引更新完成![/]\n"
                    f"原有条目: {stats['old_count']}\n"
                    f"现有条目: {stats['new_count']}\n" 
                    f"新增条目: {stats['new_entries']}\n"
                    f"更新条目: {stats['updated']}\n"
                    f"删除条目: {stats['deleted']}\n"
                    f"净变化: {stats['diff']:+d}"
                )
                input("\n按Enter继续...")
            else:
                console.print("[cyan]准备构建新索引...[/]")
                input("按Enter开始构建索引...")
                
                console.print("[cyan]正在加载字幕文件...[/]")
                searcher.load_subtitles()
                console.print("[cyan]正在构建索引...[/]")
                searcher.create_index()
                console.print("[cyan]正在保存索引...[/]")
                searcher.save_index(index_dir)
                console.print("[green]索引构建完成![/]")
                
                # 询问是否删除cloud_entries.json
                cloud_entries_path = os.path.join(os.path.dirname(__file__), 'cloud_entries.json')
                if os.path.exists(cloud_entries_path):
                    if Prompt.ask("\n是否删除cloud_entries.json?", choices=['y', 'n'], default='n', show_choices=False) == 'y':
                        try:
                            os.remove(cloud_entries_path)
                            console.print("[green]✓ cloud_entries.json已删除[/]")
                        except Exception as e:
                            console.print(f"[red]删除失败: {str(e)}[/]")
                
                input("\n按Enter继续...")
        else:
            console.print("[cyan]未找到索引文件，准备构建新索引...[/]")
            input("按Enter开始构建索引...")
            
            console.print("[cyan]正在加载字幕文件...[/]")
            searcher.load_subtitles()
            console.print("[cyan]正在构建索引...[/]")
            searcher.create_index()
            console.print("[cyan]正在保存索引...[/]")
            searcher.save_index(index_dir)
            console.print("[green]索引构建完成![/]")
            
            # 询问是否删除cloud_entries.json
            cloud_entries_path = os.path.join(os.path.dirname(__file__), 'cloud_entries.json')
            if os.path.exists(cloud_entries_path):
                if Prompt.ask("\n是否删除cloud_entries.json?", choices=['y', 'n'], default='n', show_choices=False) == 'y':
                    try:
                        os.remove(cloud_entries_path)
                        console.print("[green]✓ cloud_entries.json已删除[/]")
                    except Exception as e:
                        console.print(f"[red]删除失败: {str(e)}[/]")
            
            input("\n按Enter继续...")

    except Exception as e:
        console.print(f"[red]初始化失败: {str(e)}[/]")
        input('按Enter退出...')
        exit(1)
    
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        console.print(Panel(
            f"图像相似度阈值: {searcher.min_image_similarity:.2f} | "
            f"每页结果数: {searcher.search_k}\n"
            "命令: [cyan]q[/]退出, [cyan]s[/]设置, [cyan]Enter[/]搜索",
            title="[cyan]!VVV![/]"
        ))
        
        query = Prompt.ask("搜索关键词").strip()
        if not query:
            continue
            
        if query.lower() == 'q':
            break
        elif query.lower() == 's':
            while True:
                os.system('cls' if os.name == 'nt' else 'clear')
                console.print(Panel(
                    f"1. 图像相似度阈值 ({searcher.min_image_similarity:.2f})\n"
                    f"2. 每页结果数量 ({searcher.search_k})\n"
                    "3. 检查索引完整性\n"
                    "4. 上传向量到 Cloudflare Vectorize\n"
                    "5. 删除Cloudflare Vectorize索引数据库\n"
                    "0. 返回搜索",
                    title="设置菜单"
                ))
                
                choice = Prompt.ask("选择要修改的设置", choices=["0", "1", "2", "3", "4", "5"], show_choices=False)
                if choice == "1":
                    try:
                        new_sim = FloatPrompt.ask(
                            "输入新的图像相似度阈值", 
                            default=searcher.min_image_similarity
                        )
                        if 0 <= new_sim <= 1:
                            searcher.min_image_similarity = new_sim
                            console.print("\n[green]✓ 更新成功[/]")
                        else:
                            raise ValueError("数值必须在0到1之间")
                    except ValueError as e:
                        console.print(f"\n[red]✗ 输入无效: {str(e)}[/]")
                    finally:
                        input('\n按Enter继续...')
                elif choice == "2":
                    try:
                        new_k = IntPrompt.ask(
                            "输入新的结果数量",
                            default=searcher.search_k
                        )
                        if 1 <= new_k <= 50:
                            searcher.search_k = new_k
                            console.print("\n[green]✓ 更新成功[/]")
                        else:
                            raise ValueError("数值必须在1到50之间")
                    except ValueError as e:
                        console.print(f"\n[red]✗ 输入无效: {str(e)}[/]")
                    finally:
                        input('\n按Enter继续...')
                elif choice == "3":
                    console.print("\n[cyan]正在检查索引完整性...[/]")
                    try:
                        result = searcher.check_index_integrity()
                        if result['status'] == 'ok':
                            console.print(f"\n[green]✓ {result['message']}[/]")
                        else:
                            console.print(f"\n[yellow]! {result['message']}[/]")
                            if result['fixed']:
                                console.print("[cyan]正在保存修复后的索引...[/]")
                                searcher.save_index(index_dir)
                                console.print("[green]✓ 索引已更新[/]")
                    except Exception as e:
                        console.print(f"\n[red]✗ 检查失败: {str(e)}[/]")
                    finally:
                        input('\n按Enter继续...')
                elif choice == "4":
                    if not searcher.cf_account_id or not searcher.cf_api_token:
                        console.print("\n[red]缺少环境变量 CLOUDFLARE_ACCOUNT_ID 或 CLOUDFLARE_API_TOKEN[/]")
                        input('\n按Enter继续...')
                        continue

                    console.print("\n[cyan]准备上传向量到 Cloudflare...[/]")
                    try:
                        # 上传向量，并根据结果来决定是否显示完成消息
                        upload_successful = asyncio.run(searcher.upload_vectors_to_cloudflare())
                        # 检查返回值以避免在没有操作时显示成功
                        if upload_successful is True: # 只有当明确返回True时才显示成功
                            console.print("\n[green]✓ 向量同步完成[/]")
                        elif upload_successful is False:
                             console.print("\n[yellow]! 没有需要同步的向量。[/]") # 如果没有操作，提示用户

                    except CloudflareAPIError as e:
                        # 使用 CloudflareAPIError 的 __str__ 方法获取格式化的错误信息
                        detailed_error_message = str(e)
                        console.print(f"\n[red]✗ 上传/同步过程中发生错误:[/]\n{detailed_error_message}")

                    except Exception as e:
                        # 对于其他类型的异常，打印异常类型和消息
                        console.print(f"\n[red]✗ 上传/同步过程中发生错误: {type(e).__name__} - {str(e)}[/]")
                        traceback.print_exc()
                        
                    finally:
                        input('\n按Enter继续...')
                elif choice == "5":
                    if not searcher.cf_account_id or not searcher.cf_api_token:
                        console.print("\n[red]缺少环境变量 CLOUDFLARE_ACCOUNT_ID 或 CLOUDFLARE_API_TOKEN[/]")
                        input('\n按Enter继续...')
                        continue

                    console.print("\n[yellow]警告: 该操作将删除Cloudflare上的向量索引[/]")
                    if Prompt.ask("确定要继续吗?", choices=["y", "n"], default="n", show_choices=False) == "y":
                        try:
                            asyncio.run(searcher.delete_cloudflare_index())
                            console.print("\n[green]✓ 索引删除成功[/]")
                        except CloudflareAPIError as e:
                             # 使用 CloudflareAPIError 的 __str__ 方法获取格式化的错误信息
                            detailed_error_message = str(e)
                            console.print(f"\n[red]✗ 删除索引时发生错误:[/]\n{detailed_error_message}")
                        except Exception as e:
                            console.print(f"\n[red]✗ 删除索引时发生未预期的错误: {type(e).__name__} - {str(e)}[/]")
                    input('\n按Enter继续...')
                elif choice == "0":
                    break
            continue
        
        try:
            results = searcher.search(query)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            if not results:
                console.print(Panel(
                    f"[yellow]未找到符合条件的结果 (图像相似度阈值: {searcher.min_image_similarity:.2f})[/]",
                    title=f"搜索: {query}"
                ))
            else:
                table = Table(title=f"搜索: {query} (图像相似度阈值: {searcher.min_image_similarity:.2f})")
                table.add_column("序号", justify="right", width=4)
                table.add_column("视频名", width=12)
                table.add_column("时间戳", width=12)
                table.add_column("文本相似度", justify="right", width=10)
                table.add_column("图像相似度", justify="right", width=10)
                table.add_column("内容")
                table.add_column("打开", justify="center", width=8)
                
                for idx, result in enumerate(results, 1):
                    # 获取视频URL
                    video_url = get_video_url(result["filename"], result["timestamp"])
                    url_text = "[link]打开[/]" if video_url else "-"

                    table.add_row(
                        f"[cyan]{idx}[/]",
                        result["filename"],
                        result["timestamp"],
                        f"{result['text_similarity']:.3f}",
                        f"{result['image_similarity']:.3f}",
                        result["text"],
                        url_text if not video_url else f"[link={video_url}]打开[/]"
                    )
                console.print(table)
        except Exception as e:
            console.print(f"[red]搜索出错: {str(e)}[/]")
        finally:
            input('\n按Enter继续...')
