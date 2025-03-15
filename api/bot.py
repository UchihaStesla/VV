from http.server import BaseHTTPRequestHandler
from telegram import Update, InlineQueryResultArticle, InlineQueryResultPhoto, InputTextMessageContent
from telegram.ext import Application
from uuid import uuid4
import json
import logging
import subprocess
import os
import asyncio
import re
import base64
import struct
import requests
from typing import Optional, Tuple


class PreviewExtractor:
    def __init__(self, base_url: str = "https://vv.noxylva.org"):
        self.base_url = base_url
        self.session = requests.Session()
        self.timeout = 5

    def _fetch_index(self, group_index: int) -> bytes:
        try:
            index_url = f"{self.base_url}/{group_index}.index"
            response = self.session.get(index_url, timeout=self.timeout)
            response.raise_for_status()
            return response.content
        except requests.Timeout:
            logger.warning(f"获取索引超时: {group_index}.index")
            raise
        except Exception as e:
            logger.error(f"获取索引失败: {e}")
            raise

    def _parse_index(self, index_data: bytes, folder_id: int, frame_num: int) -> Optional[Tuple[int, Optional[int]]]:
        grid_w, grid_h, folder_count = struct.unpack("<III", index_data[:12])
        
        offset = 12 + folder_count * 4
        
        file_count = struct.unpack("<I", index_data[offset:offset+4])[0]
        offset += 4

        left, right = 0, file_count - 1
        while left <= right:
            mid = (left + right) // 2
            record_offset = offset + mid * 16
            
            curr_folder, curr_frame, curr_offset = struct.unpack("<IIQ", 
                index_data[record_offset:record_offset+16])

            if curr_folder == folder_id and curr_frame == frame_num:
                end_offset = None
                if mid < file_count - 1:
                    end_offset = struct.unpack("<Q", 
                        index_data[record_offset+24:record_offset+32])[0]
                return int(curr_offset), end_offset
            elif curr_folder < folder_id or (curr_folder == folder_id and curr_frame < frame_num):
                left = mid + 1
            else:
                right = mid - 1
                
        return None

    def extract_frame(self, folder_id: int, frame_num: int) -> Optional[bytes]:
        group_index = (folder_id - 1) // 10
        
        try:
            index_data = self._fetch_index(group_index)
            offset_info = self._parse_index(index_data, folder_id, frame_num)
            
            if not offset_info:
                return None
                
            start_offset, end_offset = offset_info
            
            headers = {}
            if end_offset:
                headers["Range"] = f"bytes={start_offset}-{end_offset-1}"
            else:
                headers["Range"] = f"bytes={start_offset}-"
                
            image_url = f"{self.base_url}/{group_index}.webp"
            try:
                response = self.session.get(image_url, headers=headers, timeout=self.timeout)
                
                if response.status_code == 416:
                    response = self.session.get(image_url, timeout=self.timeout)
                    
                response.raise_for_status()
                return response.content
            except requests.Timeout:
                logger.warning(f"获取图片超时: P{folder_id} 第{frame_num}秒")
                return None
                
        except Exception as e:
            logger.error(f"提取帧 {frame_num} (文件夹 {folder_id}) 时出错: {e}")
            return None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

preview_extractor = PreviewExtractor()

async def search_vv_quotes(query: str) -> list:
    try:
        process = subprocess.Popen(
            ['./api/subtitle_search_api'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )
        stdout, _ = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: process.communicate(input=f"query={query}\n", timeout=30)
        )
        
        quotes = []
        for line in stdout.strip().split('\n'):
            try:
                if line:
                    data = json.loads(line)
                    if all(key in data for key in ['text']):
                        quotes.append(data)
            except json.JSONDecodeError:
                continue
        
        return quotes[:5]
    except Exception as e:
        logger.error(f"搜索出错: {str(e)}")
        return []

async def handle_inline_query(update: Update):
    query = update.inline_query.query
    
    if not query:
        return [InlineQueryResultArticle(
            id=str(uuid4()),
            title="输入关键词搜索 VV 语录",
            description="试试输入：我看了都乐了、确实",
            input_message_content=InputTextMessageContent(
                message_text="我看了都乐了"
            )
        )]
    
    try:
        quotes = await search_vv_quotes(query)
        results = []
        
        for i, quote in enumerate(quotes, 1):
            text = quote.get('text', '')
            timestamp = quote.get('timestamp', '')
            filename = quote.get('filename', '')
            
            if text:
                episode_match = re.search(r'\[P(\d+)\]', filename)
                time_match = re.search(r'^(\d+)m(\d+)s$', timestamp)
                
                if episode_match and time_match:
                    folder_id = int(episode_match.group(1))
                    minutes = int(time_match.group(1))
                    seconds = int(time_match.group(2))
                    total_seconds = minutes * 60 + seconds
                    
                    preview_url = f"https://vv-indol.vercel.app/api/preview/{folder_id}/{total_seconds}"
                    
                    results.append(InlineQueryResultPhoto(
                        id=str(uuid4()),
                        photo_url=preview_url,
                        thumbnail_url=preview_url,
                        title=f"{text[:50]}{'...' if len(text) > 50 else ''}",
                        description=f"P{folder_id} {timestamp}"
                    ))
                    continue

                results.append(InlineQueryResultArticle(
                    id=str(uuid4()),
                    title=f"{text[:50]}{'...' if len(text) > 50 else ''}",
                    description="",
                    input_message_content=InputTextMessageContent(
                        message_text=text
                    )
                ))
        
        if not results:
            results.append(InlineQueryResultArticle(
                id=str(uuid4()),
                title="未找到相关语录",
                description="试试其他关键词",
                input_message_content=InputTextMessageContent(
                    message_text="没找到相关语录，试试其他关键词"
                )
            ))
        
        return results
        
    except Exception as e:
        logger.error(f"查询处理错误: {str(e)}")
        return [InlineQueryResultArticle(
            id=str(uuid4()),
            title="处理查询时出错",
            description="请稍后重试",
            input_message_content=InputTextMessageContent(
                message_text="抱歉，处理查询时出现错误，请稍后重试"
            )
        )]

class handler(BaseHTTPRequestHandler):
    async def _handle_preview(self):
        try:
            match = re.match(r'/api/preview/(\d+)/(\d+)', self.path)
            if not match:
                self.send_error(400, "Invalid URL format")
                return

            folder_id = int(match.group(1))
            time = int(match.group(2))

            frame_data = preview_extractor.extract_frame(folder_id, time)
            
            if frame_data:
                self.send_response(200)
                self.send_header('Content-Type', 'image/webp')
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.end_headers()
                self.wfile.write(frame_data)
            else:
                self.send_error(404, "Preview not found")
                
        except Exception as e:
            logger.error(f"预览图处理错误: {str(e)}")
            self.send_error(500, str(e))

    async def _handle_bot(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data)
        
        try:
            async with asyncio.timeout(45):
                update = Update.de_json(data, Application.builder().token(TOKEN).build().bot)
                
                if update.inline_query:
                    results = await handle_inline_query(update)
                    await update.inline_query.answer(results, cache_time=0, is_personal=True)
                    
                response = {"status": "ok"}
        except asyncio.TimeoutError:
            logger.error("请求超时")
            response = {"status": "error", "message": "请求超时"}
        except Exception as e:
            logger.error(f"Webhook 错误: {str(e)}")
            response = {"status": "error", "message": str(e)}
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def do_GET(self):
        if self.path.startswith('/api/preview/'):
            asyncio.run(self._handle_preview())

    def do_POST(self):
        if self.path == '/api/bot':
            asyncio.run(self._handle_bot())
