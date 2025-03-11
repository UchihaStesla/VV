import os
from PIL import Image
# 添加 ANTIALIAS 兼容性处理
Image.ANTIALIAS = Image.Resampling.LANCZOS

import numpy as np
import json
import re
from collections import defaultdict
from paddleocr import PaddleOCR
import traceback
import logging
from params import (
    USE_GPU_OCR, GPU_MEMORY_OCR, 
    OCR_MODEL_DIR
)

class SubtitleExtractor:
    def __init__(self):
        logging.disable(logging.WARNING)
        self.ocr = PaddleOCR(
            use_angle_cls=False,
            lang="ch",
            rec_model_dir=OCR_MODEL_DIR,
            rec_image_shape=[3, 1200, 90],
            show_log=False,
            use_gpu=USE_GPU_OCR,
            gpu_mem=GPU_MEMORY_OCR if USE_GPU_OCR else None,
            enable_mkldnn=True,
            det=False, 
            cls=False
        )
        
        self.subtitle_area = (235, 900, 235 + 1200, 900 + 90)
        self.pattern = r'([^_]+)_(\d+m\d+s)_sim_(\d+\.\d+)'
        self.subtitles_dict = defaultdict(list)

    def clean_text(self, text):
        """使用正则清理文本末尾的标点符号和多余空格"""
        if not text:
            return text
        # 清理末尾的标点符号和空格
        # 匹配末尾的：中文标点、英文标点、空格
        return re.sub(r'[\u3000-\u303F\uFF00-\uFFEF\u2000-\u206F.,!?;:\s]+$', '', text.strip())

    def parse_timestamp(self, timestamp):
        """将时间戳 (如 "2m28s") 转换为总秒数"""
        match = re.match(r'(\d+)m(\d+)s', timestamp)
        if match:
            minutes, seconds = map(int, match.groups())
            return minutes * 60 + seconds
        return 0

    def process_image(self, img_path):
        try:
            img = Image.open(img_path)
            
            if img.size != (1920, 1080):
                raise ValueError("Incorrect image size")
            
            # 裁剪字幕区域
            subtitle_img = img.crop(self.subtitle_area)
            
            # 将PIL Image转换为numpy数组
            img_array = np.array(subtitle_img)
            
            # 只进行文字识别
            result = self.ocr.ocr(img_array, det=False, cls=False)
            
            # 释放内存
            img.close()
            subtitle_img.close()
            del img, subtitle_img, img_array
            
            if result:
                text = result[0]
                if isinstance(text, (list, tuple)):
                    text = text[0]
                if isinstance(text, (list, tuple)):
                    text = text[0]
                    
                text = str(text).strip()
                text = self.clean_text(text)
                return text
            return None
            
        except Exception as e:
            print(f"Error processing image {img_path}:")
            traceback.print_exc()
            return None

    def process_frames(self, input_folder, output_folder):
        """处理文件夹中的所有帧并生成字幕"""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in sorted(os.listdir(input_folder)):
            if not filename.endswith(('.jpg', '.png')):
                continue
            
            match = re.match(self.pattern, filename)
            if not match:
                 continue
            
            title, timestamp, similarity = match.groups()
            video_title = f"{title}"
            print(f"处理文件: {filename}")
            
            img_path = os.path.join(input_folder, filename)
            text = self.process_image(img_path)
            
            if not text:
                continue

            print(f"识别到文本: {text}")
            
            # 检查重复
            subtitles = self.subtitles_dict[video_title]
            if subtitles and subtitles[-1]["text"] == text:
                continue
            
            # 添加字幕
            self.subtitles_dict[video_title].append({
                "timestamp": timestamp,
                "similarity": float(similarity),
                "text": text
            })

        # 保存字幕文件
        for video_title, subtitles in self.subtitles_dict.items():
            sorted_subtitles = sorted(subtitles, key=lambda x: self.parse_timestamp(x["timestamp"]))
            output_json = os.path.join(output_folder, f"{video_title}.json")
            
            try:
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump(sorted_subtitles, f, ensure_ascii=False, indent=4)
                print(f"成功保存 {video_title} 的字幕")
            except Exception as e:
                print(f"保存文件时出错: {str(e)}")

        print("\n处理完成")