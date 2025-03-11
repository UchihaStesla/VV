import os
import json
import faiss
import logging
import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from rich.console import Console
from rich.prompt import Prompt, FloatPrompt, IntPrompt
from rich.panel import Panel
from rich.table import Table

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from params import USE_GPU_SEARCH, SEARCH_BATCH_SIZE
from mapping import get_video_url  # 在文件开头添加导入

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

class SubtitleSearch:
    def __init__(self, subtitle_folder, model_name='BAAI/bge-large-zh-v1.5'):
        self.subtitle_folder = subtitle_folder
        self.use_gpu = USE_GPU_SEARCH
        self.model = SentenceTransformer(model_name, device='cuda' if self.use_gpu else 'cpu')
        self.entries = []
        self.min_image_similarity = 0.6
        self.search_k = 5
        self.index = None
        self.sentence_embeddings = None
        self.min_text_similarity = 0.5  # 添加文本相似度阈值
        
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
        self.sentence_embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=SEARCH_BATCH_SIZE
        )
        dimension = self.sentence_embeddings.shape[1]
        if USE_GPU_SEARCH:
            res = faiss.StandardGpuResources()
            self.index = faiss.GpuIndexFlatL2(res, dimension)
        else:
            self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.sentence_embeddings)

    def save_index(self, index_dir):
        """保存索引和嵌入向量到文件"""
        os.makedirs(index_dir, exist_ok=True)
        index_path = os.path.join(index_dir, 'faiss.index')
        embeddings_path = os.path.join(index_dir, 'embeddings.npy')
        entries_path = os.path.join(index_dir, 'entries.json')
        
        # 如果是GPU索引，需要先转换为CPU索引再保存
        if USE_GPU_SEARCH:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, index_path)
        else:
            faiss.write_index(self.index, index_path)
        
        # 保存嵌入向量
        np.save(embeddings_path, self.sentence_embeddings)
        
        # 保存entries数据
        entries_data = [
            {
                'text': e.text,
                'timestamp': e.timestamp,
                'filename': e.filename,
                'image_similarity': e.image_similarity
            }
            for e in self.entries
        ]
        with open(entries_path, 'w', encoding='utf-8') as f:
            json.dump(entries_data, f, ensure_ascii=False, indent=2)

    def load_index(self, index_dir):
        """从文件加载索引和嵌入向量"""
        index_path = os.path.join(index_dir, 'faiss.index')
        embeddings_path = os.path.join(index_dir, 'embeddings.npy')
        entries_path = os.path.join(index_dir, 'entries.json')
        
        # 先加载为CPU索引
        cpu_index = faiss.read_index(index_path)
        
        # 如果启用了GPU，将索引转换为GPU版本
        if USE_GPU_SEARCH:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            self.index = cpu_index
            
        # 加载嵌入向量
        self.sentence_embeddings = np.load(embeddings_path)
        
        # 加载entries数据
        with open(entries_path, 'r', encoding='utf-8') as f:
            entries_data = json.load(f)
            self.entries = [
                SubtitleEntry(**entry)
                for entry in entries_data
            ]

    def search(self, query, k=None):
        if k is None:
            k = self.search_k
        
        search_k = min(k * 30, len(self.entries))
        query_embedding = self.model.encode([query], batch_size=SEARCH_BATCH_SIZE)  # 使用配置的批处理大小
        distances, indices = self.index.search(query_embedding, search_k)
        

        similarities = 1 / (1 + distances[0])
        
        results = []
        for i in range(search_k):
            if i >= len(indices[0]):
                break
            entry = self.entries[indices[0][i]]
            
            # 严格执行图像相似度阈值过滤
            if entry.image_similarity < self.min_image_similarity:
                continue
                
            text_sim = float(similarities[i])
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
    subtitle_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'subtitle')
    index_dir = os.path.join(os.path.dirname(__file__), 'index')
    
    try:
        searcher = SubtitleSearch(subtitle_folder)
        
        # 检查索引文件是否存在
        if os.path.exists(index_dir):
            console.print("\n[cyan]发现已存在的索引文件[/]")
            console.print("\n1. 使用已有索引\n2. 重新构建索引")
            choice = Prompt.ask(
                "请选择",
                choices=["1", "2"],
                default="1"
            )
            
            if choice == "1":
                console.print("[cyan]正在加载索引...[/]")
                searcher.load_index(index_dir)
                console.print("[green]索引加载完成![/]")
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
                    "0. 返回搜索",
                    title="设置菜单"
                ))
                
                choice = Prompt.ask("选择要修改的设置", choices=["0", "1", "2"])
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