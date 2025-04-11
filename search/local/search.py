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

from mapping import get_video_url

# 配置参数
USE_GPU_SEARCH = False  # 是否在SentenceTransformer中使用GPU
SEARCH_BATCH_SIZE = 256  # 索引构建时的批处理大小

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
    id: int = None

class SubtitleSearch:
    def __init__(self, subtitle_folder, model_name='BAAI/bge-m3'):
        self.subtitle_folder = subtitle_folder
        self.use_gpu = USE_GPU_SEARCH
        self.model = SentenceTransformer(model_name, device='cuda' if self.use_gpu else 'cpu')
        self.entries = []
        self.min_image_similarity = 0.6
        self.search_k = 5
        self.index = None
        self.min_text_similarity = 0.5
    
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

    def load_subtitle_file(self, filename):
        """加载单个字幕文件"""
        filepath = os.path.join(self.subtitle_folder, filename)
        entries = []
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
            video_name = filename[:-5]  # 去除.json后缀
            for entry in data:
                entries.append(SubtitleEntry(
                    text=entry['text'],
                    timestamp=entry['timestamp'],
                    filename=video_name,
                    image_similarity=entry.get('similarity', 0.0)
                ))
        return entries

    def create_index(self):
        texts = [entry.text for entry in self.entries]
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=SEARCH_BATCH_SIZE
        )
        dimension = embeddings.shape[1]
        # 规范化向量以使用内积计算余弦相似度
        faiss.normalize_L2(embeddings)
        base_index = faiss.IndexFlatIP(dimension)  # 使用内积代替L2距离
        idmap = faiss.IndexIDMap(base_index)
        # 为每个条目分配 id（使用 顺序编号 ）
        for i, entry in enumerate(self.entries):
            entry.id = int(i)
        id_array = np.arange(len(self.entries), dtype=np.int64)
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
                    new_entries.append(exist)
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
        
        # 对全新条目分配新的 id
        existing_ids = [entry.id for entry in self.entries if entry.id is not None]
        max_id = max(existing_ids) if existing_ids else -1
        for entry in incremental_entries:
            max_id += 1
            entry.id = max_id
        
        # 针对文本变化的条目，逐条更新：
        if changed_entries:
            for entry in tqdm(changed_entries, desc="更新已变更条目"):
                # 删除旧向量
                self.index.remove_ids(np.array([entry.id], dtype=np.int64))
                # 计算更新后的向量并添加
                embedding = self.model.encode([entry.text])
                faiss.normalize_L2(embedding)
                self.index.add_with_ids(embedding, np.array([entry.id], dtype=np.int64))
        
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
        
        # 计算删除的条目数
        deleted_count = len(existing_map) - len([e for e in new_entries if e.id in [ex.id for ex in existing_map.values()]])
        
        self.entries = new_entries
        
        return {
            'old_count': len(existing_map),
            'new_count': len(self.entries),
            'diff': len(self.entries) - len(existing_map),
            'new_entries': len(incremental_entries),
            'deleted': deleted_count
        }

    def check_index_integrity(self):
        """检查索引完整性并修复"""
        # 获取entries中的所有ID
        entry_ids = set(entry.id for entry in self.entries if entry.id is not None)
        
        # 获取FAISS索引中的所有ID
        all_ids = np.arange(self.index.ntotal).astype('int64')
        _, faiss_ids = self.index.search(np.zeros((1, self.index.d), dtype='float32'), self.index.ntotal)
        faiss_ids = set(faiss_ids[0])
        
        # 比较差异
        missing_in_faiss = entry_ids - faiss_ids
        extra_in_faiss = faiss_ids - entry_ids
        
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
            entry = self.entries[indices[0][i]]
            
            if entry.image_similarity < self.min_image_similarity:
                continue
                
            text_sim = float(similarities[0][i])  # 直接使用内积结果作为相似度
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
            console.print("\n1. 使用已有索引\n2. 更新索引\n3. 重新构建索引")
            choice = Prompt.ask(
                "请选择",
                choices=["1", "2", "3"],
                default="1"
            )
            
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
                    "3. 检查索引完整性\n"
                    "0. 返回搜索",
                    title="设置菜单"
                ))
                
                choice = Prompt.ask("选择要修改的设置", choices=["0", "1", "2", "3"])
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