import os
import struct
from pathlib import Path
import json
from tqdm import tqdm
import concurrent.futures
import threading

write_lock = threading.Lock()

def process_file(args):
    file_path, idx, grid_size = args
    row = idx // grid_size[0]
    col = idx % grid_size[0]
    with open(file_path, 'rb') as infile:
        content = infile.read()
        file_size = len(content)
    return content, file_size, row, col

def create_binary_index(mapping_data, index_file):
    files = []
    for file_path, info in mapping_data['files'].items():
        folder = int(file_path.split('/')[0])
        frame_num = int(file_path.split('/')[1].split('_')[1].split('.')[0])
        offset = info['offset']
        assert 0 <= folder <= 0xFFFFFFFF, f"Folder ID out of range: {folder}"
        assert 0 <= frame_num <= 0xFFFFFFFF, f"Frame number out of range: {frame_num}"
        assert 0 <= offset <= 0xFFFFFFFFFFFFFFFF, f"Offset out of range: {offset}"
        files.append((folder, frame_num, offset))
    files.sort()
    with open(index_file, 'wb') as f:
        f.write(struct.pack('<II', *mapping_data['grid_size']))
        unique_folders = sorted(set(x[0] for x in files))
        f.write(struct.pack('<I', len(unique_folders)))
        for folder in unique_folders:
            f.write(struct.pack('<I', folder))
        f.write(struct.pack('<I', len(files)))
        for folder_id, frame_num, offset in files:
            f.write(struct.pack('<IIQ', folder_id, frame_num, offset))
    with open(index_file, 'rb') as f:
        grid_w, grid_h = struct.unpack('<II', f.read(8))
        folder_count = struct.unpack('<I', f.read(4))[0]
        folders = [struct.unpack('<I', f.read(4))[0] for _ in range(folder_count)]
        file_count = struct.unpack('<I', f.read(4))[0]
def read_frame_offset(index_file, folder_id, frame_num):
    with open(index_file, 'rb') as f:
        data = f.read()
        f.seek(0)
        grid_w, grid_h = struct.unpack('<II', f.read(8))
        folder_count = struct.unpack('<I', f.read(4))[0]
        folders = [struct.unpack('<I', f.read(4))[0] for _ in range(folder_count)]
        file_count = struct.unpack('<I', f.read(4))[0]
        print(f"{grid_w}x{grid_h}")
        print(f"{folders}")
        print(f"{file_count}")
        index_start = f.tell()
        left, right = 0, file_count - 1
        while left <= right:
            mid = (left + right) // 2
            f.seek(index_start + mid * 16)
            curr_folder, curr_frame, curr_offset = struct.unpack('<IIQ', f.read(16))
            if curr_folder == folder_id and curr_frame == frame_num:
                next_offset = None
                if mid < file_count - 1:
                    f.seek(index_start + (mid + 1) * 16)
                    next_offset = struct.unpack('<IIQ', f.read(16))[2]
                return curr_offset, next_offset
            elif curr_folder < folder_id or (curr_folder == folder_id and curr_frame < frame_num):
                left = mid + 1
            else:
                right = mid - 1
        return None, None

def combine_files(input_folders, output_file, index_file, grid_size=(60, 60)):
    mapping = {
        'grid_size': grid_size,
        'folders': input_folders,
        'files': {}
    }
    current_offset = 0
    with open(output_file, 'wb') as outfile:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            for folder in input_folders:
                folder_path = os.path.join("../frames", str(folder))
                if not os.path.exists(folder_path):
                    continue

                webp_files = [f for f in os.listdir(folder_path) if f.endswith('.webp')]
                webp_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # 修改排序逻辑

                if not webp_files:
                    continue

                total_files = min(len(webp_files), grid_size[0] * grid_size[1])
                process_args = [
                    (os.path.join(folder_path, webp_file), idx, grid_size)
                    for idx, webp_file in enumerate(webp_files[:total_files])
                ]
                futures = []
                for args in process_args:
                    future = executor.submit(process_file, args)
                    futures.append((future, webp_files[process_args.index(args)]))
                for future, webp_file in tqdm(futures, desc=f"处理文件夹 {folder}"):
                    content, file_size, row, col = future.result()

                    with write_lock:
                        outfile.write(content)
                        frame_num = int(webp_file.split('_')[1].split('.')[0])
                        mapping['files'][f"{folder}/{webp_file}"] = {
                            'offset': current_offset,
                            'size': file_size,
                            'position': [col, row]
                        }
                        current_offset += file_size
    create_binary_index(mapping, index_file)


def process_folder_groups():
    frames_dir = "../frames"
    folders = sorted([int(f) for f in os.listdir(frames_dir)
                      if os.path.isdir(os.path.join(frames_dir, f)) and f.isdigit()])

    current_group = []
    current_group_start = 1

    for folder in folders:
        if folder < current_group_start + 10 and folder >= current_group_start:
            current_group.append(folder)
        else:
            if current_group:
                group_index = (current_group_start - 1) // 10
                output_file = f"combined_{group_index}.bin"
                index_file = f"index_{group_index}.bin"
                combine_files(current_group, output_file, index_file)

            current_group = [folder]
            current_group_start = (folder - 1) // 10 * 10 + 1
    if current_group:
        group_index = (current_group_start - 1) // 10
        output_file = f"{group_index}.webp"
        index_file = f"{group_index}.index"
        combine_files(current_group, output_file, index_file)


def extract_frame(combined_file, index_file, folder_id, frame_num):
    start_offset, end_offset = read_frame_offset(index_file, folder_id, frame_num)
    if start_offset is None:
        return None
    with open(combined_file, 'rb') as f:
        f.seek(start_offset)
        if end_offset is None:
            return f.read()
        else:
            return f.read(end_offset - start_offset)


if __name__ == "__main__":
    process_folder_groups()

    folder_id = 273
    frame_num = 10
    group_index = (folder_id - 1) // 10

    combined_file = f"{group_index}.webp"
    index_file = f"{group_index}.index"
    frame_data = extract_frame(combined_file, index_file, folder_id, frame_num)
    if frame_data:
        output_path = f"frame_{folder_id}_{frame_num}.webp"
        with open(output_path, 'wb') as f:
            f.write(frame_data)
        print(f"{output_path}")