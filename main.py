#from FaceRec import FaceRecognizer
from FaceRec_insightface import FaceRecognizer
from CutSubtitle_paddleocr import SubtitleExtractor
import os
import traceback
import re
from params import *

def clean_frames_folder():
    """清理帧输出文件夹中的所有文件"""
    if os.path.exists(FRAMES_OUTPUT):
        for file in os.listdir(FRAMES_OUTPUT):
            file_path = os.path.join(FRAMES_OUTPUT, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error: {e}")

def get_video_progress(video_title):
    """获取视频处理进度"""
    if not os.path.exists(FRAMES_OUTPUT):
        return None
        
    frame_files = [f for f in os.listdir(FRAMES_OUTPUT) 
                   if f.startswith(video_title + "_")]
    if not frame_files:
        return None
        
    # 从文件名中提取时间戳
    timestamps = []
    for f in frame_files:
        match = re.match(r'.*_(\d+)m(\d+)s_.*', f)
        if match:
            minutes, seconds = map(int, match.groups())
            timestamps.append(minutes * 60 + seconds)
    
    return max(timestamps) if timestamps else None

def process_videos_in_folder():
    """处理文件夹中的所有视频"""
    # 确保输出目录存在
    os.makedirs(FRAMES_OUTPUT, exist_ok=True)
    os.makedirs(SUBTITLE_OUTPUT, exist_ok=True)
    
    # 获取已经生成字幕的视频
    completed_videos = {
        os.path.splitext(f)[0] 
        for f in os.listdir(SUBTITLE_OUTPUT) 
        if f.endswith('.json')
    }
    
    while True:
        # 获取所有视频文件
        video_extensions = ('.mp4', '.avi', '.mkv', '.mov')
        video_files = [
            f for f in os.listdir(VIDEOS_FOLDER)
            if os.path.isfile(os.path.join(VIDEOS_FOLDER, f))
            and f.lower().endswith(video_extensions)
            and os.path.splitext(f)[0] not in completed_videos  # 排除已完成字幕的视频
        ]

        if not video_files:
            print("\nAll videos processed successfully!")
            break

        print(f"Found {len(video_files)} videos to process")

        # 处理每个视频
        for i, video_file in enumerate(video_files, 1):
            video_title = os.path.splitext(video_file)[0]
            video_path = os.path.join(VIDEOS_FOLDER, video_file)
            print(f"\nProcessing video {i}/{len(video_files)}: {video_file}")
            
            try:
                # 检查是否有处理进度
                progress = get_video_progress(video_title)
                if progress is not None:
                    print(f"Resuming from timestamp: {progress//60}m{progress%60}s")
                
                # 处理视频
                face_recognizer = FaceRecognizer(FEATURES_FILE)
                face_recognizer.process_video_with_params(
                    video_path=video_path,
                    output_folder=FRAMES_OUTPUT,
                    fps=1,
                    start_time=progress
                )
                
                # 处理字幕
                subtitle_extractor = SubtitleExtractor()
                subtitle_extractor.process_frames(
                    input_folder=FRAMES_OUTPUT,
                    output_folder=SUBTITLE_OUTPUT
                )
                
                # 添加到已完成列表
                completed_videos.add(video_title)
                
                # 清理帧文件
                clean_frames_folder()
                print(f"Cleaned frames for {video_file}")

            except Exception:
                print(f"Error processing {video_file}")
                traceback.print_exc()
                continue

if __name__ == "__main__":
    process_videos_in_folder()