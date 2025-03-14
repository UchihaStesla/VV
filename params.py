# 路径配置
VIDEOS_FOLDER = "Videos"  # VV视频的文件夹
FEATURES_FILE = "face_features_insightface.npz"  # 人脸特征向量文件
FRAMES_OUTPUT = "output_frames"  # 帧文件夹
SUBTITLE_OUTPUT = "subtitle"  # 字幕输出文件夹
FACE_IMAGES_FOLDER = "target"  # VV人脸图片文件夹

# GPU配置
USE_GPU_FACE = False  # 是否在人脸识别中使用GPU
USE_GPU_OCR = False   # 是否在OCR中使用GPU
GPU_MEMORY_OCR = 500  # OCR的GPU内存限制(MB)

# OCR模型配置
OCR_MODEL_DIR = "ch_PP-OCRv4_rec_infer"  # OCR模型目录
