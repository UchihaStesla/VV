import cv2
import numpy as np
import os
from tqdm import tqdm
from insightface.app import FaceAnalysis
from PIL import Image
import traceback
from params import FACE_IMAGES_FOLDER, FEATURES_FILE, USE_GPU_FACE

class FeatureGenerator:
    def __init__(self):
        # 加载insightface模型，指定具体模型
        self.model = FaceAnalysis(name='buffalo_l')
        self.model.prepare(ctx_id=0 if USE_GPU_FACE else -1, det_size=(640, 640), det_thresh=0.5)

    def extract_face_encoding(self, image):
        # 图像预处理
        if image.shape[0] > 2000 or image.shape[1] > 2000:
            scale = min(2000/image.shape[0], 2000/image.shape[1])
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

        # 检测人脸并提取特征
        faces = self.model.get(image)
        
        if not faces:
            return None
            
        # 如果有多个人脸，选择最大的人脸
        if len(faces) > 1:
            face_areas = [face.bbox[2] * face.bbox[3] for face in faces]
            largest_face = faces[np.argmax(face_areas)]
            return largest_face.embedding
            
        return faces[0].embedding

    def generate_features(self):
        face_encodings = []
        
        # 获取所有图片文件
        image_files = [f for f in os.listdir(FACE_IMAGES_FOLDER) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Processing {len(image_files)} images...")
        
        # 使用tqdm显示进度条
        for image_file in tqdm(image_files):
            image_path = os.path.join(FACE_IMAGES_FOLDER, image_file)
            try:
                # 使用PIL读取图片
                pil_img = Image.open(image_path)
                # 转换为RGB模式（如果是RGBA或其他模式）
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                # 转换为OpenCV格式
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
                face_encoding = self.extract_face_encoding(img)
                if face_encoding is not None:
                    face_encodings.append(face_encoding)
            except Exception as e:
                print(f"Error processing {image_file}:")
                traceback.print_exc()
                
        try:
            # 保存特征向量
            if face_encodings:
                np.savez(FEATURES_FILE,
                        encodings=np.array(face_encodings))
                print(f"\nSuccessfully generated features for {len(face_encodings)} images")
                print(f"Features saved to {FEATURES_FILE}")
            else:
                print("No valid faces found in the images")
        except Exception as e:
            print("Error saving features file:")
            traceback.print_exc()

if __name__ == "__main__":
    generator = FeatureGenerator()
    generator.generate_features()
