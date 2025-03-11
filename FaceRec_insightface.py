import cv2
import numpy as np
import os
import traceback
from typing import Optional, List
from insightface.app import FaceAnalysis
from numpy.typing import NDArray
from PIL import Image
from params import USE_GPU_FACE


def cosine_similarity(A: NDArray, B: NDArray) -> NDArray:
    """计算两组向量间的余弦相似度
    Args:
        A: shape (n_samples_1, n_features)
        B: shape (n_samples_2, n_features)
    Returns:
        相似度矩阵: shape (n_samples_1, n_samples_2)
    """
    # 计算 L2 范数
    norm_A = np.linalg.norm(A, axis=1, keepdims=True)
    norm_B = np.linalg.norm(B, axis=1, keepdims=True)
    
    # 归一化向量
    A_normalized = A / norm_A
    B_normalized = B / norm_B
    
    # 计算余弦相似度
    return np.dot(A_normalized, B_normalized.T)


class FaceRecognizer:
    def __init__(self, features_file: str) -> None:
        providers = ['CPUExecutionProvider']
        if USE_GPU_FACE:
            providers.insert(0, 'CUDAExecutionProvider')
            
        self.app = FaceAnalysis(
            name='buffalo_l',
            providers=providers
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
    
        self.stored_features = np.load(features_file)
        self.known_face_encodings = self.stored_features['encodings']
        
        if USE_GPU_FACE:
            import torch
            self.known_face_encodings = torch.tensor(
                self.known_face_encodings, 
                device='cuda'
            ).float()
            self.use_cuda = True
        else:
            self.use_cuda = False


    def load_features(self, features_file: str) -> None:
        try:
            data = np.load(features_file)
            self.known_face_encodings = data['encodings']
            print(f"Successfully loaded {len(self.known_face_encodings)} face features")
        except Exception as e:
            print(f"Error loading features:")
            traceback.print_exc()
            raise

    def extract_face_encodings(self, image: NDArray) -> List[NDArray]:
        """提取人脸特征编码"""
        try:
            faces = self.app.get(image)
            return [face.embedding for face in faces] if faces else []
        except Exception as e:
            print("Error extracting face encodings:")
            traceback.print_exc()
            return []

    def get_face_similarity(self, frame: NDArray) -> Optional[float]:
        """计算人脸相似度"""
        face_encodings = self.extract_face_encodings(frame)
        if not face_encodings or len(self.known_face_encodings) == 0:
            return None

        if self.use_cuda:
            import torch
            face_encodings_tensor = torch.tensor(
                np.vstack(face_encodings), 
                device='cuda'
            ).float()

            similarities = torch.nn.functional.cosine_similarity(
                self.known_face_encodings.unsqueeze(1),
                face_encodings_tensor.unsqueeze(0),
                dim=2
            )
            return float(similarities.max().cpu().item())
        else:

            similarities = cosine_similarity(
                self.known_face_encodings,
                np.vstack(face_encodings)
            )
            return float(np.max(similarities))

    def process_video(self, video_path: str, fps: int = 1, save_frames_folder: str = "output_frames", start_time: Optional[int] = None) -> None:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        os.makedirs(save_frames_folder, exist_ok=True)
        video_title = os.path.splitext(os.path.basename(video_path))[0]

        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                raise RuntimeError("Error: Cannot open video file.")

            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            interval = max(1, int(frame_rate / fps))
            frame_count = 0
            
            # 如果有起始时间，先定位到对应位置
            if start_time is not None:
                frame_count = int(start_time * frame_rate)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    if frame_count % interval == 0:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self._process_frame(
                            frame_rgb, frame_count, frame_rate, 
                            video_title, save_frames_folder
                        )
                except Exception as e:
                    print(f"Error processing frame {frame_count}:")
                    traceback.print_exc()

                frame_count += 1

        except Exception as e:
            print("Error processing video:")
            traceback.print_exc()
        finally:
            cap.release()

        print(f"Processing completed. Frames saved to {save_frames_folder}")

    def _process_frame(
        self, 
        frame: NDArray, 
        frame_count: int, 
        frame_rate: float,
        video_title: str,
        save_frames_folder: str
    ) -> None:
        time_sec = frame_count / frame_rate
        minutes, seconds = divmod(int(time_sec), 60)
        
        similarity = self.get_face_similarity(frame) or 0.0
        
        frame_filename = os.path.join(
            save_frames_folder,
            f"{video_title}_{minutes}m{seconds:02d}s_sim_{similarity:.3f}.jpg"
        )
        
        # Convert numpy array to PIL Image and save
        Image.fromarray(frame).save(frame_filename)
        print(f"Frame {frame_count}: {minutes}m{seconds:02d}s - similarity = {similarity:.3f}")

    def process_video_with_params(self, video_path: str, output_folder: str, fps: int = 1, start_time: Optional[int] = None) -> None:
        self.process_video(video_path, fps, output_folder, start_time)