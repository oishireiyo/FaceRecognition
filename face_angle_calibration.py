# Standard python modules
import os
import sys
import time

# Logging
import logging
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
handler_format = logging.Formatter('%(asctime)s : [%(name)s - %(lineno)d] %(levelname)-8s - %(message)s')
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)

# Advanced modules
import numpy as np
import cv2
import dlib
import face_recognition

# Handmade modules
import colors
from face_manipulation import FaceLandmarksCalibration

class calibrate_face_angles_aho(object):
    '''
    顔の向きを計測するため、まず初めに正体した顔が写った画像を使って顔の角度の校正を以下の手順で行う必要がある。
        1. 同一の画像に対して、face_recognition & media-pipeを使ってランドマークを検出する。
        2, face_recognitionで取得したランドマークとmedia-pipeで取得したランドマークのユークリッド距離を計算し対応関係を取得する。
        3. media-pipeで提供されている一般的な顔の3次元情報から近似的にface_recognnitionのランドマークの3次元情報に変換する。
        4. 適当な3点で形成される平面に対応する法線ベクトルを計算し、pitch, yaw, rollに変換する。
        5. 画像が正体していると仮定しているので、4で取得したpitchm yawm rollが0になるように校正する。

    face_recognitionとmedia-pipeの関係
    ---------------------------------------------------------
                             | face_recognition | media-pipe
    ---------------------------------------------------------
    3次元のランドマーク情報を持つ | No               | Yes
    ---------------------------------------------------------
    recognitionができる       | Yes              | No
    ---------------------------------------------------------
    '''
    def __init__(self,
                 landmark_indices: list=[
                    ('nose_bridge', 0), ('top_lip', 0), ('bottom_lip', 0),
                 ],
                 facial_point_file_path: str='canonical_face_model/canonical_face_model.obj',
                 calibrated_correspondences: any=None,
                 calibration_image_path: str='../calibration/model10211041_TP_V4.jpg'
    ):
        # 法線ベクトルの算出に使用するlandmark
        self.landmark_indices = landmark_indices

        # 一般的な顔の3次元情報を格納
        self.facial_point_file_path = facial_point_file_path
        self.facial_points_3d = []

        # Calibrationの対象画像
        self.correspondences = calibrated_correspondences
        if self.correspondences is None:
            calibration_image = cv2.imread(calibration_image_path)
            self.correspondences = FaceLandmarksCalibration().compare(image=calibration_image, output_name='calibration.png')

    def parse_canonical_facial_points_3d(self):
        '''
        適当な3点に対応する3次元顔情報を取得する。
        '''
        with open(self.facial_point_file_path, mode='r') as f:
            lines = f.readlines()
            for key, index in self.landmark_indices:
                line_number = self.correspondences[key][index] # landmarkはindex=0からスタートしていることに注意
                elements = lines[line_number].split()
                self.facial_points_3d.append(
                    (float(elements[1]), float(elements[2]), float(elements[3].replace('\n', ''))))
        self.facial_points_3d = np.array(self.facial_points_3d, dtype=np.float32)

    def calibrate(self):
        '''
        適当な3点から互いに並行でない2つのベクトルを計算する。
        '''
        vec_3d_1 = self.facial_points_3d[0] - self.facial_points_3d[1]
        vec_3d_2 = self.facial_points_3d[0] - self.facial_points_3d[2]
        normal_vector = self.get_normal_vector(vec_3d_1=vec_3d_1, vec_3d_2=vec_3d_2)
        return normal_vector

    def get_normal_vector(self, vec_3d_1: np.ndarray, vec_3d_2: np.ndarray) -> np.ndarray:
        '''
        2ベクトルが作る平面に直行するベクトルを計算する。
        '''
        return np.cross(vec_3d_1, vec_3d_2)

if __name__ == '__main__':
    calibrator = calibrate_face_angles()
    calibrator.parse_canonical_facial_points_3d()
    n = calibrator.calibrate()
    logger.info(n)# Standard python modules
import os
import sys
import time

# Logging
import logging
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
handler_format = logging.Formatter('%(asctime)s : [%(name)s - %(lineno)d] %(levelname)-8s - %(message)s')
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)

# Advanced modules
import numpy as np
import cv2
import dlib
import face_recognition

# Handmade modules
import colors
from face_manipulation import FaceLandmarksCalibration

class calibrate_face_angles_aho(object):
    '''
    顔の向きを計測するため、まず初めに正体した顔が写った画像を使って顔の角度の校正を以下の手順で行う必要がある。
        1. 同一の画像に対して、face_recognition & media-pipeを使ってランドマークを検出する。
        2, face_recognitionで取得したランドマークとmedia-pipeで取得したランドマークのユークリッド距離を計算し対応関係を取得する。
        3. media-pipeで提供されている一般的な顔の3次元情報から近似的にface_recognnitionのランドマークの3次元情報に変換する。
        4. 適当な3点で形成される平面に対応する法線ベクトルを計算し、pitch, yaw, rollに変換する。
        5. 画像が正体していると仮定しているので、4で取得したpitchm yawm rollが0になるように校正する。

    face_recognitionとmedia-pipeの関係
    ---------------------------------------------------------
                             | face_recognition | media-pipe
    ---------------------------------------------------------
    3次元のランドマーク情報を持つ | No               | Yes
    ---------------------------------------------------------
    recognitionができる       | Yes              | No
    ---------------------------------------------------------
    '''
    def __init__(self,
                 landmark_indices: list=[
                    ('nose_bridge', 0), ('top_lip', 0), ('bottom_lip', 0),
                 ],
                 facial_point_file_path: str='canonical_face_model/canonical_face_model.obj',
                 calibrated_correspondences: any=None,
                 calibration_image_path: str='../calibration/model10211041_TP_V4.jpg'
    ):
        # 法線ベクトルの算出に使用するlandmark
        self.landmark_indices = landmark_indices

        # 一般的な顔の3次元情報を格納
        self.facial_point_file_path = facial_point_file_path
        self.facial_points_3d = []

        # Calibrationの対象画像
        self.correspondences = calibrated_correspondences
        if self.correspondences is None:
            calibration_image = cv2.imread(calibration_image_path)
            self.correspondences = FaceLandmarksCalibration().compare(image=calibration_image, output_name='calibration.png')

    def parse_canonical_facial_points_3d(self):
        '''
        適当な3点に対応する3次元顔情報を取得する。
        '''
        with open(self.facial_point_file_path, mode='r') as f:
            lines = f.readlines()
            for key, index in self.landmark_indices:
                line_number = self.correspondences[key][index] # landmarkはindex=0からスタートしていることに注意
                elements = lines[line_number].split()
                self.facial_points_3d.append(
                    (float(elements[1]), float(elements[2]), float(elements[3].replace('\n', ''))))
        self.facial_points_3d = np.array(self.facial_points_3d, dtype=np.float32)

    def calibrate(self):
        '''
        適当な3点から互いに並行でない2つのベクトルを計算する。
        '''
        vec_3d_1 = self.facial_points_3d[0] - self.facial_points_3d[1]
        vec_3d_2 = self.facial_points_3d[0] - self.facial_points_3d[2]
        normal_vector = self.get_normal_vector(vec_3d_1=vec_3d_1, vec_3d_2=vec_3d_2)
        return normal_vector

    def get_normal_vector(self, vec_3d_1: np.ndarray, vec_3d_2: np.ndarray) -> np.ndarray:
        '''
        2ベクトルが作る平面に直行するベクトルを計算する。
        '''
        return np.cross(vec_3d_1, vec_3d_2)

if __name__ == '__main__':
    calibrator = calibrate_face_angles()
    calibrator.parse_canonical_facial_points_3d()
    n = calibrator.calibrate()
    logger.info(n)