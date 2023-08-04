# Standard python modules
import os
import sys
import time
import math
import json
import copy

# Logging
import logging
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
handler_format = logging.Formatter('%(asctime)s : [%(name)s - %(lineno)d] %(levelname)-8s - %(message)s')
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)

# Advanced python modules
import numpy as np
import dlib
import matplotlib.pyplot as plt
import face_recognition
import cv2

class FaceLandmarks3D(object):
    def __init__(self, input_image: str, output_image: str, MLtype: str = 'cnn') -> None:
        # Input image and its attributes
        self.input_image_name = input_image
        self.input_image = cv2.imread(input_image)
        self.height, self.width, _ = self.input_image.shape

        # Output image
        self.output_image_name = output_image

        # Face recognition
        self.MLtype = MLtype

    def _print_information(self):
        logger.info('-' * 50)
        logger.info('CUDA availability: %s' % (dlib.DLIB_USE_CUDA))
        logger.info('Input Image:')
        logger.info('  Name: {}'.format(self.input_image_name))
        logger.info('  Height: {}'.format(self.height))
        logger.info('  Width: {}'.format(self.width))
        logger.info('Output Image:')
        logger.info('  Name: {}'.format(self.output_image_name))
        logger.info('-' * 50)

    def get_landmarks(self):
        face_locations = face_recognition.face_locations(self.input_image, model = self.MLtype)
        face_landmarks = face_recognition.face_landmarks(self.input_image, face_locations)

        return face_landmarks
    
    def plot_3D_landmarks(self):
        face_landmarks = self.get_landmarks()
        print(face_landmarks)


if __name__ == '__main__':
    start_time = time.time()

    input_image = '../Inputs/Images/Kanna_Hashimoto.jpg'
    output_image = '../Outputs/Images/Kanna_Hashimoto_landmark3D.jpg'
    landmarker = FaceLandmarks3D(input_image = input_image, output_image = output_image)
    landmarker.plot_3D_landmarks()

    end_time = time.time()
    logger.info('Duration: %.4f sec' % (end_time - start_time))
