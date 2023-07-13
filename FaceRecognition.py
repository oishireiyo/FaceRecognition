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

# Color definitions, (B, G, R)
COLOR_BLACK  = (0,   0,   0  )
COLOR_BLUE   = (255, 0,   0  )
COLOR_GREEN  = (0,   255, 0  )
COLOR_RED    = (0,   0,   255)
COLOR_CYAN   = (255, 255, 0  )
COLOR_PINK   = (255, 0,   255)
COLOR_YELLOW = (0,   255, 255)
COLOR_WHITE  = (255, 255, 255)

class SpeakerDetection(object):
    '''
    Class for speaker detections.
    For mode details, visit the following link.
    https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78
    '''
    def __init__(self, inputvideo: str, outputvideo: str, outputjson: str,
                 videoGen: bool = True, MLtype='cnn') -> None:
        # Input video and its attributes
        self.inputvideoname = inputvideo
        self.inputvideo = cv2.VideoCapture(inputvideo)
        self.length = int(self.inputvideo.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.inputvideo.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.inputvideo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.size = (width, height)
        self.framerate = self.inputvideo.get(cv2.CAP_PROP_FPS)

        # Output video
        self.videoGen = videoGen
        self.outputmoviename = outputvideo
        if self.videoGen:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            self.outputmovie = cv2.VideoWriter(outputvideo, fourcc, self.framerate, self.size)

        # List of known people
        self.knownfaceencodings, self.knownfacenames = [], []

        # Facial expression evolutions
        self.Unknown = 'Unknown'
        self.facialfeatures = {'FPS': self.framerate, 'Frame': {}}

        # Other attributes
        self.jsonfile = outputjson
        self.MLtype = MLtype # 'hog' (Histogram of Oriented) or 'cnn'

        self._PrintInformation()

    def _PrintInformation(self):
        logger.info('--------------------------------------------------')
        logger.info('CUDA availability: %s' % (dlib.DLIB_USE_CUDA))
        logger.info('Input Video:')
        logger.info('  Name: {}'.format(self.inputvideoname))
        logger.info('  Length: {} sec'.format(self.length / self.framerate))
        logger.info('  Count: {}'.format(self.length))
        logger.info('  Size: {}'.format(self.size))
        logger.info('  FPS: {}'.format(self.framerate))
        logger.info('Output Video:')
        logger.info('  Enable: {}'.format(self.videoGen))
        logger.info('  Name: {}'.format(self.inputvideoname))
        logger.info('Output JSON:')
        logger.info('  Name: {}'.format(self.jsonfile))
        logger.info('Other attrbutes:')
        logger.info('  Face recognition model: {}'.format(self.MLtype))
        logger.info('--------------------------------------------------')

    def GetPolygonArea(self, points: list) -> float:
        '''
        Calculate the area of surrounded by a given polygon.
        * p = p(x, y), p_{n+1} = p_{1}
        * S = \frac{1}{2}\left| \sum_{i=1}^{n} x_{i}y_{i+1} - x_{i+1}y_{i} \right|
        '''
        S = math.fabs(math.fsum(points[i][0] * points[i-1][1] - \
                                points[i][1] * points[i-1][0] for i in range(len(points)))) / 2.0
        return S
        
    def GetTwoPointsLength(self, point1: list, point2: list) -> float:
        '''
        Calculate the length of the two given points.
        * p = p(x, y)
        * L = \sqrt{(x_{1} - x_{2})^{2} + (y_{1} - y_{2})^{2}}
        '''
        L = math.sqrt(math.pow((point1[0] - point2[0]), 2) + \
                      math.pow((point1[1] - point2[1]), 2))
        return L

    def AppendKnownPerson(self, images: list, name: str) -> None:
        for image in images:
            if not os.path.isfile(image):
                logger.critical('No such file \'%s\' for \'%s\'' % (image, name))
                sys.exit(1)

            logger.info('{} is added as a representation of {}'.format(image, name))
            _image = face_recognition.load_image_file(image)
            _image_encoding = face_recognition.face_encodings(_image)[0]

            self.knownfaceencodings.append(_image_encoding)
            self.knownfacenames.append(name)

    def PlotFacialFeaturesEvolution(self, dict: dict) -> None:
        fig, axes = plt.subplots(len(dict[self.Unknown]), 1, squeeze=False)
        for person in dict:
            for i_feature, feature in enumerate(dict[person]):
                np_ys = np.array(dict[person][feature])
                np_xs = np.array(range(len(dict[person][feature]))) / self.framerate
                axes[i_feature, 0].plot(np_xs, np_ys, label=person)
            axes[i_feature, 0].set_ylabel(feature)
        axes[len(dict[self.Unknown])-1, 0].set_xlabel('Duration [sec]')
        axes[0, 0].legend(bbox_to_anchor=(0.5, 1.6), loc='upper center', ncol=2)

        fig.subplots_adjust(top=0.8)
        fig.savefig('Outputs/FacialFeatures.jpg')

    def DecorateFrame(self, frame: np.ndarray, features: dict, face_landmarks: dict) -> None:
        width, height = self.size
        for name in features:
            top, bottom, left, right = features[name]['BBoxTBLR']
            cv2.rectangle(frame, (left, top), (right, bottom), COLOR_RED, 2)
            cv2.rectangle(frame, (left, bottom-math.ceil(height/40)), (right, bottom), COLOR_RED, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left+math.ceil(width/200), bottom-math.ceil(height/200)), font, 0.6, COLOR_WHITE, 1)

            cv2.rectangle(frame, (left, top-2*math.ceil(height/40)), (right, top), COLOR_RED, cv2.FILLED)
            cv2.putText(frame, 'S = %.3f' % (features[name]['RelativeMouthSize']), (left+math.ceil(width/200), top-math.ceil(height/200)), font, 0.6, COLOR_WHITE, 1)
            cv2.putText(frame, 'L = %.3f' % (features[name]['RelativeMouthLength']), (left+math.ceil(width/200), top-math.ceil(height/40)-math.ceil(height/200)), font, 0.6, COLOR_WHITE, 1)

            for face_landmark in face_landmarks:
                cv2.fillPoly(frame, [np.array(face_landmark['top_lip'])], COLOR_CYAN)
                cv2.fillPoly(frame, [np.array(face_landmark['bottom_lip'])], COLOR_GREEN)
                cv2.fillPoly(frame, [np.array(face_landmark['top_lip'][7:] + face_landmark['bottom_lip'][7:])], COLOR_PINK)

    def MakeJSONFile(self) -> None:
        with open(self.jsonfile, 'w') as f:
            json.dump(self.facialfeatures, f)

    def FaceRecognitionOneFrame(self, iframe: int) -> dict:
        self.inputvideo.set(cv2.CAP_PROP_POS_FRAMES, iframe)
        features = {}
        face_landmarks = None
        ret, frame = self.inputvideo.read()
        if ret:
            face_locations = face_recognition.face_locations(frame, model=self.MLtype)
            face_encoding = face_recognition.face_encodings(frame, face_locations)
            face_landmarks = face_recognition.face_landmarks(frame, face_locations)

            nUnknown = 0
            for (top, right, bottom, left), face_encoding, face_landmark in \
                zip(face_locations, face_encoding, face_landmarks):
                # Who are they?
                matches = face_recognition.compare_faces(self.knownfaceencodings, face_encoding)
                distances = face_recognition.face_distance(self.knownfaceencodings, face_encoding)

                logger.debug(distances)

                bestmatchindex = np.argmin(distances)
                matchname = self.knownfacenames[bestmatchindex]
                if not matches[bestmatchindex]:
                    matchname = '{}{}'.format(self.Unknown, nUnknown)
                    nUnknown += 1

                # What features fo they have?
                AreaTopLip = self.GetPolygonArea(face_landmark['top_lip'])
                AreaBottomLip = self.GetPolygonArea(face_landmark['bottom_lip'])
                AreaOralCavity = self.GetPolygonArea(face_landmark['top_lip'][7:] + face_landmark['bottom_lip'][7:])
                AreaRelative = AreaOralCavity / (AreaTopLip + AreaBottomLip)
                logger.debug('Area top lip: {}, Area bottom lip: {}, Area observable oral cavity: {}, Relative area size'.format(AreaTopLip, AreaBottomLip, AreaOralCavity, AreaRelative))

                ### 3.2 The size of vertical and horizontal lip lengthes.
                LengthLipVertical = self.GetTwoPointsLength(face_landmark['top_lip'][3], face_landmark['bottom_lip'][3])
                LengthLipHOrizontal = self.GetTwoPointsLength(face_landmark['top_lip'][0], face_landmark['top_lip'][6])
                try:
                    LengthRelative = LengthLipVertical / LengthLipHOrizontal
                except ZeroDivisionError as e:
                    LengthRelative = 1
                    logger.error(e)
                except:
                    logger.critical('Unknown error was occured. Process was terminated.')
                    sys.exit(2)
                logger.debug('Length lip vertical: {}, Length lip vertical: {}, Length lip relative: {}'.format(LengthLipVertical, LengthLipVertical, LengthRelative))

                # 4. Put everything together
                features[matchname] = {}
                features[matchname]['BBoxTBLR'] = (top, bottom, left, right)
                features[matchname]['BBoxCenter'] = (math.floor((bottom+top) / 2), math.floor((right+left) / 2))
                features[matchname]['RelativeMouthSize'] = AreaRelative
                features[matchname]['RelativeMouthLength'] = LengthRelative

        logger.info('Now processing ... (%d/%d)' % (iframe+1, self.length))
        return ret, frame, features, face_landmarks

    def FaceRecognitionGivenInterval(self, interval: tuple) -> None:
        for i in range(*interval):
            ret, _, features, _ = self.FaceRecognitionOneFrame(i)
            if ret:
                self.facialfeatures['Frame'][i] = features

    def FaceRecognitionAll(self):
        for i in range(self.length):
            ret, frame, features, face_landmarks = self.FaceRecognitionOneFrame(i)
            if ret:
                self.facialfeatures['Frame'][i] = features

                if self.videoGen:
                    self.DecorateFrame(frame, features, face_landmarks)
                    self.outputmovie.write(frame)

    def CloseObjects(self):
        self.inputvideo.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    start_time = time.time()

    inputvideoname = 'UniqueSamples/Solokatsu/solokatsu_conversation.mp4'
    outputvideoname = 'Outputs/Solokatsu/processed_solokatsu_conversation.mp4'
    outputjsonname = 'Outputs/Solokatsu/processed_solokatsu_conversation.json'
    Speaker = SpeakerDetection(inputvideo=inputvideoname, outputvideo=outputvideoname,
                               outputjson=outputjsonname, videoGen=True, MLtype='cnn')
    Speaker.AppendKnownPerson(images=['UniqueSamples/Solokatsu/EguchiNoriko/EguchiNoriko0.jpg',
                                      'UniqueSamples/Solokatsu/EguchiNoriko/EguchiNoriko1.jpg',
                                      'UniqueSamples/Solokatsu/EguchiNoriko/EguchiNoriko2.jpeg'],
                              name='Eguchi Noriko')
    Speaker.AppendKnownPerson(images=['UniqueSamples/Solokatsu/ShibuyaKento/ShibuyaKento0.jpg',
                                      'UniqueSamples/Solokatsu/ShibuyaKento/ShibuyaKento1.jpg',
                                      'UniqueSamples/Solokatsu/ShibuyaKento/ShibuyaKento4.jpeg'],
                              name='Shibuya Kento')
    Speaker.AppendKnownPerson(images=['UniqueSamples/Solokatsu/KobayashiKinako/KobayashiKinako0.jpeg',
                                      'UniqueSamples/Solokatsu/KobayashiKinako/KobayashiKinako1.jpg',
                                      'UniqueSamples/Solokatsu/KobayashiKinako/KobayashiKinako2.jpg'],
                              name='Kobayashi Kinako')
    Speaker.FaceRecognitionAll()
    Speaker.MakeJSONFile()

    end_time = time.time()
    logger.info('Duration: %.4f sec' % (end_time - start_time))
