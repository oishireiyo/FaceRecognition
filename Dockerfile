FROM python:3

RUN mkdir -p /app/FaceRecognition && mkdir -p /app/Inputs/Videos/
WORKDIR /app/FaceRecognition

RUN ls /app/FaceRecognition && ls /app/Inputs/Videos/

# COPY ../Inputs/Videos/Solokatsu.mp4 /app/Inputs/Videos/

RUN echo "Hello World."
RUN pip install face-recognition

CMD ["python", "FaceRecognition.py"]