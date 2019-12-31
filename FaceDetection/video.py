import argparse
import cv2
import numpy as np
import time
import imutils
from imutils.video import VideoStream

argsParser = argparse.ArgumentParser()
argsParser.add_argument('-m', '--model', required=True, help='path to pre-trained caffe model')
argsParser.add_argument('-p', '--proto', required=True, help='path to prototxt file')
argsParser.add_argument('-c', '--confidence', default=0.5, type=float, help='minimum confidence to filter low confidence prediction')

args = vars(argsParser.parse_args())

#Load model from cv2 deep neural network
#Using given model and prototxt files
detector = cv2.dnn.readNetFromCaffe(args['proto'], args['model'])

#Initialize video stream
print('Initializing video stream...')
vs = VideoStream(src=0).start()
time.sleep(2.0)

#Detect face(s) in  each frame
while(True):
  img = vs.read()
  (h, w) = img.shape[:2]
  blob = cv2.dnn.blobFromImage(cv2.resize(img, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))

  detector.setInput(blob)
  result = detector.forward()

  #Draw bounding boxes
  for i in range(0, result.shape[2]):
    confidence = result[0, 0, i, 2]
    if(confidence > args['confidence']):
      box = result[0, 0, i, 3:7] * np.array([w, h, w, h])
      (xBegin, yBegin, xEnd, yEnd) = box.astype('int')

      desc = 'confidence : ' + '{:.2f}%'.format(confidence * 100)
      
      y = yBegin - 10
      if(y <= 10):
        y += 20
      cv2.rectangle(img, (xBegin, yBegin), (xEnd, yEnd), (0, 255, 0), 2)
      cv2.putText(img, desc, (xBegin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

  #Show result
  cv2.imshow('Prediction', img)
  key = cv2.waitKey(1) & 0xFF

  if(key == ord('q')):
    break

cv2.destroyAllWindows()
vs.stop()