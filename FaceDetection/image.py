import argparse
import cv2
import numpy as np
import os
from PIL import Image

argsParser = argparse.ArgumentParser()
argsParser.add_argument('-i', '--image', required=True, help='path to input image')
argsParser.add_argument('-m', '--model', required=True, help='path to pre-trained caffe model')
argsParser.add_argument('-p', '--proto',required=True, help='path to caffe prototxt file')
argsParser.add_argument('-c', '--confidence', default=0.5, type=float, help='minimum probability to disable weak prediction')

args = vars(argsParser.parse_args())

#Load model from cv2 deep neural network
#Using given model and prototxt files
detector = cv2.dnn.readNetFromCaffe(args['proto'], args['model'])

#Load input image and resize to 300x300 (default to model)
img = cv2.imread(args['image'])
(h, w) = img.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(img, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))

#Detect faces on image
print('Detecting...')
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

print('Done predicting!')

#Convert image to PIL and save into './predictions directory
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_pil = Image.fromarray(img)

filename = args['image']
if('/' in filename):
  filename = filename[filename.rindex('/')+1 : ]
newFilename = filename[:filename.rindex('.')] + '_prediction' + filename[filename.rindex('.'):]

if(not os.path.exists('./predictions')):
  try:
    os.mkdir('./predictions')
  except OSError:
    print('Creation of directory failed')

path = './predictions/'+newFilename
img_pil.save(path)

print('Prediction can be seen in ' + path)