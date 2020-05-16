import cv2, pickle
import numpy as np
import tensorflow as tf
import os
import keras
import os

from keras.models import load_model
prediction = None
model = load_model('hand_model.h5')

size = 64, 64
font = cv2.FONT_HERSHEY_SIMPLEX
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'space', 27: 'del', 28: 'nothing'}
def keras_process_image(img):
    img = cv2.resize(img, size)
    img = np.array(img, dtype=np.float32)

    img = img.astype('float32')/255.0
    img = np.reshape(img, (1, 64, 64, 3))
    return img

def keras_predict(model, image):
    processed = keras_process_image(image)
    pred_class = model.predict_classes(processed)[0]
    return pred_class

def recognize(img):
    #img = cv2.flip(img, 1)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pred_class = keras_predict(model, img)
    return labels_dict[pred_class]

#out2 = cv2.VideoWriter('out.mp4',cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (1280, 720))
cap=cv2.VideoCapture(0)
while(cap.isOpened()):
  _,img=cap.read()
  cv2.rectangle(img,(400,100),(800,500),(255,0,0),3) # bounding box which captures ASL sign to be detected by the system
  img1=img[100:500,400:800]
  text = recognize(img1)
  if text != None:
    cv2.putText(img,str(text),(50,300), font,3,(0,0,255),2)
    cv2.imshow('Frame',img)
    #out2.write(img)
  if cv2.waitKey(25) & 0xFF == ord('q'):
      break
