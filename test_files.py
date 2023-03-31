from os import listdir
import cv2
directory = './Changed_2/'
#for filename in listdir('C:/tensorflow/models/research/object_detection/images/train'):
for filename in listdir(directory):
  if filename.endswith(".png"):
    print(directory+filename)
    #cv2.imread('C:/tensorflow/models/research/object_detection/images/train/'+filename)
    cv2.imread(directory+filename)