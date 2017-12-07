from __future__ import print_function
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pylab as plt
import pickle
SIZE_FACE = 48
NUM_CLASSES = 7
NUM_TRAIN = 28708
NUM_TEST = 7179
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
# 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

def emotion_to_vec(x):
    d = np.zeros(len(EMOTIONS))
    d[x] = 1.0
    return d

def data_to_image(data):
    # print data
    data_image = np.fromstring(str(data), dtype=np.uint8, sep=' ').reshape((SIZE_FACE, SIZE_FACE))
    data_image = Image.fromarray(data_image).convert('RGB')
    data_image = np.array(data_image)[:, :, ::-1].copy()
    # data_image = format_image(data_image)
    return data_image

def readImages(FILE_PATH):
    data = pd.read_csv(FILE_PATH)

    labels = []
    images = []
    total = data.shape[0]
    for index, row in data.iterrows():
        emotion = emotion_to_vec(row['emotion'])
        image = data_to_image(row['pixels'])
        if image is not None:
            labels.append(emotion)
            images.append(image)
        else:
            print("Error")
        index += 1
        print("Progress: {}/{} {:.2f}%".format(index, total, index * 100.0 / total))

    list_images = []
    for item in images:
        list_images.append(cv2.cvtColor(item, cv2.COLOR_RGB2GRAY))

    return list_images,labels
    # print("Total: " + str(len(images)))
    # print(gray)
    # cv2.imshow("aaa", gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':

    # img = cv2.imread('sift-scene.jpg')
    # print(img)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(gray)

    # sift = cv2.xfeatures2d.SIFT_create()
    # kp = sift.detect(gray, None)
    # cv2.drawKeypoints(gray, kp, img)
    # cv2.imwrite('sift_keypoints.jpg', img)
    #

    # read list images
    listimages,listlabel = readImages('fer2013.csv')
    filehandler = open("ximg.pt", 'wb')
    pickle.dump(listimages, filehandler)
    filehandler = open("yimg.pt", 'wb')
    pickle.dump(listlabel, filehandler)


    # img_x, img_y = SIZE_FACE, SIZE_FACE
    # input_shape = (img_x, img_y, 1)
    #
    # x_img = open('ximg.pt', 'rb')
    # list_images = pickle.load(x_img)
    # list_images = np.asarray(list_images)
    #
    # x_train = list_images[0:NUM_TRAIN]
    # x_test = list_images[NUM_TRAIN:NUM_TRAIN+NUM_TEST]
    # x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
    # x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')
    # x_train /= 255
    # x_test /= 255
    # print('x_train shape:', x_train.shape)
    # print(x_train.shape[0], 'train samples')
    # print(x_test.shape[0], 'test samples')
    #
    # y_img = open('yimg.pt', 'rb')
    # list_labels = pickle.load(y_img)
    # list_labels = np.asarray(list_labels)
    #
    # y_train = list_labels[0:NUM_TRAIN]
    # y_test = list_labels[NUM_TRAIN:NUM_TRAIN+NUM_TEST]
    # print(y_train[3])