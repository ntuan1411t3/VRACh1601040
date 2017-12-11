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
from keras.models import load_model

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
    labels_emotion = []
    total = data.shape[0]
    for index, row in data.iterrows():
        emo = EMOTIONS[row['emotion']]
        emotion = emotion_to_vec(row['emotion'])
        image = data_to_image(row['pixels'])
        if image is not None:
            labels.append(emotion)
            images.append(image)
            labels_emotion.append(emo)
        else:
            print("Error")
        index += 1
        print("Progress: {}/{} {:.2f}%".format(index, total, index * 100.0 / total))

    list_images = []
    for item in images:
        list_images.append(cv2.cvtColor(item, cv2.COLOR_RGB2GRAY))

    return list_images, labels, labels_emotion


if __name__ == '__main__':
    # test SIFT
    img = cv2.imread('/home/hh/imgvratest/happy_29410.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()

    kp, des = sift.detectAndCompute(gray,None)
    print(len(kp))
    print(des[0])
    print(len(des))
    print(len(des[0]))
    print(kp[0].angle)

    cv2.drawKeypoints(gray, kp, outImage=img)
    cv2.imwrite('abc.jpg', img)

    # create data
    # list_images, list_labels, list_emotions = readImages('fer2013.csv')
    # file_handler = open("ximg.pt", 'wb')
    # pickle.dump(list_images, file_handler)
    # file_handler = open("yimg.pt", 'wb')
    # pickle.dump(list_labels, file_handler)
    # file_handler = open("limg.pt", 'wb')
    # pickle.dump(list_emotions, file_handler)

    # write images
    # x_img = open('ximg.pt', 'rb')
    # list_images = pickle.load(x_img)
    # list_images = np.asarray(list_images)
    # x_train = list_images[0:NUM_TRAIN]
    # x_test = list_images[NUM_TRAIN:NUM_TRAIN + NUM_TEST]
    #
    # l_img = open('limg.pt', 'rb')
    # list_labels = pickle.load(l_img)
    # list_labels = np.asarray(list_labels)
    # l_train = list_labels[0:NUM_TRAIN]
    # l_test = list_labels[NUM_TRAIN:NUM_TRAIN + NUM_TEST]
    #
    # index = 0
    # for item, label in zip(x_train, l_train):
    #     cv2.imwrite("/home/hh/imgvratrain/" + str(label) + "_" + str(index) + ".jpg", item)
    #     index += 1
    #
    # for item, label in zip(x_test, l_test):
    #     cv2.imwrite("/home/hh/imgvratest/" + str(label) + "_" + str(index) + ".jpg", item)
    #     index += 1

    # # prepare data
    # img_x, img_y = SIZE_FACE, SIZE_FACE
    # input_shape = (img_x, img_y, 1)
    #
    # x_img = open('ximg.pt', 'rb')
    # list_images = pickle.load(x_img)
    # list_images = np.asarray(list_images)
    # x_train = list_images[0:NUM_TRAIN]
    # x_test = list_images[NUM_TRAIN:NUM_TRAIN + NUM_TEST]
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
    # y_test = list_labels[NUM_TRAIN:NUM_TRAIN + NUM_TEST]
    # #
    # # #build a model
    # batch_size = 128
    # num_classes = 7
    # epochs = 10
    #
    # model = Sequential()
    # model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
    #                  activation='relu',
    #                  input_shape=input_shape))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Conv2D(64, (5, 5), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Flatten())
    # model.add(Dense(1000, activation='relu'))
    # model.add(Dense(num_classes, activation='softmax'))
    #
    # model.compile(loss=keras.losses.categorical_crossentropy,
    #               optimizer=keras.optimizers.Adam(),
    #               metrics=['accuracy'])
    #
    #
    # class AccuracyHistory(keras.callbacks.Callback):
    #     def on_train_begin(self, logs={}):
    #         self.acc = []
    #
    #     def on_epoch_end(self, batch, logs={}):
    #         self.acc.append(logs.get('acc'))
    #
    #
    # history = AccuracyHistory()
    #
    # model.fit(x_train, y_train,
    #           batch_size=batch_size,
    #           epochs=epochs,
    #           verbose=1,
    #           validation_data=(x_test, y_test),
    #           callbacks=[history])
    # score = model.evaluate(x_test, y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
    # plt.plot(range(1, 11), history.acc)
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.show()
    #
    # model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
    # del model  # deletes the existing model

    # predict
    # x_img = open('ximg.pt', 'rb')
    # list_images = pickle.load(x_img)
    # x_test = np.asarray(list_images)
    # x_test = x_test.reshape(x_test.shape[0], SIZE_FACE, SIZE_FACE, 1)
    # x_test = x_test.astype('float32')
    # x_test /= 255
    #
    # l_img = open('limg.pt', 'rb')
    # list_labels = pickle.load(l_img)
    # list_labels = np.asarray(list_labels)
    # l_test = list_labels
    #
    # y_img = open('yimg.pt', 'rb')
    # list_labels_y = pickle.load(y_img)
    # list_labels_y = np.asarray(list_labels_y)
    # y_test = list_labels_y
    # model = load_model('my_model.h5')  # load model
    #
    # id_test = 3000
    # xx_test = x_test[id_test:id_test+1]
    # #yy_test = y_test[1000:img1001]
    # ll_test = l_test[id_test]
    # img_test = list_images[id_test]
    #
    # #score = model.evaluate(x_test, y_test, verbose=0)
    # #print("score = " + str(score))
    # pred = model.predict(xx_test)
    # print("pred label = " + str(EMOTIONS[pred.argmax()]))
    # print("pred max = " + str(pred.max()))
    # print("true label = " + ll_test)
    # cv2.imshow('image', img_test)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
