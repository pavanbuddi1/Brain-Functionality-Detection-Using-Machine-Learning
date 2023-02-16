import os
import shutil
import numpy as np
import cv2 as cv
import glob
import tensorflow as tf
from PIL import Image
import pandas as pd
from tensorflow import keras
from keras.layers import Dense, Input, InputLayer, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
from keras.utils import load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score



def fun1(path, imag):
    for folders in os.listdir('./testPatient'): 
        if os.path.isdir(os.path.join('./testPatient', folders)):  
            path += 1

    path = path - 3

    for i in range(1,path+1):
        fp = './testPatient/Patient_{}/*thresh.png'.format(i)
        x=0
        for image in glob.glob(fp):
            x += 1
        imag.append(x)

    return path, imag


def modelLoad(x_train, x_test, validation_x, y_train, test_y, y_validation):
    m = load_model('cnnModel.h5')
    m.evaluate(x=tf.cast(np.array(x_test), tf.float64), 
                    y=tf.cast(list(map(int, test_y)), tf.int32), batch_size=32)

    predY=m.predict(x_test)
    predY=np.argmax(predY,axis=1)

    print(classification_report(predY, test_y))
    print(accuracy_score(predY, test_y))
    print(confusion_matrix(predY, test_y))

    predX=m.predict(x_train)
    predX=np.argmax(predX,axis=1)

    validP = m.predict(validation_x)
    validP = np.argmax(validP, axis=1)

    print(accuracy_score(predX, y_train))
    
    return predX, predY, validP


def create_results(pat_folders, tImages, predX, predY, validP, testValue, validationValue):
    n = len(tImages) 
    metrics = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity']
    preds = predX
    i = 1
    for j in range(0, n-2):
        t = tImages[j]
        Label = []
        Label.append(preds[0:t])
        preds = preds[t:]
        IC_Number = list(range(1,t+1))

        df = pd.DataFrame()
        rawData = {'IC_Number': IC_Number, 'Label':Label[0]}
        df = pd.DataFrame(rawData, columns = ['IC_Number', 'Label'])
        df.to_csv('./testPatient/Patient_{}/Results.csv'.format(i), index=False)

        pLabels = pd.read_csv('./testPatient/Patient_{}/Results.csv'.format(i))
        pLabels = pLabels['Label'].tolist()

        tLabels = pd.read_csv('./testPatient/newPatient_{}_Labels.csv'.format(i))
        tLabels = tLabels['Label'].tolist()
        
        mv = []

        lm = confusion_matrix(tLabels, pLabels)
        print(lm)

        accuracy = accuracy_score(tLabels, pLabels)
        precision = precision_score(tLabels, pLabels, average=None)
        spec = lm[0,0] / (lm[0,0] + lm[0,1])
        sensitivity = lm[1,1] / (lm[1,0] + lm[1,1])

        mv.append(accuracy)
        mv.append(precision[0])
        mv.append(spec)
        mv.append(sensitivity)

        df = pd.DataFrame()
        rawData = {'Metric': metrics, 'Score': mv}
        df = pd.DataFrame(rawData, columns = ['Metric', 'Score'])
        df.to_csv('./testPatient/Patient_{}/Metrics.csv'.format(i), index=False)

        i += 1
    
    t = tImages[n-2]
    Label = []
    Label.append(predY)
    IC_Number = list(range(1,t+1))

    df = pd.DataFrame()
    rawData = {'IC_Number': IC_Number, 'Label':Label[0]}
    df = pd.DataFrame(rawData, columns = ['IC_Number', 'Label'])
    df.to_csv('./testPatient/Patient_{}/Results.csv'.format(testValue), index=False)

    tLabels = pd.read_csv('./testPatient/newPatient_{}_Labels.csv'.format(testValue))
    tLabels = tLabels['Label'].tolist()
    pLabels = pd.read_csv('./testPatient/Patient_{}/Results.csv'.format(testValue))
    pLabels = pLabels['Label'].tolist()

    mv = []

    lm = confusion_matrix(tLabels, pLabels)

    accuracy = accuracy_score(tLabels, pLabels)
    precision = precision_score(tLabels, pLabels, average=None)
    spec = lm[0,0] / (lm[0,0] + lm[0,1])
    sensitivity = lm[1,1] / (lm[1,0] + lm[1,1])

    mv.append(accuracy)
    mv.append(precision[0])
    mv.append(spec)
    mv.append(sensitivity)

    df = pd.DataFrame()
    rawData = {'Metric': metrics, 'Score': mv}
    df = pd.DataFrame(rawData, columns = ['Metric', 'Score'])
    df.to_csv('./testPatient/Patient_{}/Metrics.csv'.format(testValue), index=False)

    t = tImages[n-1]
    Label = []
    Label.append(validP)
    IC_Number = list(range(1,t+1))

    df = pd.DataFrame()
    rawData = {'IC_Number': IC_Number, 'Label':Label[0]}
    df = pd.DataFrame(rawData, columns = ['IC_Number', 'Label'])
    df.to_csv('./testPatient/Patient_{}/Results.csv'.format(validationValue), index=False)

    pLabels = pd.read_csv('./testPatient/Patient_{}/Results.csv'.format(validationValue))
    pLabels = pLabels['Label'].tolist()

    tLabels = pd.read_csv('./testPatient/newPatient_{}_Labels.csv'.format(validationValue))
    tLabels = tLabels['Label'].tolist()

    mv = []

    lm = confusion_matrix(tLabels, pLabels)
    
    precision = precision_score(tLabels, pLabels, average=None)
    accuracy = accuracy_score(tLabels, pLabels)
    spec = lm[0,0] / (lm[0,0] + lm[0,1])
    sensitivity = lm[1,1] / (lm[1,0] + lm[1,1])

    mv.append(accuracy)
    mv.append(precision[0])
    mv.append(spec)
    mv.append(sensitivity)

    df = pd.DataFrame()
    rawData = {'Metric': metrics, 'Score': mv}
    df = pd.DataFrame(rawData, columns = ['Metric', 'Score'])
    df.to_csv('./testPatient/Patient_{}/Metrics.csv'.format(validationValue), index=False)
