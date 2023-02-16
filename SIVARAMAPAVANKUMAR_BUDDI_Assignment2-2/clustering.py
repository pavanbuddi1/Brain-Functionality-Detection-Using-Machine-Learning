from tkinter import filedialog
import cv2
import numpy as np
import os
import shutil
import csv
import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter
#method for clusters
def cluster(path):
    #getting path
    dir_list = os.listdir(path)
    #checking folder
    if not os.path.isdir("./Clusters"):
        os.mkdir("./Clusters")
    for tem in dir_list:
        count=[]
        onlyfiles = os.listdir(path+tem+'/')
        for tem1 in onlyfiles:
            img = cv2.imread(path+tem+'/'+tem1)
            col = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(col)
            thresh_img = cv2.threshold(s, 92, 255, cv2.THRESH_BINARY)[1]

            if not os.path.isdir("./Clusters/" + tem):
                os.mkdir("./Clusters/" + tem)
            #saving cluster images
            cv2.imwrite("./Clusters/" + tem + '/'+ tem1.split('.')[0] + '.png', thresh_img)
            Y, X = np.where(thresh_img==255)
            pack = np.column_stack((X,Y))
            if len(pack)>0:
                #using dbscan in sckit library
                clustering = DBSCAN(eps=5, min_samples=5).fit(pack)
                labels=clustering.labels_
                ls, cs = np.unique(labels,return_counts=True)
                dic = dict(zip(ls,cs))
                idx = [i for i,label in enumerate(labels) if dic[label] >135 and label >= 0]
                cnt=0
                counts = np.bincount(labels[labels>=0])
                for i in counts:
                        if i>135:
                                cnt+=1
                count.append([int(tem1.split('.')[0]),cnt])
            else:
                count.append([int(tem1.split('.')[0]),0])
        count = sorted(count, key=lambda x:x[0])
        #writing to csv
        csv_header = ['SliceNo','ClusterCount']
        with open("./Clusters/"+tem+'/'+tem+'.csv', 'w',  newline='') as file:
            writer = csv.writer(file)
            writer.writerow(csv_header)
            writer.writerows(count)
#method for slices
def slices(image_path, file_name):
    if not os.path.isdir("./Slices"):
        os.mkdir("./Slices")
    if not os.path.isdir("./Slices/" + file_name.split('.')[0]):
        os.mkdir("./Slices/" + file_name.split('.')[0])
    image = cv2.imread(image_path + '/' + file_name, cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.inRange(gray,255,255)
    edge = cv2.Canny(thresh, 255,255)
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, -1, (0,255,0), 1)
    flg1=0
    flg2=0
    h=0
    w=0
    x1, y1, w1, h1 = cv2.boundingRect(contours[len(contours)-1])
    for i in reversed(range(len(contours))):
        x2, y2, w2, h2 = cv2.boundingRect(contours[i])
        if(y2 == y1 and x2!=x1) and flg1==0:
            flg1=1
            w = abs(x2-x1)
        if(x2 == x1 and y2!=y1) and flg2==0:
            flg2=1
            h = abs(y2-y1)
    flg=0
    temp=1
    img_height, img_width = image.shape[:2]
    x, y, w1, h1 = cv2.boundingRect(contours[-1])
    for i in reversed(range(len(contours))):
        flg=0
        x2, y2, w2, h2 = cv2.boundingRect(contours[i])
        img_x = x2 + w
        img_y = y2 - h
        if img_x < x or img_y < y:
            continue
        image1 = image[img_y:y2, x2+w1:img_x]
        gray_version_1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        if cv2.countNonZero(gray_version_1) == 0:
            flg=1
        if flg==0:
            #saving to slices folder
            cv2.imwrite("./Slices/" + file_name.split('.')[0] + '/'+ str(temp) + '.png', image1)
            temp+=1