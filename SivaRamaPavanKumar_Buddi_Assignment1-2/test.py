import cv2
import os
import numpy as np
from PIL import Image
from brainExtraction import *
#creating respective thresh folders in slices and boundaries
def folder(s):
    newpath = r'Slices/'+s
    newpath1 = r'Boundaries/'+s 
    #if there is no folder create one
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    if not os.path.exists(newpath1):
        os.makedirs(newpath1) 
#creating two folder slices and boundaries
path1 = r'Slices' 
if not os.path.exists(path1):
    os.makedirs(path1)
path2 = r'Boundaries' 
if not os.path.exists(path2):
    os.makedirs(path2)
#getting the images that end with thresh from data
for file in os.listdir("testPatient"):
    #getting the basename
    base=os.path.basename(file)
    #split the text
    split_base=os.path.splitext(base)[0]
    #if the image name ends with thresh
    if split_base.endswith("thresh"):
        #create folders for each thresh image in both slices and boundaries
        folder(split_base)
        #then perform brain extraction
        main_function(split_base)