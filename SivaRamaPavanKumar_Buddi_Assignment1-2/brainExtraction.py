import cv2
import os
import numpy as np
from PIL import Image
#main_function takes every image that ends with thresh
def main_function(image_path):
    #bound function is draw boundaries for slices
    def bound(image,j,image_path):
        #reading the slice
        image=cv2.imread(image)
        #converting into rgb and gray image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        #finding the contours for the image
        contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        #drawing the contours
        image=cv2.drawContours(image, contours, -1, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        #appending the resulting slices to boundary folder
        cv2.imwrite('Boundaries/'+image_path+'/boundary'+str(j)+'.png',image)
    #Initializing an empty list
    x=[]
    j=0
    #reading image from data with ends with thresh
    img_rgb = cv2.imread('testPatient/'+image_path+'.png')
    #changing image to gray
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    #reading the templare R image for template matching
    template = cv2.imread('R.png',0)
    #getting the shapes of template
    w, h = template.shape[::-1]
    #Match template matches the template image and actual gray scale image and returns the coordinates
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCORR_NORMED)
    #setting the threshold
    threshold = 0.8
    #pushing the coordinates to a variable loc as numpy array
    loc = np.where( res >= threshold)
    #converting into list
    l=list(loc)
    #zipping all the x and y coordinates from list l
    for pt in zip(*l[::-1]):
        #now here we are drawing rectangle for one R with the image, start_point end_point color and thickness.
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 1)
    #again we are taking all the coordinates and appending them to list
    for pt in zip(*loc[::-1]):
        x.append(pt)
    #to find the start point for every row i.e (0,126), (0,244).......initialize a set of all y coordinates so there won't be any repeated values
    a=set(loc[0])
    #sorting the set
    b=sorted(a)
    #iterating the set and adding the starting points to x 
    for i in range(len(a)):
        #inserting the starting points
        x.insert(i*8,(0,b[i]))
    #Now here we are taking the image
    ima=Image.open('testPatient/'+image_path+'.png')
    for i in range(9,len(x)):
        start=x[i-9]
        #if the x coordinate is pt[0] i am cropping last part by adding 118 to x coordinate,next y coordinate
        if(start[0]==pt[0]):
            k=list(end)
            k[0]=pt[0]+118            
            g=tuple(k)
            end=g
        #else i am just taking end 
        else:
            end=x[i]
        #cropping the image and getting each slice
        c_img=ima.crop((start[0]+28,start[1]+w,end[0],end[1]+10))
        #checking if there is blank image without brain slice
        if c_img.getbbox()!=None:
            #save the slices to slices folder in the respective thresh folder
            c_img.save('Slices/'+image_path+'/slice'+str(j)+'.png')
            #passing the image to get the boundary of the brain
            bound('Slices/'+image_path+'/slice'+str(j)+'.png',j,image_path)
        j=j+1   