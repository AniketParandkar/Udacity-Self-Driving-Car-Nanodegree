import sys
import time
import csv
import cv2
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from random import randint 

rows = []



with open('driving_log.csv') as csvfile:
    cell = csv.reader(csvfile)
    for line in cell:
        #print(line)
        rows.append(line)

#print(rows)

centreImage = []
rightImage = []
leftImage = []
imageList = []
steeringList = []

def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
#        file.write("%s%s%s \r" % (prefix, "#"*x))
        file.write("%s%s%s | %i/%i\r" % (prefix, "#"*x, " "*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.write("Done")
    file.write("\n")
    file.flush()

    
#def progressbar(it, prefix="", size=60, file=sys.stdout):
#    count = len(it)
#    def show(j):
#        x = int(size*j/count)
#        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
#        file.flush()        
#    show(0)
#    for i, item in enumerate(it):
#        yield item
#        show(i+1)
#    file.write("\n")
#    file.flush()
        
def getImages(cellRows, imageList, steeringList, imagePosition):
    correction = 0.15
    for i in progressbar(range(15), "Processing | ", 40):
        for line in cellRows:
            path = line
            #print(path).
            image = cv2.imread(path[imagePosition])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            imageList.append(image)
            if imagePosition == 1:    #Left side Image
                side = 0
            elif imagePosition == 2:  #Right Side Image
                side = 2
            else:
                side = 1              #Centre Image
            steering = float(line[3])
            steeringList.append(steering + (1-side)*correction)
            
    return imageList, steeringList


#Getting Centre image
imagePosition = 0
print('Getting Centre Camera Images...')
imageList, steeringList = getImages(rows, imageList, steeringList, imagePosition)
len1 = len(imageList)
#Getting Left Image
imagePosition = 1
print('Getting Left Camera Images...')
imageList, steeringList = getImages(rows, imageList, steeringList, imagePosition)
len2 = len(imageList)
#Getting Right Image
imagePosition = 2
print('Getting Right Camera Images...')
imageList, steeringList = getImages(rows, imageList, steeringList, imagePosition)
len3 = len(imageList)


#Image Test
plt.imshow(imageList[randint(0,len1)])
plt.show()

plt.imshow(imageList[randint(len1,len2)])
plt.show()

plt.imshow(imageList[randint(len2, len3)])
plt.show()


#For the model, the inputs are Images and output is SteeringAngle
dataBase = list(zip(imageList, steeringList))

##Creating Dataset for training
def getData(Data, batchSize):
    for i in progressbar(range(15), "Processing | ", 40):
        sampleSize = len(Data)
        #print(batchSize)
        dataBase = shuffle(Data)
        #print("##############")
        #print(sampleSize)

        for offset in range(0, sampleSize, batchSize):
            #print('offset loop')
            dataBatch = dataBase[offset:offset + batchSize]
            for img, steeringAngle in dataBatch:
                #print('dataset loop')
                DatasetX = np.array(img)
                DatasetY = np.array(steeringAngle)

    return DatasetX, DatasetY



#Getting Dataset
trainingData, validationData = train_test_split(dataBase, test_size = 0.2)


print('Training Dataset: {}'.format(len(trainingData)))
print('Validation Dataset: {}'.format(len(validationData)))

print('Generating Training Dataset...')
XTraining,YTraining = getData(trainingData, 64)
print('Generating Validation Dataset...')
XValidation,YValidation = getData(validationData, 64)


