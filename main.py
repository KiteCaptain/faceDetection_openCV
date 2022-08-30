import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

# ### face detectiion is accomplished using a classifier
# ### a classifier is an algorithm that decides whether a given image is negative or positive
# ##
# image = cv.imread("C:\\Users\\User\\Pictures\\Saved Pictures\\d johnson\\download%20(2).jpg")
# #cv.imshow('person', image)
# gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
# cv.imshow('gray person', gray)
#
# haar_cascade = cv.CascadeClassifier('haar_face')
# faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=1)
# print(f"Number of faces found = {len(faces_rect)}")
# cv.waitKey(0)

# ## haar_cascade can also be used to detect faces in a video frame by frame



# FACE TRAIN
people = []
for i in os.listdir(r'C:\Users\User\Pictures\trainer'):
    people.append(i)
#print(people)
DIR = r'C:\Users\User\Pictures\trainer'
haar_cascade = cv.CascadeClassifier('haar_face')
features = []
labels = []
def create_train ():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)
create_train()
print('training done____________')

features = np.array(features, dtype='object')
labels = np.array(labels)


face_recognizer = cv.face.LBPHFaceRecognizer_create()

## --> training the recognizer on the features list and labels list
face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')

np.save('features.npy', features)
np.save('labels.npy', labels)


#FACE RECOGNITION

haar_cascade = cv.CascadeClassifier('haar_face')

people = ['d johnson', 'jim carrey', 'ryan reynolds', 'w smith']
# features = np.load('features.npy')
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')
img = cv.imread(r'C:\Users\User\Pictures\trainer\d johnson\images%20(9).jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('gray', gray)

##  detecting face on image
faces_rect = haar_cascade.detectMultiScale(gray,1.1,4)
for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'label = {people[label]} with a confidence of {confidence}')

    cv.putText(img, str(people[label]),(20,20), cv.FONT_HERSHEY_COMPLEX,1.0,(255),2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv.imshow('Detected face', img)


cv.waitKey(0)
















