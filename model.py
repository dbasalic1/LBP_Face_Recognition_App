import cv2 as cv
import numpy as np
import os
from preprocess import preprocessImage

#oznake mogucih rezultata
possibleResults = ['Osoba A', 'Osoba B', 'Osoba C', 'Osoba D']
#sva lica koristena za treniranje
faces = []
#labele dodijeljene licima
labels = []

path = os.path.join(os.path.dirname(__file__), 'train')
print("Treniranje modela, molimo sacekajte......")

for i in range(len(possibleResults)):
    currentDir = os.path.join(path, possibleResults[i])
    images = os.listdir(currentDir)
    for imgName in images:
        imgPath = os.path.join(currentDir, imgName)
        img = cv.imread(imgPath)
        face, region = preprocessImage(img)
        if face is not None and region is not None:
            faces.append(face)
            labels.append(i)
print("Model uspjesno istreniran")
print("Broj lica: " +str(len(faces)) + " Broj labela: " + str(len(labels)))
model = cv.face_LBPHFaceRecognizer.create()
model.setThreshold(30.0)
model.train(faces, np.array(labels))
model.write('model.yaml')
















