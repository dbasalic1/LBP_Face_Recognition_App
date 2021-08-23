import cv2 as cv
import os
from preprocess import preprocessImage
possibleResults = ['Osoba A', 'Osoba B', 'Osoba C', 'Osoba D']
testpath = os.path.join(os.path.dirname(__file__), 'test')
recognizer = cv.face_LBPHFaceRecognizer.create()
recognizer.read('model.yaml')
images = os.listdir(testpath)
for imgName in images:
    imgPath = os.path.join(testpath, imgName)
    print(imgPath)
    img = cv.imread(imgPath)
    face, (x, y, w, h) = preprocessImage(img)
    result = recognizer.predict(face)
    if result[0] == -1:
        osoba = "Nepoznato lice"
    else:
        osoba = possibleResults[result[0]]
    print("Predikcija: " + osoba + " Sigurnost: " + str(result[1]))
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.putText(img, osoba, (x-2, y-2), cv.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 2)
    cv.imshow("Prepoznavanje lica", img)
    cv.waitKey(0)