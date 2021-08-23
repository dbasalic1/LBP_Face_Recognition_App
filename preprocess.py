import cv2 as cv
def preprocessImage(img):
    grayscale = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    facedetector = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
    facesdetected = facedetector.detectMultiScale(grayscale, scaleFactor=1.05, minNeighbors=6)
    if len(facesdetected) > 0:
        (x, y, w, h) = facesdetected[0]
        face = grayscale[y:y+w, x:x+h]
        return face, facesdetected[0]
    else:
        return None, None