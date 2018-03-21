import numpy as np
import cv2



images_many = list()
for i in range(0, 20):
    images_many.append(cv2.imread("many/many_" + str(i) + '.jpg'))

images_other = list()
for i in range(0, 20):
    images_other.append(cv2.imread("other/other_" + str(i) + '.jpg'))


def profile_images():
    images_profile = list()
    for iter in range(0, 20):
        images_profile.append(cv2.imread("profile/profile_" + str(iter) + '.jpg'))

    face_cascade = cv2.CascadeClassifier('classifiers/haarcascade_profileface.xml')
    eye_cascade = cv2.CascadeClassifier('classifiers/haarcascade_eye.xml')

    for j in range(0, 20):
        gray = cv2.cvtColor(images_profile[j], cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            img = images_profile[j]
            img = cv2.rectangle(img, (x, y),(x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex,ey),(ex+ew,ey+eh), (0, 255, 0), 2)

        cv2.imshow('img',images_profile[j])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def frontal_images():
    images_anfas = list()
    for i in range(0, 20):
        images_anfas.append(cv2.imread("frontal/frontal_" + str(i) + '.jpg'))
    face_cascade = cv2.CascadeClassifier('classifiers/haarcascade_smile.xml')
    # eye_cascade = cv2.CascadeClassifier('classifiers/haarcascade_eye.xml')

    for j in range(0, 20):
        gray = cv2.cvtColor(images_anfas[j], cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            img = images_anfas[j]
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # roi_gray = gray[y:y + h, x:x + w]
            # roi_color = img[y:y + h, x:x + w]
            # eyes = eye_cascade.detectMultiScale(roi_gray)
            # for (ex, ey, ew, eh) in eyes:
            #     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        cv2.imshow('img', images_anfas[j])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

frontal_images()
