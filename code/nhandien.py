import cv2
import numpy as np

import os
from keras.models import load_model
from keras.preprocessing import image

def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

faceDetect = cv2.CascadeClassifier('E:/2020/phantichthongke_BTH/HT/NCKH_CNN/FaceCNN/haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (203, 23, 252)

while(True):
    # camera read
    ret, img = cam.read()
    img_copy=img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    index = 0
    for (x, y, w, h) in faces:

        cropped = img_copy[y:y + h, x:x + w]
        cv2.imwrite('D:/' + str(index) + '.jpg', cropped)
        # img = Image.fromarray(cropped)
        # img = img.resize((150,150))
        img = image.load_img('D:/' + str(index) + '.jpg', target_size=(150, 150))
        x1 = image.img_to_array(img)
        x1 = np.expand_dims(x1, axis=0)
        os.remove('D:/' + str(index) + '.jpg')
        model = load_model('E:/2020/phantichthongke_BTH/HT/NCKH/NCKH_BAINOP/KQ_train/Luu_train.h5')
        model.compile(loss='categorical_crossentropy',
                          optimizer='rmsprop',
                          metrics=['accuracy'])
        # predicting images
        images = np.vstack([x1])
        classes = model.predict_classes(images, batch_size=16)  # label cua y
        print(classes)
        if classes[0] == 0:
            prediction = 'duy'
        elif classes[0] == 1:
            prediction = 'luu'
        elif classes[0] >= 2:
            prediction = 'Khong xac dinh'

        cv2.rectangle(img_copy, (x, y),
                     (x + w, y + h), (255, 0, 0), 2)  # Ve hinh chu nhat
        #cv2.putText(img, "Name: " + prediction, (x, y + h + 30), fontface, fontscale, fontcolor, 2)
        cv2.putText(img_copy, prediction, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    cv2.imshow('khuon mat', img_copy)
    if cv2.waitKey(1) == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()


'''
 if classes[0] == 0:
            prediction = 'Thanh An'
        elif classes[0] == 1:
            prediction = 'bao'
        elif classes[0] == 2:
            prediction = 'Chuyen'
        elif classes[0] == 3:
            prediction = 'Dinh'
        elif classes[0] == 4:
            prediction = 'duy'
        elif classes[0] == 5:
            prediction = 'Hau'
        elif classes[0] == 6:
            prediction = 'khang'
        elif classes[0] == 7:
            prediction = 'lien'
        elif classes[0] == 8:
            prediction = 'long'
        elif classes[0] == 9:
            prediction = 'luu'
        elif classes[0] == 10:
            prediction = 'nam'
        elif classes[0] == 11:
            prediction = 'oanh'
        elif classes[0] == 12:
            prediction = 'phan'
        elif classes[0] == 13:
            prediction = 'phuc'
        elif classes[0] == 14:
            prediction = 'quynh'
        elif classes[0] == 15:
            prediction = 'sinh'
        elif classes[0] == 16:
            prediction = 'thinh'
        elif classes[0] == 17:
            prediction = 'toan'
        elif classes[0] == 18:
            prediction = 'toi'
        elif classes[0] == 19:
            prediction = 'tritrung'
        elif classes[0] == 20:
            prediction = 'truc'
        elif classes[0] == 21:
            prediction = 'tung'
        else:
            prediction = 'Khong xac dinh'
'''