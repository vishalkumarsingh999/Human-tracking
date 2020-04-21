# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 10:22:05 2020

@author: VISHAL KUMAR SINGH
"""
import cv2
import numpy as np
import pickle
import winsound
import threading
import time

labels={}
font = cv2.FONT_HERSHEY_SIMPLEX
total_stu=[]

def makesound():
    winsound.PlaySound("meow.wav", winsound.SND_ASYNC)
    time.sleep(1)

cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
recg=cv2.face.LBPHFaceRecognizer_create()
recg.read("trained.yml")
with open("label.pickle",'rb') as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}

for k,v in labels.items():
   total_stu.append(v)

while True:
    try:
        if cv2.waitKey(1) == 27:
            break
        ret, frame = cap.read()
        if (not ret):
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
        # frame = cv2.GaussianBlur(frame,(11,11),0)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(img, 1.3, 4)
        count = 0
        for (x, y, w, h) in face:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            gray_face = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
            id_, conf = recg.predict(gray_face)
            print(conf)
            if conf >= 45 and conf <= 90:
                print(labels[id_])
                cv2.putText(frame, labels[id_], (x + w, y), font, 1, (100, 0, 255), 2, cv2.LINE_AA)
                count += 1
                cap.release()
                t1 = threading.Thread(makesound())
                t1.start()
            else:
                cap.release()
                t1 = threading.Thread(makesound())
                t1.start()
                cv2.putText(frame, "UNknown", (x + w, y), font, 1, (100, 0, 255), 2, cv2.LINE_AA)

        """cv2.putText(frame, "Total student =" + str(len(total_stu)), (30, 30), font, 1, (100, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Present student =" + str(count), (30, 60), font, 1, (100, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Exit student =" + str(len(total_stu)-count), (30, 90), font, 1, (100, 0, 255), 2, cv2.LINE_AA)"""
        pro = cv2.resize(frame, (1080, 800))
        cv2.imshow("live face", pro)
    except:
        print("error aarha hai")

cap.release()
cv2.destroyAllWindows()