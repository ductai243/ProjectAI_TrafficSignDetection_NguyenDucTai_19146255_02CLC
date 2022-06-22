
from json import load
from tracemalloc import stop
from PyQt6 import QtCore, QtGui, QtWidgets


# from queue import Empty
from sys import flags
import numpy as np
import cv2 as cv
import numpy as np
import cv2 as cv
import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model


# traffic_cascade = cv.cvtColor(frame, )
traffic_cascade = cv.CascadeClassifier()
traffic_cascade.load('trafficSign_cascade.xml')

model = load_model('Final_official.h5')

def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img
def getCalssName(classNo):
    if   classNo == 0: return 'No Horn'
    elif classNo == 1: return 'No Entry'
    elif classNo == 2: return 'No Parking Stopping'
    elif classNo == 3: return 'No Cars'
    elif classNo == 4: return 'No Trucks'
    elif classNo == 5: return 'Turn Right'
    elif classNo == 6: return 'Turn Left'
    elif classNo == 7: return 'Go Ahead'
    elif classNo == 8: return 'No Traffic Ways'
    elif classNo == 9: return 'Roundabout'
    elif classNo == 10: return 'Stop'
    elif classNo == 11: return 'Lim 50km/h'
    elif classNo == 12: return 'Keep Right'
    elif classNo == 13: return 'Keep Left'
    elif classNo == 14: return 'Lim 30km/h'
    elif classNo == 15: return 'Lim 60km/h'
    elif classNo == 16: return 'Lim 70km/h'
    elif classNo == 17: return 'Lim 80km/h'
    elif classNo == 18: return 'Lim 20km/h'
    elif classNo == 19: return 'Keep Ahead & Left'
    elif classNo == 20: return 'Keep Ahead & Right'
   


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(799, 625)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(110, 190, 601, 361))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("../Downloads/640x360.png"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(210, 570, 113, 32))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(480, 570, 113, 32))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(110, 20, 581, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(90, 60, 581, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(10, 130, 751, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_4.setObjectName("label_4")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
    
    def ImageUpdateSlot(self, Image):
        self.label.setPixmap(QtGui.QPixmap.fromImage(Image))

    def CancelFeed(self):
        self.Worker1.stop()

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pushButton.setText(_translate("Dialog", "Select Image"))
        self.pushButton_2.setText(_translate("Dialog", "Use Camera"))
        self.label_2.setText(_translate("Dialog", "Ho Chi Minh City University of Technology and Education"))
        self.label_3.setText(_translate("Dialog", "Artificial Intelligence"))
        self.label_4.setText(_translate("Dialog", "TRAFFIC SIGN DETECTION AND CLASSIFICATION IN VIET NAM \n"
" USING CNN ALGORITHM"))
        self.label_4.setStyleSheet("font-weight: bold")
        self.label_3.setStyleSheet("font-weight: bold")
        self.label_2.setStyleSheet("font-weight: bold")
        #click button to open image
        self.pushButton.clicked.connect(self.select_image)
        #click button to open camera
        self.pushButton_2.clicked.connect(self.use_camera)

    def select_image(self, Dialog):
        #open image dialog and get image
        self.filename = QtWidgets.QFileDialog.getOpenFileName(None, 'Open File', '/')
        self.label.setPixmap(QtGui.QPixmap(self.filename[0]))
        frame = cv.imread(self.filename[0])
        self.label.setScaledContents(True)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # scaleFactor =  1.05
        # minNeighbors = 1
        # minSize = (1,1)
        # maxSize = (120, 120)

        traffic = traffic_cascade.detectMultiScale(frame)


       
        for (x,y,w,h) in traffic:
            # cv.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),1)
            roi = frame[y:y+h,x:x+w]
            roi = np.asarray(roi)
            
            if(w*h>1255):
                # cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                roi = cv.resize(roi,(32,32))
                roi = preprocessing(roi)
                roi = roi.reshape(1,32,32,1)

                predict = ['0','1','2','3','4','5','6','7','8','9','10','11','12']
                predict = np.array(predict)
                result = np.argmax(model.predict(roi),axis=-1)
                # text = predict[result]
                result = int(result)
                
                text = getCalssName(result)
                # text = predict[result]
                text = str(text)
               

                # print(predictions)
                if result==0 or result==1 or result==2 or result==3 or result==4 or result==8 or result==10 or result==11 or result==14 or result==15 or result==16 or result==17 or result==18:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
                    cv2.rectangle(frame, (x,y-17),(x+w, y), (0,255,0),-2)
                    cv.putText(frame,text,(x,y),cv.FONT_HERSHEY_COMPLEX,0.5,(38,4,209),1,cv.LINE_AA)

                else: 
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,255),1)
                    cv2.rectangle(frame, (x,y-17),(x+w, y), (255,0,255),-2)
                    cv.putText(frame,text,(x,y),cv.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),1,cv.LINE_AA)

              

        # save to image local
        cv.imwrite("result.jpg",frame)
        # load image to label
        self.label.setPixmap(QtGui.QPixmap("result.jpg"))


    is_running = False

    def use_camera(self, Dialog):
        if self.is_running:
            #dang chay
            self.Worker1.stop()
            self.pushButton_2.setText("Use Camera")
            self.is_running = False
        else:
            # dang dung
            self.Worker1 = Worker1()
            self.Worker1.start()
            self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
            self.pushButton_2.setText("Stop")
            self.is_running = True
            
        
class Worker1(QtCore.QThread):
    ImageUpdate = QtCore.pyqtSignal(QtGui.QImage)
    def run(self):
        self.ThreadActive = True
        Capture = cv2.VideoCapture(0)
        while self.ThreadActive:
            ret, frame = Capture.read()
            # frame = cv.resize(frame,[960,540])

            if not ret:
                print(' can not read video frame. Video ended?')
                break

            # your code
            scaleFactor =  1.05
            minNeighbors = 6
            minSize = (10,10)
            maxSize = (120, 120)

            traffic = traffic_cascade.detectMultiScale(frame)
           
        

            for (x,y,w,h) in traffic:
                # cv.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),1)
                roi = frame[y-int(h/10):y+h+int(h/15),x-int(w/15):x+w+int(h/15)]
                roi = np.asarray(roi)
                # cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)
                
                
                try:
                    roi = cv.resize(roi,(32,32))
                    roi = preprocessing(roi)
                    roi = roi.reshape(1,32,32,1)
                
                    predict = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14']
                    predict = np.array(predict)
                    result = np.argmax(model.predict(roi),axis=-1)
                    # text = predict[result]
                    result = int(result)
                    
                    text = getCalssName(result)
                    # text = predict[result]
                    text = str(text)
                   
                    # if result==0 or result==1 or result==2 or result==3 or result==4 or result==8 or result==10 or result==11 or result==14 or result==15 or result==16 or result==17 or result==18:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
                    cv2.rectangle(frame, (x,y-10),(x+w, y), (0,255,0),-2)
                    cv.putText(frame,text,(x,y),cv.FONT_HERSHEY_COMPLEX,0.35,(38,4,209),1,cv.LINE_AA)
                    # else: 
                    #     cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,255),1)
                    #     cv2.rectangle(frame, (x,y-10),(x+w, y), (255,0,255),-2)
                    #     cv.putText(frame,text,(x,y),cv.FONT_HERSHEY_COMPLEX,0.35,(0,0,0),1,cv.LINE_AA)
                    
                except Exception as e:
                    print(str(e))
            if ret:
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ConvertToQtFormat = QtGui.QImage(Image.data, Image.shape[1], Image.shape[0], QtGui.QImage.Format.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)

        # listen to press F5 to stop thread
        if self.ThreadActive:
            self.ThreadActive = False
            Capture.release()
            cv2.destroyAllWindows()

        
    def stop(self):
        self.ThreadActive = False
        self.quit()
        

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec())
