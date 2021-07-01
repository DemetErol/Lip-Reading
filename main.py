from PyQt5.uic import loadUi
from PyQt5.QtWidgets import*
import PyQt5.QtGui as QtGui
import numpy as np
from imutils import face_utils
import imutils
import dlib
import cv2
import time
import shutil
import os
import imageio
from tensorflow.keras.models import load_model


class predict:

    def predictData(self,X_test):
        #Predict unseen data with keras model
        self.model = load_model("keras_model.h5")
        #Unseen data preprocessing
        X_test = np.array(X_test)
        X_test = self.scaler(X_test)
        X_test = np.expand_dims(X_test, axis=4)
        y_pred= self.model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        #'Begin', 'Choose', 'Connection'
        if(y_pred[0]==0):
            return "BEGIN"
        elif(y_pred[0]==1):
            return "CHOOSE"
        else:
            return "CONNECTION"

    def scaler(self,X):
        v_min = X.min(axis=(2, 3), keepdims=True)
        v_max = X.max(axis=(2, 3), keepdims=True)
        X = (X - v_min) / (v_max - v_min)
        X = np.nan_to_num(X)
        return X



class loadUi_example(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("raichu.ui",self)
        self.setWindowTitle("Raichu")
        self.setWindowIcon(QtGui.QIcon("raichu.png"))
        self.isSpeak=False
        self.control=True
        self.disply_width = 640
        self.display_height = 480
        self.start.clicked.connect(self.start_video)
        self.stop.clicked.connect(self.stop_video)
        self.shapePredictorPath = 'shape_predictor_68_face_landmarks.dat'
        self.faceDetector = dlib.get_frontal_face_detector()
        self.facialLandmarkPredictor = dlib.shape_predictor(self.shapePredictorPath)

    def start_video(self):
        self.isSpeak=True
        self.control=True
        self.text_message.setText("")
        self.label.setText("Please say begin,choose or connection")
        for filename in os.listdir("words"):
            filepath = os.path.join("words", filename)
            try:
                shutil.rmtree(filepath)
            except OSError:
                os.remove(filepath)

        self.vs = cv2.VideoCapture(0)
        time.sleep(1.0)
        frame_count = 0
        while self.control:
            _,frame = self.vs.read()
            frame = imutils.resize(frame, width=400)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.namedWindow("WEBCAM")
            faces = self.faceDetector(gray, 0)
            for (i, face) in enumerate(faces):
                facialLandmarks = self.facialLandmarkPredictor(gray, face)
                facialLandmarks = face_utils.shape_to_np(facialLandmarks)
                (x, y, w, h) = cv2.boundingRect(np.array([facialLandmarks[49:68]]))
                mouth = gray[y:y + h, x:x + w]
                mouth = cv2.resize(mouth, (100, 100))
                if frame_count % 5 == 0:
                    cv2.imwrite("words/word" + str(frame_count) + ".png", mouth)
                frame_count += 1

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                for (a, b) in facialLandmarks[49:68]:
                    cv2.circle(frame, (a, b), 1, (0, 0, 255), -1)

            cv2.imshow("WEBCAM", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break


    def stop_video(self):
        if self.isSpeak:
            self.label.setText("Please Wait...")
            self.control = False
            cv2.destroyWindow("WEBCAM")
            X = []

            image_list = os.listdir('words')
            sequence = []
            count_im=0
            for im in image_list:
                mouth = imageio.imread('words/'+im)
                mouth = mouth.astype(np.uint8)
                sequence.append(mouth)
                count_im+=1
                if(count_im>=20):
                    break
            pad_array = [np.zeros((100, 100))]
            sequence.extend(pad_array * (20 - len(sequence)))
            sequence = np.array(sequence)
            X.append(sequence)
            m=predict()
            pre_word=m.predictData(X)
            self.label.setText("You said that")
            self.text_message.setText(pre_word)
        else:
            self.label.setText("Firstly press start button")


app=QApplication([])
window=loadUi_example()
window.show()
app.exec_()