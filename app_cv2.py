import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array



face_class = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
classifier = load_model(r'Custom_CNN_model.keras')



emotion_label = ['Angry', 'Sad' , 'Disgust', 'Happy', 'Fear', 'Neutral', 'Surprise']

cap = cv2.VideoCapture(0)

while True:
    frame = cap.read()
    grey =cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

    face_style = face_class.detectMultiscale(gray)

    for(x, y, w, h) in face_style:
        cv2.rectangle(frame , (x, y), (x +w, y+h),(0, 255, 255),2 )


        r_gray= gray[y:y+h, x:x+w]

        r_gray = cv2.resize(r_gray,(48, 48),interpolation=cv2.INTER_AREA)


        if np.sum([r_gray])!=0:
            roi = r_gray.astype('float')/255.0
            roi= np.expand_dims(roi, axis=0)

            pred= classifier.predict(roi)[0]
            label = emotion_label[pred.argmax()]
            label_pos =(x, y)


            cv2.putText(frame, label, label_pos , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('emotion_detect', frame)

    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break


cap.release()
cv2.destroyAllWindow()








