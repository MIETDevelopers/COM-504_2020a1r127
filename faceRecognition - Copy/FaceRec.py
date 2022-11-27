# [120 420 264 2765] -> Here 120,420 are the top left point where a face has been detected and 264,2765 the the bottom right point where a face has been detected. So, a rectangle/square would be drawn around the detected face.

import cv2
from simple_facerec import SimpleFacerec

cap = cv2.VideoCapture(0)

# Encodefaces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")


# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)

    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        # print(face_loc)
        y1, x1, y2, x2 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name, (x1,y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 0, 200), 1)

    cv2.imshow('Input', frame)

    c = cv2.waitKey(1)
    if c==27:
        break

cap.release()
cv2.destroyAllWindows()
