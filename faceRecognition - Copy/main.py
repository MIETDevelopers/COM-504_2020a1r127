#OpenCv by default uses bgr format by default so first we'll convert it to rgb format.
import cv2
import face_recognition

img = cv2.imread("AbhaySolan1.jpg")
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(rgb_img)[0]

# img2 = cv2.imread("images/Elon Musk.jpg")
# rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
# img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

img2 = cv2.imread("AbhaySolan2.jpg")
rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]


result = face_recognition.compare_faces([img_encoding], img_encoding2) #Now it'll compare the two images to see if it's the same person.
print("Result: ", result)


# cv2.imshow("Img", img)
cv2.imshow("Img", img2)
cv2.waitKey(0)