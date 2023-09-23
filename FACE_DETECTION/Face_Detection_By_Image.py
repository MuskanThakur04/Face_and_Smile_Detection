import cv2
import matplotlib.pyplot as plt

img = plt.imread("./group_image.jpg")
print("Shape of image is : ", img.shape)
plt.imshow(img)
plt.show()

# Loading model

detector = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

# all_faces = detector.detectMultiScale(img)

all_faces = detector.detectMultiScale(img , 1.2 , 5)    # If there are extra rectangles
print(all_faces)
print("Shape of all faces",all_faces.shape)

# # For 1 face plotting
# x , y, w, h = all_faces[0]
# img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),2)


# For all the faces

for face in all_faces:
    x,y,w,h = face
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),2)


plt.imshow(img)
plt.show()

