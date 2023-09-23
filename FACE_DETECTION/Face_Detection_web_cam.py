import cv2

cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

while True:
    ret , frame = cam.read()

    if ret == False:
        continue

    all_faces = detector.detectMultiScale(frame,1.2,5)

    for face in all_faces:
        x,y,w,h = face
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)

    cv2.imshow("Face Detection",frame)

    key_pressed = cv2.waitKey(1) & 0xFF

    if key_pressed == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()