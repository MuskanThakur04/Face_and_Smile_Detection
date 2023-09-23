import cv2

detector = cv2.CascadeClassifier("./haarcascade_smile.xml")
cam = cv2.VideoCapture(0)

while True:
    ret , frame = cam.read()

    if ret == False:
        continue

    all_smiles = detector.detectMultiScale(frame,1.5,69)

    sorted_smiles = sorted(all_smiles , key= lambda s:s[-1]*s[-2])

    if sorted_smiles:
        x,y,w,h = sorted_smiles[0]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)

    # for face in all_faces:
    #     x,y,w,h = face
    #     frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)

    cv2.imshow("Smile Detection",frame)

    key_pressed = cv2.waitKey(1) & 0xFF

    if key_pressed == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()