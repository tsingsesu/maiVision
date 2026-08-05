import cv2

capture = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades +\
'haarcascade_frontalface_default.xml')

while True:
    ret, frame = capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #灰度用来检测
    faces = face_detector.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow('video', frame)

    key = cv2.waitKey(1)
    if key == 113:  # 按下 'q' 键退出循环
        break
capture.release()
cv2.destroyAllWindows()

