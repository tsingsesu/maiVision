import cv2

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    key = cv2.waitKey(1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    #灰度用来检测
    cv2.rectangle(frame,(20,20),(100,100),(0,255,0),2) #彩色用来画框
    cv2.imshow('video', frame)
    if key == 113:  # 按下 'q' 键退出循环
        break
capture.release()
cv2.destroyAllWindows()

