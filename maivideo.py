import cv2

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    key = cv2.waitKey(1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # 让灰度图获得三通道BGR
    cv2.rectangle(gray_bgr,(20,20),(100,100),(0,255,0),2)
    cv2.imshow('video', gray_bgr)
    if key == 113:  # 按下 'q' 键退出循环
        break
capture.release()
cv2.destroyAllWindows()

