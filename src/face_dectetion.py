import cv2
import numpy as np
# 加载人脸检测的Haar级联文件
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # 如果读取成功
    if not ret:
        break

    # 转为灰度图，便于检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # 绘制矩形框标识出人脸
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 显示结果
    cv2.imshow('Face Detection', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源
cap.release()
cv2.destroyAllWindows()
