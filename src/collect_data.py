import cv2
import os
import numpy as np

# 用户ID
user_id = 1
# 数据存储目录
data_dir = 'data/user{}'.format(user_id)

# 创建文件夹用于保存用户的训练数据
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# 加载Haar Cascade分类器
face_cascade_path = os.path.join('C:', 'Users', 'a1339', 'Desktop', 'tools', 'python', 'Lib', 'site-packages', 'cv2', 'data', 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(face_cascade_path)
# 捕获摄像头图像
cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # 截取人脸区域
        roi_gray = gray[y:y + h, x:x + w]
        
        # 保存人脸图像
        img_path = os.path.join(data_dir, '{}.jpg'.format(count))
        cv2.imwrite(img_path, roi_gray)
        count += 1
        
        # 在图像上绘制矩形框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    # 显示捕获的图像
    cv2.imshow('Collecting Data', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f'{count} images collected for user {user_id}')