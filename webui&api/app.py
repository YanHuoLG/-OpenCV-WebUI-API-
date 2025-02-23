from flask import Flask, request, Response, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# 加载人脸检测模型
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
    return frame

# 实时视频流API
@app.route('/video_feed') 
def video_feed():
    def generate():
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = detect_faces(frame)
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        cap.release()
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 图片检测API
@app.route('/detect', methods=['POST'])
def detect_image():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    processed_img = detect_faces(img)
    _, jpeg = cv2.imencode('.jpg', processed_img)
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

# 启动服务
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, threaded=True)
