import cv2
import numpy as np
import obd
import time
from kalman_filter import kalman_filter
from math import *
from ultralytics import YOLO

pixelx = 800
pixely = 600

connection = obd.OBD()
if connection.is_connected():
    print("OBD-II connection successful")
else:
    print("OBD-II connection failed. Make sure your OBD-II adapter is connected properly.")
    exit()

def get_speed():
    cmd = obd.commands.SPEED
    response = connection.query(cmd)

    if response.is_null():
        print("Speed data not available.")
    else:
        speed_kph = response.value.magnitude
        speed_mps = speed_kph/3.6
    return speed_mps

cap = cv2.VideoCapture(0)

model = YOLO('yolov8n.pt')  # load a pretrained YOLOv8n detection model

objName = 'car'

f = 720

minwidth = 1.76
maxwidth = 2.06

def Zoomin(img, times):
    py,px = img.shape[0],img.shape[1]
    crp_img = img[int(py*(1-1/times)/2):int(py*(1+1/times)/2), int(px*(1-1/times)/2):int(px*(1+1/times)/2)]
    return crp_img

def DetectObject(objName,img,n=1):
    py,px = img.shape[0],img.shape[1]
    output = model(img,verbose=False)
    first_output = output[0]
    names = first_output.names
    width = 0
    for i in first_output.boxes:
        if names[int(i.cls)] == objName:
            x1,y1,x2,y2 = int(i.xyxy[0][0]),int(i.xyxy[0][1]),int(i.xyxy[0][2]),int(i.xyxy[0][3])
            xmid, ymid = (x1+x2)/2,(y1+y2)/2
            if ((ymid/xmid < (py/px*8/3-2/3*py/xmid)) and (ymid/xmid < (-py/px*8/3+2*py/xmid)) and (ymid < 3/4*py)):
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                width = max(width,x2-x1)
    while not width and n < 2:
        n = n+1
        crp_frame = Zoomin(img,n)
        width = DetectObject(objName,crp_frame,n)
    return width

def EstDistance(pixellength):
    d1 = f*minwidth/pixellength
    d2 = f*maxwidth/pixellength
    return d1,d2

first = True
frameNumber = 0
frameTime = 0
x1 = [0.0, 0.0]
p1 = np.diag([1,1])
x2 = [0.0, 0.0]
p2 = np.diag([1,1])
start = time.time()

while True:
    frameNumber = frameNumber+1
    _, frame = cap.read()
    speed_obd = get_speed()
    pixelrange = DetectObject(objName,frame)
    d1,d2 = 0,0
    if pixelrange:
        d1,d2 = EstDistance(pixelrange)
        frameTime = time.time()-start
        if first:
            first = False
            frameNumber = 0
            x1 = [d1, 0.0]
            p1 = np.diag([1,1])
            x2 = [d2, 0.0]
            p2 = np.diag([1,1])
            frameTime = 0
        x1,p1 = kalman_filter(d1, frameTime, 0.1, 5.0, x1, p1)
        x2,p2 = kalman_filter(d2, frameTime, 0.1, 5.0, x2, p2)
        print('Min Distance: ',x1[0],'Max Distance: ',x2[0])
        if (x1[0]/max(speed_obd,speed_obd-x1[1]))<=3:
            print("WARNING!!TOO CLOSE!!")
        frameNumber = 0
        start = time.time()
    else:
        if frameNumber>10:
            first = True
    frame = cv2.resize(frame,(pixelx,pixely))
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
connection.close()