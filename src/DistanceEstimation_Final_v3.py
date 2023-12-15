import cv2
import numpy as np
import obd
import time
from kalman_filter import kalman_filter
from math import *

pixelx = 800
pixely = 600

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f]

net = cv2.dnn.readNet("yolov3-tiny.cfg", "yolov3-tiny_final.weights")

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

objName = 'bicycle'

f = 720

minwidth = 1.76
maxwidth = 2.06

def Zoomin(img, times):
    py,px = img.shape[0],img.shape[1]
    crp_img = img[int(py*(1-1/times)/2):int(py*(1+1/times)/2), int(px*(1-1/times)/2):int(px*(1+1/times)/2)]
    return crp_img

def DetectObject(objName,frame,n=1):
    # Get the frame dimensions
    height, width = frame.shape[:2]

    # Preprocess the image for YOLO input
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get output layer names
    layer_names = net.getUnconnectedOutLayersNames()

    # Forward pass through the network
    outs = net.forward(layer_names)

    # Get bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    pixel = 0
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                if classes[class_id] == objName:
                    xmid = int(detection[0] * width)
                    ymid = int(detection[1] * height)
                    if ((ymid/xmid < (height/width*8/3-2/3*height/xmid)) and (ymid/xmid < (-height/width*8/3+2*height/xmid)) and (ymid < 3/4*height)):
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(xmid - w / 2)
                        y = int(ymid - h / 2)
                        pixel = max(pixel,w)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        # class_ids.append(class_id)

    # Apply non-maximum suppression to remove redundant boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # Draw bounding boxes on the frame
    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        # label = classes[class_ids[i]]
        # confidence = confidences[i]

        color = (0, 255, 0)  # Green
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        # cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    while not pixel and n < 4:
        n = n+1
        crp_frame = Zoomin(frame,n)
        pixel = DetectObject(objName,crp_frame,n)
    return pixel

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