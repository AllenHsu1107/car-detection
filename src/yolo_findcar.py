import cv2
from ultralytics import YOLO

pixelx = 800
pixely = 600

model = YOLO('yolov8n.pt')  # load a pretrained YOLOv8n detection model

objName = 'car'

def DetectObject(objName,img):
    output = model(img,verbose=False)
    first_output = output[0]
    names = first_output.names
    for i in first_output.boxes:
        if names[int(i.cls)] == objName:
            x1,y1,x2,y2 = int(i.xyxy[0][0]),int(i.xyxy[0][1]),int(i.xyxy[0][2]),int(i.xyxy[0][3])
            xmid, ymid = (x1+x2)/2,(y1+y2)/2
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return xmid,ymid


path = 'example/GT_left (14).jpg'
frame = cv2.imread(path)
frame = cv2.resize(frame,(pixelx,pixely))
carx,cary = DetectObject(objName,frame)
cv2.imshow('ObjectDetection',frame)
cv2.waitKey(0)
