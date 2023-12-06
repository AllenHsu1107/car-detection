import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # load a pretrained YOLOv8n detection model

pixelx = 800
pixely = 600

objName = 'car'

def DetectObject(objName,img):
    py,px = img.shape[0],img.shape[1]
    output = model(img,verbose=False)
    first_output = output[0]
    names = first_output.names
    for i in first_output.boxes:
        if names[int(i.cls)] == objName:
            x1,y1,x2,y2 = int(i.xyxy[0][0]),int(i.xyxy[0][1]),int(i.xyxy[0][2]),int(i.xyxy[0][3])
            xmid, ymid = (x1+x2)/2,(y1+y2)/2
            if ((ymid/xmid < (py/px*8/3-2/3*py/xmid)) and (ymid/xmid < (-py/px*8/3+2*py/xmid)) and (ymid < 3/4*py)):
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                finalx,finaly = (x1+x2)/2,(y1+y2)/2
    return finalx,finaly

path = 'example/GT_left (6).jpg'
frame = cv2.imread(path)
x,y = DetectObject(objName,frame)
print(x,y)
frame = cv2.resize(frame,(pixelx,pixely))
cv2.imshow('ObjectDetection',frame)
cv2.waitKey(0)
