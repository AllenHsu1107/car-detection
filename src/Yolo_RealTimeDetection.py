import cv2
from ultralytics import YOLO

# Load image or video
cap = cv2.VideoCapture(0)
model = YOLO('yolov8n.pt')  # load a pretrained YOLOv8n detection model

def DetectObject(img):
    output = model(img,verbose=False)
    first_output = output[0]
    for i in first_output.boxes:
        x1,y1,x2,y2 = int(i.xyxy[0][0]),int(i.xyxy[0][1]),int(i.xyxy[0][2]),int(i.xyxy[0][3])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return x1,y1,x2,y2



#read until video is completed
while True:
    #capture frame_right by frame_right
    _, frame = cap.read()
    x1,y1,x2,y2 = DetectObject(frame)
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
