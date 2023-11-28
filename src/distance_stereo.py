import cv2
import numpy as np
from ultralytics import YOLO
from math import *
from kalman_filter import kalman_filter

# Load image or video
path_right = 'example/test_r.mp4'
cap_right = cv2.VideoCapture(path_right) if path_right.endswith('.mp4') else None
path_left = 'example/test_l.mp4'
cap_left = cv2.VideoCapture(path_left) if path_left.endswith('.mp4') else None

pixelx_left = 540
pixely_left = 960

pixelx_right = 540
pixely_right = 960

dis_camera = 0.31
w_left = 31.5/180*pi
w_right = 34/180*pi

model = YOLO('yolov8n.pt')  # load a pretrained YOLOv8n detection model

objName = 'chair'

def DetectObject(objName,img):
    output = model(img,verbose=False)
    first_output = output[0]
    names = first_output.names
    point = []
    r = []
    for i in first_output.boxes:
        if names[int(i.cls)] == objName:
            x1,y1,x2,y2 = i.xyxy[0][0],i.xyxy[0][1],i.xyxy[0][2],i.xyxy[0][3]
            xmid, ymid = (x1+x2)/2,(y1+y2)/2
            print(xmid,ymid)
            if ((ymid/xmid > (pixely_right/pixelx_right*2)) or (ymid/xmid > (-pixely_right/pixelx_right*2+2*pixely_right/xmid))):
                point.append([(x1+x2)/2,(y1+y2)/2])
                r.append([(x2-x1)/2,(y2-y1)/2])
        # print("Detected object: ", names[int(i.cls)], " with probability: ", i.conf[0], "Bounded by: ", i.xyxy)
    return point,r

#use trained cars XML classifiers
car_cascade = cv2.CascadeClassifier('cars.xml')

# pixelx_left = 1110
# pixely_left = 1474

# pixelx_right = 1109
# pixely_right = 1479

def DetectCars(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.03, 3)
    point = []
    r = []
    for (x,y,w,h) in cars:
        point.append([x+w/2,y+h/2])
        r.append((w+h)/4)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow('image', frame)
    cv2.waitKey(0)
    return point, r

def FindMatch(img_right,img_left,points_img_right,roi_list):
    # Create an ORB detector
    orb = cv2.ORB_create()
    # Detect keypoints and compute descriptors in both images
    keypoints1, descriptors1 = orb.detectAndCompute(img_right, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img_left, None)
    # Create a brute-force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors of points_img_right in img_right with descriptors in img_left
    matches = bf.match(descriptors1, descriptors2)
    # Sort matches based on their distances
    matches = sorted(matches, key=lambda x: x.distance)
    # Define a threshold for matching
    threshold = 35
    # Filter matches based on the threshold
    good_matches = [match for match in matches if match.distance < threshold]
    # Matching within a range
    final_matches_img = []
    final_matches = []
    for i in range(len(points_img_right)):
        specified_point = points_img_right[i]
        x_specified, y_specified = specified_point
        match_temp = []
        match_img_temp = []
        for match in good_matches:
            query_point_index = match.queryIdx
            train_point_index = match.trainIdx
            # Get the coordinates of the matched points in img_right and img_left
            point_img_right = keypoints1[query_point_index].pt
            point_img_left = keypoints2[train_point_index].pt
            x_img_right, y_img_right = point_img_right
            if x_specified - roi_list[i][0] <= x_img_right <= x_specified + roi_list[i][0] and \
           y_specified - roi_list[i][1] <= y_img_right <= y_specified + roi_list[i][1]:
                match_temp.append((point_img_left[0],point_img_right[0]))
                match_img_temp.append(match)
        if match_temp: final_matches.append(match_temp)
    #     if match_img_temp: final_matches_img.append(match_img_temp)
    # if final_matches_img:
    #     matched_image = cv2.drawMatches(img_right, keypoints1, img_left, keypoints2, final_matches_img[0], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #     cv2.imshow('Matches', matched_image)
    #     cv2.waitKey(0)
    return final_matches,final_matches_img

def EstDistance(match,pixelxl,pixelxr,dis_cam,anglel,angler):
    beta_left = (pi-anglel)/2
    beta_right = (pi-angler)/2
    h_out = []
    for i in range(len(match)):
        sub_match = match[i]
        h = []
        for j in range(len(sub_match)):
            P1 = pixelxl-sub_match[j][0]
            H1 = pixelxl
            P2 = sub_match[j][1]
            H2 = pixelxr
            h.append(dis_cam*sin(P2*w_right/H2+beta_right)*sin(P1*w_left/H1+beta_left)/sin(pi-(P2*w_right/H2+beta_right+P1*w_left/H1+beta_left)))
        h_out.append(sum(h)/len(h))
    return h_out


#read until video is completed
if not cap_right:
    frame_left = cv2.imread(path_left)
    # frame_left = cv2.resize(frame_left,(pixelx_left,pixely_left))
    frame_right = cv2.imread(path_right)
    # frame_right = cv2.resize(frame_right,(pixelx_right,pixely_right))
    points_img_right,r = DetectObject(objName,frame_right)
    final_match = []
    final_match_img = []
    final_match,final_match_img = FindMatch(frame_right,frame_left,points_img_right,r)
    distance = []
    distance = EstDistance(final_match,pixelx_left,pixelx_right,dis_camera,w_left,w_right)
    # print(final_match)
    print(distance)

    
    
else:

    first = True
    while True:
        #capture frame_right by frame_right
        _, frame_right = cap_right.read()
        _, frame_left = cap_left.read()
        #convert video into gray scale of each frame_rights
        # frame_left = cv2.resize(frame_left,(pixelx_left,pixely_left))
        # frame_right = cv2.resize(frame_right,(pixelx_right,pixely_right))
        points_img_right,r = DetectObject(objName,frame_right)
        final_match = []
        final_match_img = []
        final_match,final_match_img = FindMatch(frame_right,frame_left,points_img_right,r)
        distance = []
        distance = EstDistance(final_match,pixelx_left,pixelx_right,dis_camera,w_left,w_right)
        # print(final_match)
        print("Observe Distance:", distance)
        if first:
            first = False
            if distance:
                x = [distance[0], 0.0]
            else:
                x = [0.0, 0.0]
            p = np.diag([1,1])
        x,p = kalman_filter(distance, 0.05, 0.1, 5.0, x, p)
        print("Distance by k-filter:", x)

