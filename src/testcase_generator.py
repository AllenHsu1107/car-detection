import cv2
import math
from spring_mass import spring_mass
from ultralytics import YOLO
import os
import numpy as np
import shutil

def zoom_in(image, zoom_factor, center=None):

    cv2.imshow('original', image) 
    image_shape = image.shape
    height = image_shape[0]
    width = image_shape[1]
    if center is None:
        cx = width / 2
        cy = height / 2
    else:
        cx = center[0]
        cy = center[1]

    left = int(cx - cx / zoom_factor)
    right = int((cx + ((width - cx) / zoom_factor))+1)
    top = int(cy - cy / zoom_factor)
    bottom = int((cy + ((height - cy) / zoom_factor))+1)
    image = image[ top : bottom, left : right, :]
    image = cv2.resize(image, image_shape[:2])

    return image

def DetectObject(objName, original_image):
    image = np.copy(original_image)
    output = model(image,verbose=False)
    first_output = output[0]
    names = first_output.names
    for i in first_output.boxes:
        if names[int(i.cls)] == objName:
            x1,y1,x2,y2 = int(i.xyxy[0][0]),int(i.xyxy[0][1]),int(i.xyxy[0][2]),int(i.xyxy[0][3])
            xmid, ymid = (x1+x2)/2,(y1+y2)/2
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return (xmid, ymid)

def create_image(image, center, acutal_distance, target_distance):
    zoom_factor = acutal_distance / target_distance
    image = zoom_in(image, zoom_factor, center)
    return image

def remove_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f'The folder {folder_path} has been removed.')
    else:
        print(f'The folder {folder_path} does not exist.')

def create_folder(folder_path):
    try:
        os.makedirs(folder_path)
        print(f'The folder {folder_path} has been created.')
    except FileExistsError:
        print(f"Folder '{folder_path}' already exists.")

if __name__ == "__main__":
    testcase_folder_path = '../testcases'
    testcase_image_path = f'{testcase_folder_path}/images'
    remove_folder(testcase_image_path)
    create_folder(testcase_image_path)

    model = YOLO('yolov8n.pt')  # load a pretrained YOLOv8n detection model
    objName = 'car'
    time_interval = 0.1
    true_distances, _ = spring_mass(time_interval)
    step = 7.4
    offset = 1.6
    for i, true_distance in enumerate(true_distances):
        multiple = int(math.ceil((true_distance - offset) / step))
        photo_distance = multiple * step + offset
        input_image_path = f'../raw_photos/richard/{multiple}.jpg'
        image = cv2.imread(input_image_path)
        center = DetectObject(objName, image)
        created_image = create_image(image, center, photo_distance, true_distance)
        zoomed_image_path = f'{testcase_image_path}/{i}.jpg'
        cv2.imwrite(zoomed_image_path, created_image)

    testcase_distance_path = f'{testcase_folder_path}/ground_truth.txt'
    with open(testcase_distance_path, "w") as file:
        file.write(f'The time interval between two adjacent distances is {time_interval}\n')
        file.write('Ground truth distances:\n')
        for true_distance in true_distances:
            file.write(f'{true_distance}\n')