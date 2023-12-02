import cv2
import math

def zoom_at(img, zoom_factor, center=None):

    cv2.imshow('original', img) 
    height = img.shape[0]
    width = img.shape[1]
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
    img1 = img[ top : bottom, left : right, :]

    return img1

input_image_path = '../example/chair_rf.jpg'
img = cv2.imread(input_image_path)
img_zoomed = zoom_at(img, 10.0, center=(img.shape[1]/2, img.shape[0]/2))
zoomed_image_path = '../zoomed/zoomed_example.jpg'
cv2.imwrite(zoomed_image_path, img_zoomed)