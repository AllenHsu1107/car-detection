import cv2
import numpy as np
from local_utils import detect_lp
from os.path import splitext
from keras.models import model_from_json
import glob

def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)

def preprocess_video(video_source,resize=False):
    _,img = video_source.read()
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img1 = img1 / 255
    if resize:
        img1 = cv2.resize(img, (224,224))
    return img,img1

def get_plate(video_source, Dmax=608, Dmin=304):
    vehicle, vehicle1 = preprocess_video(video_source)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    LPimg, cor = detect_lp(wpod_net, vehicle1, bound_dim, lp_threshold=0.5)
    return LPimg, cor, vehicle, vehicle1

def draw_box(LPimg, cor, vehicle_image, thickness=3):
    if not cor:
        return vehicle_image
    else:
        allbox=[]
        x_coordinates=[]
        y_coordinates=[]
        for i in range(len(LPimg)):
            x_coordinates.append(cor[i][0])
            y_coordinates.append(cor[i][1])
        # store the top-left, top-right, bottom-left, bottom-right 
        # of the plate license respectively
        print(len(LPimg))
        print(cor)
        print(x_coordinates[0][0])
        for i in range(len(LPimg)):
            pts=[]
            for j in range(4):
                pts.append([int(x_coordinates[i][j]),int(y_coordinates[i][j])])
            pts = np.array(pts, np.int32)
            pts = pts.reshape((-1,1,2))
            allbox.append(pts)
        for i in range(len(allbox)):
            cv2.polylines(vehicle_image,[allbox[i]],True,(0,255,0),thickness)
        return vehicle_image

wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)



video_paths = glob.glob("Plate_examples/*.mp4")


test_video = video_paths[0]
video1 = cv2.VideoCapture(test_video)
while True:
    LPimg, cor,veh_img,veh_img1 = get_plate(video1)
    img = draw_box(LPimg, cor, veh_img)
    cv2.imshow("test", img)
    key = cv2.waitKey(1) 
    if key == ord("q"): 
        break

cv2.destroyAllWindows() 
video1.release()