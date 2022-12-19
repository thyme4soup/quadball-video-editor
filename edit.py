import numpy as np
import cv2
# import human_tracker
import encode_with_centers
from PIL import Image
from pykalman import KalmanFilter

# Testing imports
from perlin_noise import PerlinNoise

FILE = 'video/quad-test.mp4'
noise = PerlinNoise(octaves=6)

# Open the video
cap = cv2.VideoCapture(FILE)

# Initialize frame counter
cnt = 0

# Some characteristics from the original video
w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)

# Output screen size
h,w = 720,1280

# output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output/result.mp4', fourcc, fps, (w, h))

def get_roi_center(cnt, frame):
    # Convert frame to PIL image
    color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(color_frame)
    processed = encode_with_centers.process_image(im)

    # Pass frame to model wrapper
    attention = encode_with_centers.get_center_from_image(processed, lite_model_file='./models/center.tflite')
    return attention

    '''
    perc = cnt / frames
    n1 = noise([perc, 0])
    n2 = noise([0, perc])
    usable_width, usable_height = w_frame, h_frame
    center_x, center_y = w_frame / 2, h_frame / 2
    roi_x, roi_y = center_x + n1 * usable_width / 2, center_y + n2 * usable_height / 2
    # print(f'cx: {center_x}, cy: {center_y}, usable_width:{usable_width}, width:{w_frame}')
    # print(f'y: {roi_y}/{roi_y/h_frame}, x: {roi_x}/{roi_x/w_frame}')
    return int(roi_y), int(roi_x)
    '''

# Get the top left corner of the frame from a center point (clips to frame)
def get_corner_from_roi(y, x):
    new_y = max(min(y - h // 2, h_frame - h - 1), 0)
    new_x = max(min(x - w // 2, w_frame - w - 1), 0)
    return new_y, new_x

# TODO: Bump up history a lot
max_pop = 120
centers = []
def get_stabilized_center(cnt, frame):
    center = get_roi_center(cnt, frame)
    centers.append(center)
    if len(centers) > max_pop:
        centers.pop(0)
    average = [sum(x)//len(x) for x in zip(*centers)]
    return average, center

while(cap.isOpened()):
    ret, frame = cap.read()

    cnt += 1 # Counting frames

    # Avoid problems when video finish
    if ret==True:
        stab_center, actual_center = get_stabilized_center(cnt, frame)
        center_x, center_y = stab_center
        y, x = get_corner_from_roi(center_y, center_x)
        # Croping the frame
        crop_frame = frame[y:y+h, x:x+w].copy()
        # Percentage
        xx = cnt *100/frames
        print(int(xx),'%')

        out.write(crop_frame)

        # humans
        # human_tracker.get_humans(frame, cnt)

        # Just to see the video in real time
        cv2.circle(frame, stab_center, 50, (255, 0, 0), 5)
        cv2.circle(frame, actual_center, 15, (0, 255, 0), 5)
        cv2.imshow('frame',frame)
        cv2.imshow('cropped',crop_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("?")
            break
    else:
        print("??")


cap.release()
out.release()
cv2.destroyAllWindows()
