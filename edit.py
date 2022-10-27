import numpy as np
import cv2

# Testing imports
from perlin_noise import PerlinNoise

FILE = 'video/360-vid-test.mp4'
noise = PerlinNoise(octaves=6)

# Open the video
cap = cv2.VideoCapture(FILE)

# Initialize frame counter
cnt = 0

# Some characteristics from the original video
w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)

# Output screen size
h,w = 1080,1920

# output
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output/result.mp4', fourcc, fps, (w, h))

def get_roi_center(cnt, frame):
    perc = cnt / frames
    n1 = noise([perc, 0])
    n2 = noise([0, perc])
    usable_width, usable_height = w_frame, h_frame
    center_x, center_y = w_frame / 2, h_frame / 2
    roi_x, roi_y = center_x + n1 * usable_width / 2, center_y + n2 * usable_height / 2
    print(f'cx: {center_x}, cy: {center_y}, usable_width:{usable_width}, width:{w_frame}')
    print(f'y: {roi_y}/{roi_y/h_frame}, x: {roi_x}/{roi_x/w_frame}')
    return int(roi_y), int(roi_x)

# Get the top left corner of the frame from a center point (clips to frame)
def get_corner_from_roi(y, x):
    new_y = max(min(y - h // 2, h_frame - h - 1), 0)
    new_x = max(min(x - w // 2, w_frame - w - 1), 0)
    return new_y, new_x

while(cap.isOpened()):
    ret, frame = cap.read()

    cnt += 1 # Counting frames

    # Avoid problems when video finish
    if ret==True:
        y, x = get_corner_from_roi(*get_roi_center(cnt, frame))
        # Croping the frame
        crop_frame = frame[y:y+h, x:x+w]

        # Percentage
        xx = cnt *100/frames
        print(int(xx),'%')

        out.write(crop_frame)

        # Just to see the video in real time
        cv2.imshow('frame',frame)
        cv2.imshow('cropped',crop_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


cap.release()
out.release()
cv2.destroyAllWindows()
