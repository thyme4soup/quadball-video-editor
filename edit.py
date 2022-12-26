import cv2
import time
# import human_tracker
import encode_with_centers
from PIL import Image
from alive_progress import alive_bar, alive_it
from vidstab.VidStab import VidStab

# Testing imports
from perlin_noise import PerlinNoise

FILE = 'video/quad-test.mp4'
FILES = [
    'video/BREAKERSvVIPERS.1.1.MP4',
    'video/BREAKERSvVIPERS.1.2.MP4',
    'video/BREAKERSvVIPERS.1.3.MP4',
    'video/BREAKERSvVIPERS.1.4.MP4',
]
SHOW_VIDEO = False

noise = PerlinNoise(octaves=6)

# Initialize filters
bg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
stabilizer = VidStab()
smoothing_window = 30

# Open video and get some characteristics
caps = [cv2.VideoCapture(file) for file in FILES]
w_frame, h_frame = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH)), int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
fps, frames = caps[0].get(cv2.CAP_PROP_FPS), caps[0].get(cv2.CAP_PROP_FRAME_COUNT)

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

# Define stab as last n seconds
max_pop = 10 * fps
centers = []
def get_stabilized_center(cnt, frame):
    center = get_roi_center(cnt, frame)
    centers.append(center)
    if len(centers) > max_pop:
        centers.pop(0)
    average = [sum(x)//len(x) for x in zip(*centers)]
    return average, center

for index, cap in enumerate(caps):
    print(f'File {index + 1} of {len(caps)}')
    # Initialize frame counter
    cnt = 0
    with alive_bar(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as bar:
        while(cap.isOpened()):
            ret, frame = cap.read()
            cnt += 1
            bar()

            # Avoid problems when video finish
            if ret==True:
                # Stabilize frame (destructive)
                frame = stabilizer.stabilize_frame(input_frame=frame, smoothing_window=smoothing_window)

                # Perform background subtraction
                bg_removed = bg.apply(frame)

                # Get attention center
                stab_center, actual_center = get_stabilized_center(cnt, bg_removed)
                center_x, center_y = stab_center
                y, x = get_corner_from_roi(center_y, center_x)

                # Crop the frame using attention center
                crop_frame = frame[y:y+h, x:x+w].copy()

                out.write(crop_frame)

                # humans
                # human_tracker.get_humans(frame, cnt)

                if SHOW_VIDEO:
                    # Just to see the video in real time
                    cv2.circle(frame, stab_center, 50, (255, 0, 0), 5)
                    cv2.circle(frame, actual_center, 15, (0, 255, 0), 5)
                    cv2.imshow('frame',frame)
                    cv2.imshow('cropped',crop_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Found break key")
                    break
            elif cnt > frames * 1.1:
                break
        cap.release()

out.release()
cv2.destroyAllWindows()
