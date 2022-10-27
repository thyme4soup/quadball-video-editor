import cv2
import imutils
import numpy as np
from keras.applications.vgg16 import preprocess_input


# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["person"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
net = cv2.dnn.readNetFromCaffe("models/VGG_16.prototxt", "models/VGG_16.caffemodel")


# Human detection vars
HOGCV = cv2.HOGDescriptor()
HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

humans = []

def detect(frame):
    frame = imutils.resize(frame, width=400)
    # grab the frame dimensions and convert it to a blob
    resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    processedimage = preprocess_input(resized_image)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(processedimage)
    detections = net.forward()
    print(detections)

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx],
                confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    cv2.imshow('output', frame)

    return frame

def get_humans(frame, i):
    if len(humans) >= i:
        return humans[i]
    else:
        detections = detect(frame)
        humans.append(detections)
        return detections
