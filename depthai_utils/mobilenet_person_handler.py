import cv2
import numpy as np
from datetime import datetime

def decode_mobilenet_ssd(nnet_packet, **kwargs):
    config = kwargs['config']
    detections = []
    # the result of the MobileSSD has detection rectangles (here: entries), and we can iterate through them
    for _, e in enumerate(nnet_packet.entries()):
        # for MobileSSD entries are sorted by confidence
        # {id == -1} or {confidence == 0} is the stopper (special for OpenVINO models and MobileSSD architecture)
        if e[0]['id'] == -1.0 or e[0]['confidence'] == 0.0:
            break
        # save entry for further usage (as image package may arrive not the same time as nnet package)
        # the lower confidence threshold - the more we get false positives
        if e[0]['label'] == 15.0:
            if e[0]['confidence'] > config['depth']['confidence_threshold']:
                detections.append(e)
    return detections


def nn_to_depth_coord(x, y, nn2depth):
    x_depth = int(nn2depth['off_x'] + x * nn2depth['max_w'])
    y_depth = int(nn2depth['off_y'] + y * nn2depth['max_h'])
    return x_depth, y_depth

def average_depth_coord(pt1, pt2, padding_factor):
    factor = 1 - padding_factor
    x_shift = int((pt2[0] - pt1[0]) * factor / 2)
    y_shift = int((pt2[1] - pt1[1]) * factor / 2)
    avg_pt1 = (pt1[0] + x_shift), (pt1[1] + y_shift)
    avg_pt2 = (pt2[0] - x_shift), (pt2[1] - y_shift)
    return avg_pt1, avg_pt2

def show_mobilenet_ssd(entries_prev, frame, **kwargs):
    config = kwargs['config']
    labels = kwargs['labels']
    img_h = frame.shape[0]
    img_w = frame.shape[1]

    last_detected = datetime.now()
    # iterate through pre-saved entries & draw rectangle & text on image:
    iteration = 0
    bboxes = []
    for e in entries_prev:

        pt1 = int(e[0]['left']  * img_w), int(e[0]['top']    * img_h)
        pt2 = int(e[0]['right'] * img_w), int(e[0]['bottom'] * img_h)
        bboxes.append((pt1, pt2))
        color = (0, 0, 255) # bgr

        x1, y1 = pt1
        x2, y2 = pt2

        cv2.rectangle(frame, pt1, pt2, color)
        # Handles case where TensorEntry object label is out if range
        if e[0]['label'] > len(labels):
            print("Label index=",e[0]['label'], "is out of range. Not applying text to rectangle.")
        else:
            pt_t1 = x1, y1 + 20
            cv2.putText(frame, labels[int(e[0]['label'])], pt_t1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            pt_t2 = x1, y1 + 40
            cv2.putText(frame, '{:.2f}'.format(100*e[0]['confidence']) + ' %', pt_t2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

        iteration += 1
    return frame, bboxes

