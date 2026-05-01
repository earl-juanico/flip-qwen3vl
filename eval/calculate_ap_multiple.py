"""
This script calculates the mean average precision (mAP)
for a set of precision-recall values.

It assumes that the bounding boxes are in the format [x1, y1, x2, y2]
and confidence scores are non-existent because the language model
is not an object detector.

Corrected the case if there are number of detections (predictions) is
greater than or equal the number of ground truths (based on coco annotations)

TODO: Handle "iscrowd=True" cases for the IoU computation


Updated:
2025-12-29: Fixed bug in PR function when number of predictions is less 
            than or equal to number of ground truths.
"""
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
from scipy.integrate import simpson as simps
import pandas as pd

def calc_iou(bbox_a, bbox_b):
    x1a,y1a,x2a,y2a = bbox_a
    x1b,y1b,x2b,y2b = bbox_b
    w1 = x2a - x1a
    h1 = y2a - y1a
    w2 = x2b - x1b
    h2 = y2b - y1b
    x_left = max(x1a, x1b)
    y_top = max(y1a, y1b)
    x_right = min(x2a, x2b)
    y_bottom = min(y2a, y2b)
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area
    if union_area == 0:
        return 0.0
    return intersection_area / union_area

def calculate_ap(points:list[tuple[float,float]])->tuple[float, Image.Image]:
    x_values = [p[1] for p in points]
    y_values = [p[0] for p in points]
    if len(x_values) > 1:
        mrec = np.concatenate(([0.], x_values, [x_values[-1]]))
    elif len(x_values) == 1:
        mrec = np.concatenate(([0.], x_values, [x_values[0]]))
    else:
        mrec = np.concatenate(([0.], x_values, [1.]))
    mpre = np.concatenate(([0.], y_values, [0.]))
    for i in range(mpre.size-1,0,-1):
        mpre[i-1] = np.maximum(mpre[i-1], mpre[i])
    k = np.where(mrec[1:] != mrec[:-1])[0]
    auc = np.sum((mrec[k+1] - mrec[k]) * mpre[k+1])
    plt.plot(x_values, y_values, '-')
    plt.plot(mrec.tolist(), mpre.tolist(), 'ro--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tick_params(axis='both', which='major', labelsize=14)
    image = plt.gcf()
    return auc, image

def PR(ground_truths:list, 
       predictions:list[tuple[float, float]], 
       iou_threshold=0.5,
       conf_threshold=0.0,
       verbose=True)->tuple[int, int, int, list[float]]:
    IoU = np.zeros(len(predictions))
    assert(len(ground_truths)>0)
    if verbose: print(f'IoU size: {len(IoU)}')
    if len(predictions) > len(ground_truths):
        tp = np.zeros(len(predictions))
        fp = np.ones(len(predictions))
        fn = np.ones(len(ground_truths))
        exclude=[]
        IoU = np.zeros(len(predictions))
        for i, gt in enumerate(ground_truths):
            state=[]
            for j, pred in enumerate(predictions):
                bbox, confidence = pred
                iou = calc_iou(bbox, gt)
                if (iou >= iou_threshold and j not in exclude):
                    state.append((j,iou))
                else:
                    state.append((j,0))
            if sum(x[1] for x in state) > 0:
                k = max(state, key=lambda x: x[1])[0]
                if predictions[k][1] >= conf_threshold:
                    tp[k] = 1
                    fp[k] = 0
                    fn[i] = 0
                    exclude.append(k)
                    IoU[k] = next((s[1] for s in state if s[0]==k), 0)
        IoU = IoU.tolist()
        TP = int(np.sum(tp))
        FP = int(np.sum(fp))
        FN = int(np.sum(fn))
        if verbose: print(f'|pd|+|gt| = {len(predictions)+len(ground_truths)} vs. TP+FP+FN={int(TP+FP+FN)}')
        return TP, FP, FN, IoU
    else:
        tp = np.zeros(len(predictions))
        fp = np.zeros(len(predictions))
        fn = np.ones(len(ground_truths))
        exclude=[]
        state = []
        for i, pred in enumerate(predictions):
            bbox, confidence = pred            
            temp_state=[]
            for j, gt in enumerate(ground_truths):
                iou = calc_iou(bbox, gt)
                if iou >= iou_threshold and i not in exclude:
                    temp_state.append((i,iou))
                else:
                    temp_state.append((i,0))
            state_dict={}
            for s in temp_state:
                if s[0] not in state_dict or s[1] > state_dict[s[0]]:
                    state_dict[s[0]]=s[1]
            first_item = list(state_dict.items())[0] if state_dict else (i,0)
            state.append(first_item)
            if (sum(x[1] for x in temp_state) > 0):
                tp[i] = 1
                # set the matched ground truth index to zero in fn if available
                try:
                    matched_gt_idx = max(temp_state, key=lambda x: x[1])[0]
                    fn[matched_gt_idx] = 0
                except:
                    pass
                exclude.append(i)
            else:
                fp[i] = 1
        IoU = [s[1] if s[1] > 0 else 0 for s in state]
        TP = int(np.sum(tp))
        FP = int(np.sum(fp))
        FN = int(np.sum(fn))
        return TP, FP, FN, IoU