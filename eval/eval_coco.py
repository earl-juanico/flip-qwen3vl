"""
This script extracts the detections (aka predictions)
from the VLM known as LLaVA. 

Updates:
2025-12-30: Scale font sizes and thicknesses according to image size to avoid clutter.
2025-12-29: Modified to support environment variable for COCO_ROOT path.
"""
import re
import json
import ast
import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COCO_SYNSET_PATH = os.path.join(BASE_DIR, 'coco_synset.jsonl')
LVIS_TO_COCO_MAP = os.path.join(BASE_DIR, 'coco_to_synset.json')

# Load coco synset categories (file next to this script)
with open(COCO_SYNSET_PATH,'r') as f:
    COCO_SYNSET_CATEGORIES = [json.loads(line) for line in f]

# Use COCO_ROOT env var; fallback to legacy path
COCO_ROOT = os.environ.get('COCO_ROOT', '/data/vlm/playground/')
# Accept user providing either '/.../playground' or '/.../playground/data'
if os.path.basename(COCO_ROOT.rstrip('/')) == 'data':
    DATA_ROOT = COCO_ROOT
else:
    DATA_ROOT = os.path.join(COCO_ROOT, 'data')

coco_data_json = os.path.join(DATA_ROOT, 'coco', 'annotations', 'instances_val2017.json')
coco_caps_json = os.path.join(DATA_ROOT, 'coco', 'annotations', 'captions_val2017.json')
lvis_data_json = os.path.join(DATA_ROOT, 'lvis_v1', 'annotations', 'lvis_v1_val.json')

# Try to load files (fail early with descriptive message)
for p in (coco_data_json, coco_caps_json, LVIS_TO_COCO_MAP, lvis_data_json):
    if not os.path.exists(p):
        # don't raise, but warn — consumer scripts may handle missing files
        print(f"[warn] expected file not found: {p}")

with open(coco_data_json, 'r') as f:
    coco_data = json.load(f)
with open(coco_caps_json, 'r') as f:
    coco_caps = json.load(f)
with open(LVIS_TO_COCO_MAP,'r') as f:
    synset_lvis = json.load(f)
#with open(lvis_data_json, 'r') as f:
#    lvis_data = json.load(f)

objects = coco_data.get('annotations', [])
images = coco_data.get('images', [])
cats = coco_data.get('categories', [])
#lats = lvis_data.get('categories', [])
captions = coco_caps.get('annotations', [])

def get_predictions_from_llava(text: str) -> list:
    pattern1 = r'\[(\d+\.\d+), (\d+\.\d+), (\d+\.\d+), (\d+\.\d+)\]'
    pattern2 = r'\((\d+\.\d+), (\d+\.\d+), (\d+\.\d+), (\d+\.\d+)\)'
    matches1 = re.findall(pattern1, text)
    matches2 = re.findall(pattern2, text)
    numbers1 = [[float(match) for match in match_tuple] for match_tuple in matches1]
    numbers2 = [[float(match) for match in match_tuple] for match_tuple in matches2]
    numbers2 = [f'[{", ".join(map(str, match_tuple))}]' for match_tuple in numbers2]
    numbers2 = [ast.literal_eval(match_tuple) for match_tuple in numbers2]
    predictions = numbers1 + numbers2
    if not predictions:
        predictions = [[0.0, 0.0, 0.0, 0.0]]
    return predictions

def convert_json_to_jsonl(json_filename: str)-> str:
    with open(json_filename, 'r') as f:
        json_data = json.load(f)
    jsonl_filename = json_filename.replace('.json','.jsonl')
    with open(jsonl_filename, 'w') as f:
        for item in json_data:
            f.write(json.dumps(item) + '\n')
    return jsonl_filename

def object_type_in_image(image_id: int, cat_id: int) -> list:
    object_bbox = []
    H = next((image['height'] for image in images if image['id']==image_id),None)
    W = next((image['width'] for image in images if image['id']==image_id),None)
    for object in objects:
        if object['image_id'] == image_id and object['category_id'] == cat_id:
            x,y,w,h = object['bbox']
            if W and H:
                object_bbox.append(f'[{x/W:.2f}, {y/H:.2f}, {(x+w)/W:.2f}, {(y+h)/H:.2f}]')
    return object_bbox

def draw_bboxes(image_path: str, cls: list, bbox_gt: list, bbox_pt: list, TP:int, FP:int, IoU:list, save_path:str):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[warn] cannot open image: {image_path}")
        return
    H, W, _ = image.shape
    if not cls:
        category = "unknown"
    else:
        category = cls[0]
    text = category
    try:
        font = cv2.FONT_HERSHEY_SIMPLEX
    except:
        font = cv2.FONT_HERSHEY_SIMPLEX

    # scale fonts and thickness according to image size to avoid clutter
    # use max dimension to determine visual scale; normalized against 1000px baseline
    scale_factor = max(0.4, min(2.5, max(W, H) / 1000.0))
    title_font_scale = 0.9 * scale_factor
    info_font_scale = 0.75 * scale_factor
    iou_font_scale = 0.6 * scale_factor
    title_thickness = max(1, int(round(2 * scale_factor)))
    info_thickness = max(1, int(round(1.5 * scale_factor)))
    iou_thickness = max(1, int(round(1 * scale_factor)))

    # thickness used for bbox borders (make dashed red match solid green)
    border_thickness = max(1, int(round(3 * scale_factor)))

    font_color = (255, 0, 0)
    thickness = title_thickness

    text_size, _ = cv2.getTextSize(text, font, title_font_scale, title_thickness)
    text_x = (W - text_size[0]) // 2
    text_y = text_size[1] + int(20 * scale_factor)
    tp_text = f"TP: {TP}"
    fp_text = f"FP: {FP}"
    (tw1, th1), _ = cv2.getTextSize(text, font, title_font_scale, title_thickness)
    (tw2, th2), _ = cv2.getTextSize(tp_text, font, info_font_scale, info_thickness)
    (tw3, th3), _ = cv2.getTextSize(fp_text, font, info_font_scale, info_thickness)
    (text_width, text_height) = (max(tw1, tw2, tw3), th1 + th2 + th3)
    text_x = W - text_width
    text_y = text_height
    image = cv2.rectangle(image,(text_x,text_y-15),(text_x+text_width+2,text_y+text_height+15),(0,255,255),cv2.FILLED)
    image = cv2.putText(image, text, (text_x, text_y), font, title_font_scale, font_color, title_thickness)
    image = cv2.putText(image, tp_text, (text_x, text_y + int(30 * scale_factor)), font, info_font_scale, font_color, info_thickness)
    image = cv2.putText(image, fp_text, (text_x, text_y + int(60 * scale_factor)), font, info_font_scale, font_color, info_thickness)
    for bbox in bbox_gt:
        try:
            x1, y1, x2, y2 = ast.literal_eval(bbox)
            x1, y1, x2, y2 = int(x1*W), int(y1*H), int(x2*W), int(y2*H)
            image = cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), border_thickness)
        except:
            continue
    for i, bbox in enumerate(bbox_pt):
        try:
            x1, y1, x2, y2 = ast.literal_eval(bbox)
            x1, y1, x2, y2 = int(x1*W), int(y1*H), int(x2*W), int(y2*H)
        except:
            continue
        dash_length = max(6, int(round(10 * scale_factor)))
        gap_length = max(4, int(round(6 * scale_factor)))
        for x in range(x1, x2, dash_length + gap_length):
            x_end = min(x + dash_length, x2)
            image = cv2.line(image, (x, y1), (x_end, y1), (0,0,255), border_thickness)
        for x in range(x1, x2, dash_length + gap_length):
            x_end = min(x + dash_length, x2)
            image = cv2.line(image, (x, y2), (x_end, y2), (0,0,255), border_thickness)
        for y in range(y1, y2, dash_length + gap_length):
            y_end = min(y + dash_length, y2)
            image = cv2.line(image, (x1, y), (x1, y_end), (0,0,255), border_thickness)
        for y in range(y1, y2, dash_length + gap_length):
            y_end = min(y + dash_length, y2)
            image = cv2.line(image, (x2, y), (x2, y_end), (0,0,255), border_thickness)
        #iou_text = f"IoU: {IoU[i]:.2f}" if i < len(IoU) else "IoU: 0.00"
        iou_text = f"{IoU[i]:.2f}" if i < len(IoU) else "0.00"
        (text_width_iou, text_height_iou), _ = cv2.getTextSize(iou_text, font, iou_font_scale, iou_thickness)
        box_coords = ((x1, y1+int(10 * scale_factor)), (x1 + text_width_iou + int(2 * scale_factor), y1 + int(10 * scale_factor) - text_height_iou - int(2 * scale_factor)))
        image = cv2.rectangle(image, box_coords[0], box_coords[1], (255,0,0), cv2.FILLED)
        image = cv2.putText(img=image, text=iou_text, org=box_coords[0], fontFace=font, fontScale=iou_font_scale, color=(0,255,255), thickness=iou_thickness)
    # ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, image)

import statistics
def evaluate_coco(coco_gt_file: str, llava_answers_file: str):
    ground_truth=[]
    grounds=[]
    cls=[]
    with open(coco_gt_file,'r') as f:
        lines = f.readlines()
        grounds = [json.loads(line).get('answers', '') for line in lines]
        cls = [json.loads(line).get('class','') for line in lines]
    for gt in grounds:
        ground_truth.append([ast.literal_eval(g) for g in gt])
   
    predictions=[]
    with open(llava_answers_file,'r') as f:
        lines = f.readlines()
        predictions = [[(bbox, round((bbox[2]-bbox[0])*(bbox[3]-bbox[1]), 4)) for bbox in get_predictions_from_llava(json.loads(line).get('text', ''))] for line in lines]
        #print(f'Predicted boxes with areas: {predictions}')
    '''
    detections=[]
    confs=[]
    with open(llava_answers_file,'r') as f:
        lines = f.readlines()
        predictions = [get_predictions_from_llava(json.loads(line).get('text', '')) for line in lines]
        # predictions must be in bbox list format after running extract_bbox_from_answer.py beforehand
        #  the confidence proxy can be calculated from the bbox area
        print(f'Predicted boxes: {predictions}')
    for pred in predictions:
        for bbox in pred:
            x1,y1,x2,y2=tuple(bbox)
            area = (x2-x1)*(y2-y1)
            detections.append((bbox,round(area,4)))
            confs.append(area)
    #predictions = detections
    # debug 
    print(f'Transformed predictions: {detections}')  # debug
    print(f'Conf Maximum: {max(confs)}')
    print(f'Conf Minimum: {min(confs)}')
    print(f'Conf Ave: {statistics.mean(confs)}')
    print(f'Conf Median: {statistics.median(confs)}')
    print(f'Conf Mode: {statistics.mode(confs)}')
    print(f'Conf Variance: {statistics.variance(confs)}')
    
    print(f'Number of gt prompts answered: {len(ground_truth)}') #debug
    print(f'Number of pred prompts answered: {len(predictions)}') #debug
    #assert(len(ground_truth)==len(predictions))
    '''
    return cls, ground_truth, predictions