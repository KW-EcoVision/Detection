from xml.etree import ElementTree as ET
import json
import cv2
import numpy as np
import cv2 as cv
from tqdm import tqdm
from torchvision.transforms import transforms
import os
import torch
import config
import random

'''
'xml_path' is mapped by each img path
'img_dir' is not direct_path just dir
'img_path' is direct img_path
'''

img_path_abs = '/media/unsi/media/data/생활 폐기물 이미지/Training/'
classes_num = {'종이류': 0, '플라스틱류': 1, '유리병류': 2, '캔류': 3, '고철류': 4, '의류': 5,
               '전자제품': 6, '스티로폼류': 7, '도기류': 8, '비닐류': 9, '가구': 10, '자전거': 11,
               '형광등': 12, '페트병류': 13, '나무류': 14}
classes_num_rev = {
    0: 'paper',
    1: 'plastic',
    2: 'glass',
    3: 'cans',
    4: 'scrap metal',
    5: 'clothes',
    6: 'electronics',
    7: 'styrofoam',
    8: 'pottery',
    9: 'Vinyls',
    10: 'furniture',
    11: 'bicycle',
    12: 'fluorescent lamp',
    13: 'PET bottles',
    14: 'trees',
}

IMG_SIZE = 224.0

CELL = 32.0

from PIL import Image as image


def ParseJson(Json):
    img_path = Json['img_path']
    bbox = Json['bbox']
    cls = Json['cls']
    gps: tuple = Json['gps'].split('_')  # 위도 경도 tuple
    img = np.array(image.open(img_path))
    width = img.shape[1]
    height = img.shape[0]
    label = np.zeros(shape=(7, 7, 20), dtype=float)
    ration_x = IMG_SIZE / width
    ratio_y = IMG_SIZE / height

    class_name = classes_num_rev[cls]

    xmin = float(bbox[0])
    ymin = float(bbox[1])
    xmax = float(bbox[2])
    ymax = float(bbox[3])

    cx = xmin + ((xmax - xmin) / 2)
    cy = ymin + ((ymax - ymin) / 2)
    w = xmax - xmin
    h = ymax - ymin

    # mapped by new ratio
    rcx = cx * ration_x
    rcy = cy * ratio_y
    rw = w * ration_x
    rh = h * ratio_y

    cell_position_x = int(rcx / CELL)
    cell_position_y = int(rcy / CELL)

    # 각 셀의 시작 x, y 좌표를 계산합니다.
    start_x = cell_position_x * CELL
    start_y = cell_position_y * CELL

    rcx = (rcx - start_x) / CELL
    rcy = (rcy - start_y) / CELL

    rw = rw / IMG_SIZE
    rh = rh / IMG_SIZE

    label[cell_position_x, cell_position_y][15] = rcx
    label[cell_position_x, cell_position_y][16] = rcy
    label[cell_position_x, cell_position_y][17] = rw
    label[cell_position_x, cell_position_y][18] = rh
    label[cell_position_x, cell_position_y][19] = 1.0
    label[cell_position_x, cell_position_y][cls] = 1.0
    img = cv2.resize(img / 224.0, dsize=(224, 224))
    img = (torch.from_numpy(img)).permute(2, 0, 1)
    label = torch.from_numpy(label).float()

    return img, label


def decoding_label(label):
    bbox = []
    class_box = []

    for i in range(7):
        for j in range(7):
            cx = (label[i][j][15] * CELL) + (i * CELL)
            cy = (label[i][j][16] * CELL) + (j * CELL)
            w = label[i][j][17] * IMG_SIZE
            h = label[i][j][18] * IMG_SIZE

            xmin = int(cx - w / 2)
            xmax = int(cx + w / 2)
            ymin = int(cy - h / 2)
            ymax = int(cy + h / 2)
            conf = label[i][j][24]
            class_pred = np.max(label[i][j][:15])
            bbox.append([class_pred, conf, xmin, ymin, xmax, ymax])
            class_box.append(label[i][j][:15])

        for j in range(7):
            cx = (label[i][j][21] * CELL) + (i * CELL)
            cy = label[i][j][22] * CELL + (j * CELL)
            w = label[i][j][23] * IMG_SIZE
            h = label[i][j][24] * IMG_SIZE

            xmin = int(cx - w / 2)
            xmax = int(cx + w / 2)
            ymin = int(cy - h / 2)
            ymax = int(cy + h / 2)
            conf = label[i][j][25]
            class_pred = np.max(label[i][j][:15])
            bbox.append([class_pred, conf, xmin, ymin, xmax, ymax])
            class_box.append(label[i][j][:15])
    return bbox, class_box


def for_test_decode_label(label):
    bbox = []
    class_box = []

    for i in range(7):
        for j in range(7):
            cx = (label[i][j][15] * CELL) + (i * CELL)
            cy = (label[i][j][16] * CELL) + (j * CELL)
            w = label[i][j][17] * IMG_SIZE
            h = label[i][j][18] * IMG_SIZE

            xmin = int(cx - w / 2)
            xmax = int(cx + w / 2)
            ymin = int(cy - h / 2)
            ymax = int(cy + h / 2)
            conf = label[i][j][19]
            class_pred = np.max(label[i][j][:15])
            bbox.append([class_pred, conf, xmin, ymin, xmax, ymax])
            class_box.append(label[i][j][:15])
    return bbox, class_box


def draw(bbox_list, img, class_list):
    img = img.astype(np.uint8).copy()
    for i in range(len(bbox_list)):
        img = cv.rectangle(
            img=img,
            pt1=(bbox_list[i][2], bbox_list[i][3]),
            pt2=(bbox_list[i][4], bbox_list[i][5]),
            color=(0, 225, 0), thickness=1)
        max_val = np.max(class_list[i])
        max_index = np.argmax(class_list[i])
        img = cv2.putText(
            img,
            f'{classes_num_rev[max_index]} | {np.max(class_list[i])}',
            (bbox_list[i][2], bbox_list[i][3]),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA
        )

    return img


def get_augmentor():
    return transforms.Compose([
        transforms.Resize(size=(int(IMG_SIZE), int(IMG_SIZE))),
        transforms.ToTensor()
    ])


def get_json_data(json_dir='/media/unsi/media/data/생활 폐기물 이미지/label/Training_라벨링데이터'):
    path_list = []
    for file_path, _, file_name in os.walk(json_dir):
        for name in file_name:
            full_path = os.path.join(file_path, name)
            path_list.append(full_path)
    label = []

    for i in tqdm(path_list):
        try:
            with open(i, 'r', encoding='utf-8') as file:
                raw_json = json.load(file)
                file_name = raw_json['FILE NAME']
                file_name_tmp = file_name.rpartition('_')[0]
                GPS = raw_json['GPS']

                location = raw_json['Bounding'].pop()

                cls = location['CLASS']
                detail = location['DETAILS']
                path = os.path.join(img_path_abs, '[T원천]' + cls + '_' + detail + '_' + detail)
                x1 = location['x1']
                y1 = location['y1']
                x2 = location['x2']
                y2 = location['y2']
                tmp_path = os.path.join(path, file_name_tmp, file_name)
                if os.path.exists(tmp_path):
                    label.append(dict(img_path=tmp_path, bbox=(x1, y1, x2, y2),
                                      cls=classes_num[cls], gps=GPS))
        except json.JSONDecodeError as e:
            print(f'\n[ERROR] JSONDecodeError 발생\n')
        except Exception as e:
            print(f'\n[ERROR] 파일 처리 중 오류 발생\n')
    random.shuffle(label)
    with open('./output.json', 'w', encoding='utf-8') as file:
        json.dump(label, file, ensure_ascii=False, indent=4)
    return label


def load_and_split_json(dir='output.json'):
    with open(dir, 'r', encoding='utf-8') as file:
        raw = json.load(file)
        l = len(raw)
        train = raw[:int(l * 0.95)]
        test = raw[int(l * 0.95):]
    return train, test


def bbox_to_coords(t):
    """Changes format of bounding boxes from [x, y, width, height] to ([x1, y1], [x2, y2])."""

    width = bbox_attr(t, 2)
    x = bbox_attr(t, 0)
    x1 = x - width / 2.0
    x2 = x + width / 2.0

    height = bbox_attr(t, 3)
    y = bbox_attr(t, 1)
    y1 = y - height / 2.0
    y2 = y + height / 2.0

    return torch.stack((x1, y1), dim=4), torch.stack((x2, y2), dim=4)


def get_iou(p, a):
    p_tl, p_br = bbox_to_coords(p)  # (batch, S, S, B, 2)
    a_tl, a_br = bbox_to_coords(a)

    # Largest top-left corner and smallest bottom-right corner give the intersection
    coords_join_size = (-1, -1, -1, config.B, config.B, 2)
    tl = torch.max(
        p_tl.unsqueeze(4).expand(coords_join_size),  # (batch, S, S, B, 1, 2) -> (batch, S, S, B, B, 2)
        a_tl.unsqueeze(3).expand(coords_join_size)  # (batch, S, S, 1, B, 2) -> (batch, S, S, B, B, 2)
    )
    br = torch.min(
        p_br.unsqueeze(4).expand(coords_join_size),
        a_br.unsqueeze(3).expand(coords_join_size)
    )

    intersection_sides = torch.clamp(br - tl, min=0.0)
    intersection = intersection_sides[..., 0] \
                   * intersection_sides[..., 1]  # (batch, S, S, B, B)

    p_area = bbox_attr(p, 2) * bbox_attr(p, 3)  # (batch, S, S, B)
    p_area = p_area.unsqueeze(4).expand_as(intersection)  # (batch, S, S, B, 1) -> (batch, S, S, B, B)

    a_area = bbox_attr(a, 2) * bbox_attr(a, 3)  # (batch, S, S, B)
    a_area = a_area.unsqueeze(3).expand_as(intersection)  # (batch, S, S, 1, B) -> (batch, S, S, B, B)

    union = p_area + a_area - intersection

    # Catch division-by-zero
    zero_unions = (union == 0.0)
    union[zero_unions] = 1e-7
    intersection[zero_unions] = 0.0

    return intersection / union


'''
겹치는 부분 / 전체 박스인데! 귀찮으니까 센터끼지 차이로 보자
sqrt(pred x,y - label x,y)
20번째 21번째가 center 좌표
'''


def intersection_over_union(boxes_preds, boxes_labels):
    box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
    box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
    box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
    box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def bbox_attr(data, i):
    attr_start = 15 + i
    return data[..., attr_start::5]


def non_max_suppression(bboxes, conf_th, iou_threshold, class_list):
    # 박스 필터링 및 클래스 저장
    filtered_boxes = [(box, cls) for box, cls in zip(bboxes, class_list) if box[1] > conf_th]
    # 신뢰도로 정렬
    filtered_boxes.sort(key=lambda x: x[0][1], reverse=True)

    selected_boxes = []
    selected_classes = []

    while filtered_boxes:
        current_box, current_class = filtered_boxes.pop(0)
        selected_boxes.append(current_box)
        selected_classes.append(current_class)
        filtered_boxes = [(box, cls) for box, cls in filtered_boxes
                          if intersection_over_union(torch.Tensor(current_box[2:]),
                                                     torch.Tensor(box[2:])) <= iou_threshold and max(cls) > 0.0 and max(
                cls < 1.0)]

    return selected_boxes, selected_classes


if __name__ == "__main__":
    print()
    get_json_data()
