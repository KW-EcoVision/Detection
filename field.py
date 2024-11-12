from flask import Flask, request, jsonify
import torch
from PIL import Image
from model import YOLO  # YOLO 모델은 사용자 정의 모델을 가정
from utils import decoding_label, non_max_suppression, draw
import torchvision
import numpy as np
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
app = Flask(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACK = 'RES'
MODEL_PATH = f'./weights/{BACK}.pt'

# YOLO 모델 로드 함수
def load_yolo_model():
    ResNet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='DEFAULT')
    modules = list(ResNet50.children())[:-2]
    backbone = torch.nn.Sequential(*modules)
    for param in backbone.parameters():
        param.requires_grad = False
    model = YOLO(backbone[:-1], True)  # 사용자 정의 YOLO 모델로 가정
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_yolo_model()

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    img = img.resize((224, 224))
    img = torch.from_numpy(np.array(img)).to(DEVICE).permute(2, 0, 1).to(dtype=torch.float32).unsqueeze(0)/224.0

    with torch.no_grad():
        pred = model(img)
        if len(pred.shape) == 2:
            pred = pred.view(1, 7, 7, 25)

        boxes, class_list = decoding_label(pred.cpu().squeeze(0).numpy())
        boxes, class_list = non_max_suppression(boxes, 0.2, 0.5, class_list)
    max_index = np.argmax(class_list)
    result = classes_num_rev[max_index]
    return jsonify({'predictions': result})


# 서버 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
