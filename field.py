from flask import Flask, request, jsonify
import torch
from PIL import Image
import io
from model import YOLO  # YOLO 모델이 정의된 파일 임포트
from utils import decoding_label, non_max_suppression, draw
import torchvision

app = Flask(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACK = 'RES'
MODEL_PATH = f'./weights/{BACK}.pt'


def load_yolo_model():
    ResNet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='DEFAULT')
    modules = list(ResNet50.children())[:-2]
    backbone = torch.nn.Sequential(*modules)
    for param in backbone.parameters():
        param.requires_grad = False
    model = YOLO(backbone[:-1], True)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


model = load_yolo_model()


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')

    img = img.resize((224, 224))
    img = torch.tensor([torchvision.transforms.ToTensor()(img)]).to(DEVICE)

    # 예측 수행
    with torch.no_grad():
        pred = model(img)
        if len(pred.shape) == 2:
            pred = pred.view(-1, 7, 7, 25)

        boxes, class_list = decoding_label(pred.cpu().squeeze(0).numpy())
        boxes, class_list = non_max_suppression(boxes, 0.2, 0.5, class_list)

    # 예측 결과 반환
    results = []
    for box, cls in zip(boxes, class_list):
        results.append({
            'class': cls,
            'box': [int(b) for b in box]
        })

    return jsonify({'predictions': results})


'''
import requests

url = 'http://localhost:5000/predict'
files = {'image': open('test_image.jpg', 'rb')}
response = requests.post(url, files=files)

print(response.json())

'''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
