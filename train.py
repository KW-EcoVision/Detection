import os
import random

import cv2
import numpy as np
import cv2 as cv
import torch
from model import YOLO
from testmodel import Yolov1 as YOLO_DARK
from utils import decoding_label, non_max_suppression, draw, load_and_split_json
from dataset import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from field import field
import matplotlib.pyplot as plt

BACK = 'RES'
torch.manual_seed(100)
torch.cuda.manual_seed(100)
np.random.seed(100)
random.seed(100)
LEARNING_RATE = 2e-4
DEVICE = "cuda"
BATCH_SIZE = 64
WEIGHT_DECAY = 0
NUM_WORKERS = 10
PIN_MEMORY = False
EPOCHS = 500
LOAD_MODEL = False
MODEL_SAVE_PATH = f'./weights/{BACK}.pt'
LOAD_MODEL_FILE = f'./weights/{BACK}.pt'
COMMON_PATH = '/media/sien/DATA/DATA/dataset/voc_data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/for_detection'
IMG_DIR = COMMON_PATH + '/JPEGImages/'
LABEL_DIR = COMMON_PATH + '/Annotations/'
MODE = 'train'

train_json, test_json = load_and_split_json()
train_ds = Dataset(train_json)
test_ds = Dataset(train_json)

if BACK == 'VGG':
    VGG16 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16')
    for i in range(len(VGG16.features[:-1])):
        if type(VGG16.features[i]) == type(torch.nn.Conv2d(32, 32, 3)):
            VGG16.features[i].weight.requires_grad = False
            VGG16.features[i].bias.requires_grad = False
            VGG16.features[i].padding = 1
    model = YOLO(VGG16.features[:-1], False)
elif BACK == 'RES':
    ResNet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='DEFAULT')
    modules = list(ResNet50.children())[:-2]
    backbone = torch.nn.Sequential(*modules)
    for param in backbone.parameters():
        param.requires_grad = False
    model = YOLO(backbone[:-1], True)
elif BACK == 'DARK':
    model = YOLO_DARK(split_size=7, num_boxes=2, num_classes=20)
    print(model)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
data_stream = torch.cuda.Stream()


def train_step(train_loader, model, optimizer, loss_fn):
    model.train()
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for x, y in loop:
        try:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)

            # 모델 출력 차원 확인 및 조정
            if len(pred.shape) == 2:
                pred = pred.view(-1, 7, 7, 25)

            # 손실 계산 및 역전파
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 진행 상태 업데이트
            loop.set_postfix(loss=loss.item())
            mean_loss.append(loss.item())

        except Exception as e:
            print(f"Warning: Exception encountered during training - {e}")
            print("Skipping this batch and continuing with the next.")

    # 에포크 손실 출력
    if mean_loss:
        epoch_loss = sum(mean_loss) / len(mean_loss)
        print(f'one epoch loss : {epoch_loss}')
    else:
        epoch_loss = np.inf  # 모든 배치에서 예외 발생 시

    return epoch_loss


def validation_step(validation_loader, model, loss_fn):
    model.eval()
    with torch.no_grad():
        loop = tqdm(validation_loader, leave=True)
        loss_store = []
        for x, y in loop:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss_store.append(loss.item())
            loop.set_postfix(loss=loss.item())
        print(f'valdation loss : {sum(loss_store) / len(loss_store)}')
    return sum(loss_store) / len(loss_store)


def test_step(train_loader, model):
    model.eval()
    pred_store = []
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x)
        if len(pred.shape) == 2:
            pred = pred.view(-1, 7, 7, 25)
        boxes, class_list = decoding_label(np.array(pred.cpu().detach().squeeze(0)))
        boxes, class_list = non_max_suppression(boxes, 0.2, 0.5, class_list)
        img = torch.permute(x.squeeze(0), (1, 2, 0))
        img = np.array(img.cpu().detach())
        img = (img * 224).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = draw(boxes, img, class_list)
        pred_store.append(img)


def load_model(model, model_path, device):
    """모델의 가중치를 로드하고 학습 모드로 전환"""
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.train()
    return model


if __name__ == "__main__":
    from myloss import YoloLoss as myloss

    loss_fn = myloss()
    # =loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)

    if LOAD_MODEL:
        model = load_model(model, f'./weights/{BACK}.pt', DEVICE)
    else:
        model.to(DEVICE)

    best_loss = np.inf

    for epoch in range(1, EPOCHS + 1):
        print(f'Epoch [{epoch}/{EPOCHS}] 시작')

        loss = train_step(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn
        )
        print(f'훈련 손실: {loss:.4f}')

        if epoch % 5 == 0:
            print('검증 시작')
            val_loss = validation_step(
                validation_loader=test_loader,
                model=model,
                loss_fn=loss_fn
            )
            print(f'검증 손실: {val_loss:.4f}')

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f'최상의 검증 손실 갱신: {val_loss:.4f} - 모델 저장 완료')

    print("모델 학습 완료")

    if MODE == 'inference':
        std = torch.load(f'weights/{BACK}.pt')
        model.load_state_dict(std)
        model = model.to('cuda')
        model.eval()
        test_step(test_loader, model)
    if MODE == 'field':
        std = torch.load(f'weights/{BACK}.pt')
        model.load_state_dict(std)
        model = model.to('cpu')
        model.eval()
        field(model)
