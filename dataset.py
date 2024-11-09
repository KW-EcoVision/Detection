import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from utils import for_test_decode_label, load_and_split_json, ParseJson, non_max_suppression, draw
from PIL import Image as image
import functools


class Dataset(Dataset):
    def __init__(self, json_dir, cache_size=40000):
        self.json_dir = json_dir
        self.cache_size = cache_size

    @functools.lru_cache(maxsize=40000)  # 캐시 크기를 조절하여 메모리 관리
    def load_data(self, idx):
        img, label = ParseJson(self.json_dir[idx])
        return img, label

    def __getitem__(self, idx):
        img, label = self.load_data(idx)
        return torch.Tensor(img).float().to('cuda'), torch.Tensor(label).to('cuda')

    def __len__(self):
        return len(self.json_dir)


if __name__ == "__main__":
    train_json, test_json = load_and_split_json()
    ds = Dataset(train_json)


    while True:
        dl = DataLoader(ds, batch_size=8,shuffle=True)
        img, label = next(iter(dl))

        tmp = img[0].detach().permute(1, 2, 0).cpu().numpy()

        tmp = (tmp * 224.0).astype('uint8')

        tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR)

        boxes, class_list = for_test_decode_label(label[0].numpy())
        boxes, class_list = non_max_suppression(boxes, 0.3, 0.4, class_list)
        img = draw(boxes, tmp, class_list)

        cv2.imshow('Image Window', img)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        plt.show()
