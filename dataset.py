import cv2
import numpy as np
import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
from utils import for_test_decode_label, load_and_split_json, ParseJson, non_max_suppression, draw
from PIL import Image as image
import functools
import os
from PIL import ImageFile
import json
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True
#test
class Dataset(Dataset):
    def __init__(self, json_dir, cache_size=40000):
        self.json_dir = json_dir
        self.cache_size = cache_size


    '''@functools.lru_cache(maxsize=0)  # 캐시 크기를 조절하여 메모리 관리
    def load_data(self, idx):
        img, label = ParseJson(self.json_dir[idx])
        return img, label

    def __getitem__(self, idx):
        img, label = self.load_data(idx)
        return torch.Tensor(img).float().to('cuda'), torch.Tensor(label).to('cuda')'''

    def __getitem__(self, idx):
        x = np.load(self.json_dir[idx]["x"])
        y = np.load(self.json_dir[idx]["y"])
        return torch.Tensor(x).float().to('cuda'), torch.Tensor(y).to('cuda')

    def __len__(self):
        return len(self.json_dir)


save_dir = "/media/unsi/media/tmp"
output_json_path = "./file_paths.json"

# 경로 저장을 위한 리스트 초기화
file_paths = []


def save_all_files(data_loader, prefix):
    for i, (x, y) in enumerate(tqdm(data_loader, desc=f"Saving {prefix} files")):
        # 개별 파일 이름 설정
        x_path = os.path.join(save_dir, f"{prefix}_x{i + 1}.npy")
        y_path = os.path.join(save_dir, f"{prefix}_y{i + 1}.npy")

        # 파일 저장
        np.save(x_path, x.cpu().numpy())  # 텐서를 numpy 배열로 변환하여 저장
        np.save(y_path, y.cpu().numpy())

        # 경로 저장
        file_paths.append({"x": x_path, "y": y_path})


if __name__ == "__main__":
    # JSON 파일을 로드하여 train/test 데이터를 분할하는 함수
    '''train_json, test_json = load_and_split_json()

    # Dataset 및 DataLoader 초기화
    ds = Dataset(test_json)
    dl = DataLoader(ds, batch_size=32, shuffle=True)

    # DataLoader에서 데이터를 저장
    save_all_files(dl, "data")

    # 파일 경로 정보를 JSON으로 저장
    with open(output_json_path, "w") as f:
        json.dump(file_paths, f, indent=4)'''


