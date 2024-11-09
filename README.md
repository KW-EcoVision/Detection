# Detection
## 진행상황 
### 1. 데이터 파싱 
![img_2.png](img_2.png)
- 가구와 같이 전문적 수거가 필요한 항목은 train set에서 제거한다.
- zero shot detection이 필요할 경우 hugging face에서 owl-vit를 로드한다.

### 2. train
- vram
- time
- loss optim
- config
- so.on

### 3. inference
- local cam, test cam, app cam의 rgb 분포를 최대한 동일시 한다.
- iou th, confidential th를 최적화 한다.

### done
![모델출력1](https://github.com/user-attachments/assets/57492148-bd50-47a0-abee-ed6d4df5462d)
![모델출력2](https://github.com/user-attachments/assets/751e6a29-3959-4c98-8828-33d6554e38e1)
![모델출력3](https://github.com/user-attachments/assets/5d7b83f1-89d2-44bc-a512-33d472c11218)
![모델출력4](https://github.com/user-attachments/assets/bb629bfd-92dd-4ff0-b7d8-098df4779089)
![모델출력5](https://github.com/user-attachments/assets/d290249c-31b2-49f1-b0ab-36afa498ab02)
