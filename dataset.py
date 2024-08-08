import torch 
from torch.utils.data import Dataset # PyTorch 데이터셋의 기본 클래스 제공
from torchvision import datasets #torchvision은 여러개의 데이터셋과 이미지변환 기능 제공 , datasets 모듈은 표준 데이터셋
from torchvision.transforms import ToTensor #ToTensor 로 이미지 데이터 -> PyTorch Tensor로 변환
import matplotlib.pyplot as plt #데이터 시각화 라이브러리

training_data = datasets.FashionMNIST(
    root = "data", #데이터 저장 경로 지정
    train = True, #train = True 면 학습용, False 면 테스트용 
    download = True, 
    transform=ToTensor() #이미지를 텐서로 변환
)

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform=ToTensor()
)

labels_map = {  #labels_map 은 이름을 매핑하는 딕셔너리
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1): # range(start, stop) 에서 stop에 해당하는값은 포함되지 않는다.
    sample_idx = torch.randint(low=0, high=len(training_data), size=(1,)).item()
    sample_idx = int(sample_idx)
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

import os  #운영체제와 상호작용(저장된 데이터의 주소를 가져오거나 이럴 때 사용)
import pandas as pd #pandas는 엑셀파일 읽고 쓸때 사용
from torchvision.io import read_image

"""
사용자 정의 데이터셋은 __init__(초기화 메서드),__len__(데이터 크기 반환 메서드),__getitem__(이미지를 텐서로 변환시키고 변형시키는 메서드)
로 구성됨
"""

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = str(self.img_labels.iloc[idx, 0])
        img_dir = str(self.img_dir)
        img_path = os.path.join(img_dir, img_name)
        print(f"img_path: {img_path}")
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
          



from torch.utils.data import DataLoader #DataLoader로 데이터셋을 배치 단위로 불러오고 순회

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze() #.squeeze()는 불필요한 차원 제거
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")