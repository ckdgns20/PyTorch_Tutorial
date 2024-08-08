import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
"""
원-핫 인코딩은 범주형 데이터(이미지 데이터)를 이진 벡터로 변환
"""
ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0,torch.tensor(y),value=1))
    # Lambda(lambda y: ...) -> 데이터 변환을 정의하는데 사용
    # .scatter_(0, torch.tensor(y), value =1) 여기서 0은 차원(dim),torch.tensor(y)는 텐서로 변환,value=1은 할당할 값
)