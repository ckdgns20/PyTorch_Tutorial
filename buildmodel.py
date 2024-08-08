import os
import torch
from torch import nn #torch.nn은 신경망의 구성 요소 포함
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class NeuralNetwork(nn.Module): #신경망 모델을 nn.module의 하위클래스로 정의
    def __init__(self):
        super().__init__() #신경망 초기화
        self.flatten = nn.Flatten() #nn.Flatten()은 입력 텐서를 평탄화(1차원으로 변환) ->28*28 이미지를 선형 레이어에 입력하기 위해 784 크기의 벡터로 변환
        self.linear_relu_stack = nn.Sequential( #nn.Sequential()은 레이어를 정의하고 데이터가 순서대로 통과하도록 함
            nn.Linear(28*28, 512), #nn.Linear() 선형레이어
            nn.ReLU(),             #nn.ReLU() 비선형레이어
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device) #NeuralNetwork 클래스의 인스턴스 생성, .to()로 계산을 ()안 디바이스에서 수행
print(model) #모델의 구조 출력

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits) #nn.Softmax로 확률 값으로 변환
y_pred = pred_probab.argmax(1) #pred_probab.argmax(1)은 소프트맥스 출력에서 가장 높은 확률을 가지는 클래스의 인덱스(모델의 최종 예측 클래스)를 찾음
print(f"Predicted class: {y_pred}")

print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")