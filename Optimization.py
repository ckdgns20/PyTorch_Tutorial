import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
"""
아래의 Hyperparameters은 모델 학습 과정을 조정하는 설정값으로 학습 전에 설정
-Epoch : 전체 훈련 데이터셋을 모델에 통과시키는 횟수(몇 번 학습시킬지)
 -> 너무 많은 Epoch는 과적합 발생
-Batch Size : 모델의 파라미터를 업데이트하기 전에 한번에 처리하는 데이터 샘플 수
 ->작은 크기는 안정적, 계산 자원 사용 비효율적 / 큰 크기는 학습 속도 증가, 덜 정밀할 가능성
-Learning Rate : 모델의 파라미터를 조정할 때 사용하는 단계 크기
 -> 작을수록 더 안정적이고 정확하지만 수렴 시간 김 / 클수록 학습 속도 빠름, 성능 저하
"""
learning_rate = 1e-3
batch_size = 64
epochs = 5
# 하나의 Epoch는 train loop 와 test loop로 구성
"""
train loop에서 일어나는 몇가지 개념
1. 획득한 결과와 실제 값 사이의 틀린 정도를 측정하여 최소하하는 과정(손실 함수 사용)
 *손실함수로 nn.MSELoss(평균 제곱 오차),nn.NLLLoss(음의 로그 우도), nn.CrossEntropyLoss(nn.LogSoftmax와 nn.NLLLoss 합침)
2. Optimizer : 최적화는 각 학습 단계에서 모델의 오류를 줄이기 위해 모델 매개변수를 조정하는 과정
 모든 최적화 절차는 optimizer 객체에 캡슐화
 *여기서는 SGD 옵티마이저를 사용하고 있으며, PyTorch에는 ADAM이나 RMSProp과 같은 다른 종류의 모델과 데이터에서 더 잘 동작하는 다양한 옵티마이저가 있음
"""

def train_loop(dataloader, model, loss_fn, optimizer): #loss_fn 이 손실함수
    size = len(dataloader.dataset)
    # 모델을 학습(train) 모드로 설정 - 배치 정규화(Batch Normalization) 및 드롭아웃(Dropout) 레이어들에 중요
    # 이 예시에서는 없어도 되지만, 모범 사례를 위해 추가해두었습니다.
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # 예측(prediction)과 손실(loss) 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # 모델을 평가(eval) 모드로 설정합니다 - 배치 정규화(Batch Normalization) 및 드롭아웃(Dropout) 레이어들에 중요합니다.
    # 이 예시에서는 없어도 되지만, 모범 사례를 위해 추가해두었습니다.
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # torch.no_grad()를 사용하여 테스트 시 변화도(gradient)를 계산하지 않도록 합니다.
    # 이는 requires_grad=True로 설정된 텐서들의 불필요한 변화도 연산 및 메모리 사용량 또한 줄여줍니다.
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

loss_fn = nn.CrossEntropyLoss() #손실함수 부분
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #optimizer 부분

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")