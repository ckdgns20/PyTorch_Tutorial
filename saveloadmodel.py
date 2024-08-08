import torch
import torchvision.models as models

model = models.vgg16(weights='IMAGENET1K_V1')
#vgg16 네트워크를 로드 / vgg16이 ImageNet 데이터셋으로 사전 훈련된 가중치를 갖도록 함.
torch.save(model.state_dict(), 'model_weights.pth')
# model_weight.pth에 상태 사전을 저장
#torch.save(model, 'model.pth') 로 모델의 형태를 포함하여 저장하고 불러올수있음
model = models.vgg16() # 여기서는 ``weights`` 를 지정하지 않았으므로, 학습되지 않은 모델을 생성
model.load_state_dict(torch.load('model_weights.pth', weights_only=True)) #저장된 모델의 가중치를 로드하는 과정
# model = torch.load('model.pth')로 모델의 형태를 포함하여 불러오기 가능
model.eval() #eval() 메서드는 모델을 평가 모드로 설정
#이 접근 방식은 Python pickle 모듈을 사용하여 모델을 직렬화(serialize)하므로, 모델을 불러올 때 실제 클래스 정의(definition)를 적용(rely on)합니다.
#그래서 (torch.load('model_weights.pth', weights_only=True)) 로 가중치만 로드

