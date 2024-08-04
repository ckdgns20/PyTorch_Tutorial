import torch
import numpy as np # np 라는 이름으로 numpy 라이브러리를 불러옴.

"""
tensor는 배열과 행렬과 유사한 데이터 구조다.
Numpy의 ndarray 와 비슷하지만 GPU를 사용해 연산을 빠르게 실행시킬 수 있다.
"""
data = [[1,2], [3,4]]
x_data = torch.tensor(data) 
# 데이터를 직접 사용해서 Tensor 생성하는 방법
"""
np_array = np.array(data)
x_np = torch.from_numpy(np_array) -> Numpy를 이용해 Tensor 생성 방법
"""

if torch.cuda.is_available(): #cuda(gpu)가 사용가능한지 
    print("GPU is available.")
    x_data = x_data.to("cuda")
# Tensor 이름.to("cuda") -> 저 이름을 가진 Tensor를 GPU를 사용하겠다.
x_ones = torch.ones_like(x_data) # torch.ones_like(Tensor 이름) -> 저 이름을 가진 Tensor의 형태를 가져와서 원소들을 1로 바꿔서 저장
x_rand = torch.rand_like(x_data, dtype=torch.float) # torch.rand_like(Tensor 이름, dtype = torch.float) -> 저 이름을 가진 Tensor의 형태를 가져와서 원소들을 0-1 사이의 수를 랜덤으로 입력하여 저장

print(f"Origin Tensor: \n {x_data}\n")
print(f"Ones Tensor: \n {x_ones}\n")
print(f"Random Tensor: \n {x_rand}\n")
#f-string -> 문자열과 데이터를 같이 출력시에 사용.
shape = (2,3)
# shape -> Tensor의 차원을 지정 (여기서는 2x3 Tensor 생성)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape) # torch.zeros()는 원소들을 0으로 바꿔서 입력
# torch.rand는 형태를 직접 지정, torch.rand_like는 기존 텐서의 형태를 기반으로 새 텐서를 생성

print(f"Random Tensor(with shape):\n {rand_tensor}\n")
print(f"Ones Tensor(with shape):\n {ones_tensor}\n")
print(f"Zeros Tensor(with shape):\n {zeros_tensor}\n")

print(f"Shape of x_data: \n {x_data.shape}\n") # 데이터명.shape ->Tensor의 차원 표시
print(f"Datatype of x_data: \n {x_data.dtype}\n") # 데이터명.dtype -> Tensor의 데이터 타입 표시
print(f"Device x_data is stored on: \n {x_data.device}\n") # 데이터명.device -> Tensor가 저장된 하드웨어 표시

tensor = torch.ones(4,4)
print(f"First row: {tensor[0]}") #tensor[0] Tensor의 첫번째 행 
print(f"First column: {tensor[:,0]}")#tensor[:,0]은 Tensor의 첫번째 열
print(f"First row: {tensor[...,-1]}") #...은 모든 차원이라는 의미 , -1 은 마지막 열

tensor[:,1] = 0 #Tensor의 두번째 열을 0으로 바꿈

print(f"2행을 0으로 바꾼 Tensor:\n {tensor}\n")

tensor[0] = 0 #Tensor의 첫번쨰 행을 0으로 바꿈

print(f"1열을 0으로 바꾼 Tensor:\n {tensor}\n")

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(f"t1 Tensor:\n{t1}")

t2 = torch.cat([tensor,tensor,tensor],dim=0)
print(f"t2 Tensor:\n{t2}\n")
#torch.cat 함수 : 각 Tensor들을 지정한 축(행,열)을 기준으로 이어붙임 (dim=0 은 행 기준, dim=1 은 열 기준)
#torch.stack 함수 : torch.cat과 달리 새로운 차원에서 이어붙여 새로운 Tensor 생성
print(f"Tensor.T:\n {tensor.T}\n")
# Tensor이름.T -> 저 이름을 가진 Tensor의 전치 행렬
y1 = tensor @ tensor.T 
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor,tensor.T, out=y3)
"""
@ 이나 Tensor이름.matmul(진행하고자하는 Tensor 이름)은 행렬 곱셈 수행
torch.matmul(Tensor1,Tensor2, out=Tensor3) 도 가능
"""
print(f"y1 tensor:\n{y1}\n")
print(f"y2 tensor:\n{y2}\n")
print(f"y3 tensor:\n{y3}\n")

z1 = tensor * tensor.T
z2 = tensor.mul(tensor.T)
z3 = torch.rand_like(tensor)
torch.mul(tensor,tensor.T, out=z3)
"""
* 이나 Tensor이름.mul(진행하고자하는 Tensor 이름)은 원소별 곱셈 수행
torch.mul(Tensor1,Tensor2, out=Tensor3) 도 가능
"""
print(f"z1 tensor:\n{z1}\n")
print(f"z2 tensor:\n{z2}\n")
print(f"z3 tensor:\n{z3}\n")

agg = tensor.sum() # Tensor 이름.sum()으로 단일 요소 Tensor 생성
agg_item = agg.item() #단일 요소 Tensor.item()으로 Python 숫자 값으로 변환
print(agg_item, type(agg_item))

t3 = tensor.add_(5) # 제자리 연산(원래 Tensor에 괄호 안 값을 더하여 저장)
print(f"t3 Tensor:\n{t3}\n")

t = torch.ones(5) 
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
# Numpy와 PyTorch의 Tensor은 동일한 메모리 위치 공유하여 하나 바꾸면 다른 것도 변경.
n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
# 이부분은 Numpy로 생성한 배열을 PyTorch로 공유