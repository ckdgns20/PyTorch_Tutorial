import torch

x=torch.ones(5)
y=torch.zeros(3)
w=torch.randn(5,3,requires_grad=True)
b=torch.randn(3,requires_grad=True) #requires_grad 는 자동 미분(손실 함수의 변화도를 위해) 사용(*x.requires_grad(True) 메소드를 사용해 나중에 처리 가능 )
z=torch.matmul(x,w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z,y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

loss.backward() #손실함수의 기울기를 구하기 위해 호출 해야함(requires_grad=True로 설정되어 있기 때문에, loss.backward() 호출 후에 w.grad 와 b.grad 속성에 기울기가 저장됨)
print(w.grad)
print(b.grad)

z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad(): #변화도 추적 멈추기 방법1
    z = torch.matmul(x, w)+b
print(z.requires_grad)

z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad) #변화도 추적 멈추기 방법2
"""
변화도 추적을 멈춰야 하는 이유:
신경망의 일부 매개변수를 고정된 매개변수(frozen parameter)로 표시
변화도를 추적하지 않는 텐서의 연산이 더 효율적이기 때문에, 순전파 단계만 수행할 때 연산 속도가 향상
"""