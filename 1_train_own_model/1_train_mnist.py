import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 데이터셋 로딩
transform = transforms.ToTensor()
train_loader = DataLoader(datasets.MNIST(root='./data', train=True, download=True, transform=transform), batch_size=128, shuffle=True)
test_loader = DataLoader(datasets.MNIST(root='./data', train=False, download=True, transform=transform), batch_size=1000)

# 모델 정의
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 학습
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

print("학습 시작!!")

for epoch in range(10):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
    print(f"Epoch {epoch}, Loss 값: {loss.item():.4f}")

# 테스트
model.eval()
correct = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += pred.eq(y).sum().item()
print(f"테스트 정확도: {correct / len(test_loader.dataset):.4f}")

# 모델 저장
save_path = 'mnist_model.pth'
torch.save(model.state_dict(), save_path)
print(f"모델 저장 완료! 경로: {save_path}")