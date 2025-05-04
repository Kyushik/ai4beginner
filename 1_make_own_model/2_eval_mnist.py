import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random

# 모델 정의 (저장 당시와 동일하게)
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

# 모델 불러오기
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
model.load_state_dict(torch.load('mnist_model.pth', map_location=device))
model.eval()

# 테스트 데이터셋 로딩
transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 1개 샘플 무작위 선택
idx = random.randint(0, len(test_dataset) - 1)
image, label = test_dataset[idx]

# 이미지 시각화
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"True Label: {label}")
plt.axis('off')
plt.show()

# 모델 추론
image_tensor = image.unsqueeze(0).to(device)  # (1, 1, 28, 28)
with torch.no_grad():
    logits = model(image_tensor)
    probs = F.softmax(logits, dim=1)
    pred_class = probs.argmax(dim=1).item()

# 출력
probs_np = probs.cpu().numpy().squeeze()

print(f"예측 결과: {pred_class} (확률: {probs_np[pred_class]:.4f})")