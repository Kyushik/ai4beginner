import requests
from torchvision import datasets, transforms
import random
import io
import matplotlib.pyplot as plt

# MNIST 테스트셋 로딩
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

# 랜덤 이미지 선택
idx = random.randint(0, len(test_dataset) - 1)
image, label = test_dataset[idx]
image_pil = transforms.ToPILImage()(image)

# 이미지 시각화
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"True Label: {label}")
plt.axis('off')
plt.show()

# 이미지 바이트로 인코딩
buffer = io.BytesIO()
image_pil.save(buffer, format='PNG')
buffer.seek(0)

# 서버로 요청
files = {'file': ('mnist.png', buffer, 'image/png')}
response = requests.post("http://localhost:9000/predict", files=files)

# 결과 출력
if response.ok:
    data = response.json()
    print("✅ 실제 라벨:", label)
    print("🔢 예측 결과:", data["predicted_label"])
    print(f"📈 확률: {data['confidence']:.4f}")
else:
    print("❌ 요청 실패:", response.status_code)
