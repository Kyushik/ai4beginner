from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import io

app = FastAPI()

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

# 모델 로딩
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
model.load_state_dict(torch.load('mnist_model.pth', map_location=device))
model.eval()

# 변환 정의
transform = transforms.Compose([
    transforms.ToTensor(),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert('L')  # 흑백 변환
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        probs_np = probs.squeeze().cpu().numpy().tolist()

    return JSONResponse({
        "predicted_label": pred,
        "confidence": round(probs_np[pred], 4),
        "probabilities": [round(p, 4) for p in probs_np]
    })
