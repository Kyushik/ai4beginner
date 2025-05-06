from diffusers import SanaSprintPipeline
import torch
import matplotlib.pyplot as plt

# 모델 로드
pipeline = SanaSprintPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers",
    torch_dtype=torch.bfloat16
)

# 프롬프트 생성
prompt = "A sports car is driving in the forest"

# 이미지 생성
image = pipeline(prompt=prompt, num_inference_steps=2).images[0]

# 이미지 출력
plt.imshow(image)
plt.axis('off')
plt.title("Generated Image")
plt.show()
