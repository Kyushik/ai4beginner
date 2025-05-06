import base64
from openai import OpenAI
from dotenv import load_dotenv

# .env 파일을 사용하여 환경변수 불러오기
load_dotenv()

client = OpenAI()

img = client.images.generate(
    model="gpt-image-1",
    prompt="거대한 외계의 바닷가 끝자락에 홀로 서 있는 우주비행사. 해안에는 생체 발광하는 파도가 잔잔히 밀려들고, 두 개의 달이 수면 위에 비친다. 머리 위엔 거대한 해파리들이 조용히 떠다니며, 몽환적이고 비현실적인 분위기를 자아낸다.",
    n=1,
    size="1024x1024"
)

image_bytes = base64.b64decode(img.data[0].b64_json)
with open("gpt_img_gen_output.png", "wb") as f:
    f.write(image_bytes)
