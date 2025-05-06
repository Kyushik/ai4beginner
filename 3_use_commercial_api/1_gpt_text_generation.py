import openai
from dotenv import load_dotenv

# .env 파일을 사용하여 환경변수 불러오기
load_dotenv()

# GPT에 응답 요청 
response = openai.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "서울 하루 여행 일정 짜줘"},
    ],
)

# 결과 출력
print(response.choices[0].message.content)