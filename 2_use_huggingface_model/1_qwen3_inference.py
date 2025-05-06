from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"

# 토크나이저와 모델 불러오기
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# 모델 입력 준비
prompt = "안녕!!"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False  # 사고 모드와 비사고 모드를 전환 (기본값은 True)
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 텍스트 생성 수행
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# 생성된 토큰을 문자열로 디코딩
content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

print("출력:", content)