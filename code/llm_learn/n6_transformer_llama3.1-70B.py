from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "/data/like//hf-models/llama3.1-70B-strip-layers/"

import torch

# 初始化时设置padding_side和pad_token
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # 关键修正点

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="eager"
)

# 批处理输入
batch_prompts = [
    "Please introduce MoE model",
    "Please compare MoE and Dense model",
    "What is your name",
]

# 编码配置
batch_inputs = tokenizer(
    batch_prompts,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
).to("cuda")

# 生成配置需与编码设置匹配
    #"num_beams": 3,
    #"num_return_sequences": 2,
    #"top_p": 0.9,
    #"early_stopping": True,
generation_config = {
    "max_new_tokens": 20,
    "do_sample": True,
    "temperature": 0.7,
    "pad_token_id": tokenizer.pad_token_id,
}

# 执行推理
with torch.no_grad():
    outputs = model.generate(**batch_inputs, **generation_config)

# 解码结果
print("len out:{}".format(len(outputs)))
responses = [tokenizer.decode(out, skip_special_tokens=True)
             for out in outputs]
for out in responses:
    print(f"out:{out}")
