from transformers import AutoTokenizer, LlamaForCausalLM

model_name = "/mnt/yrfs/llm_weights/Meta-Llama-3.1-8B/"

from transformers import AutoTokenizer, LlamaForCausalLM
import torch

# 初始化时设置padding_side和pad_token
#model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # 关键修正点

model = LlamaForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="eager"
)

# 批处理输入
batch_prompts = [
    "Hello, my name is",
    "Hi, do you know who is the president of the United States",
    "The capital of France is",
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
generation_config = {
    "max_new_tokens": 128,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "pad_token_id": tokenizer.pad_token_id,
}

# 执行推理
with torch.no_grad():
    outputs = model.generate(**batch_inputs, **generation_config)

# 解码结果
responses = [tokenizer.decode(out, skip_special_tokens=True)
             for out in outputs]
print(f"response:{responses}")
