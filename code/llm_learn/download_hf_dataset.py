from datasets import load_dataset

# 指定本地缓存目录（建议你自己设一个路径）
cache_dir = "/share_data/users/like/huggingface_cache"

# 下载 GSM8K（main split）
dataset = load_dataset(
    "gsm8k",
    "main",
    cache_dir=cache_dir
)

# 可选：打印确认
print(dataset)
