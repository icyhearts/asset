from huggingface_hub import list_repo_files, get_hf_file_metadata, HfFileMetadata

repo_id = "deepseek-ai/DeepSeek-V4-Flash"  # 模型仓库ID
# 如果需要 gated model，请先登录 huggingface-cli login

# 列出仓库中的所有文件
all_files = list_repo_files(repo_id)

# 筛选出 .safetensors 文件
safetensor_files = [f for f in all_files if f.endswith('.safetensors')]

# 获取每个 safetensor 文件的元数据，其中就包含 sha256
for file in safetensor_files:
    metadata: HfFileMetadata = get_hf_file_metadata(repo_id, path=file)
    # metadata.lfs 是一个 LFS 对象，包含文件的 sha256 值
    if metadata.lfs is not None:
        print(f"文件: {file}, SHA256: {metadata.lfs.sha256}")
