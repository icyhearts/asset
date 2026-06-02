## 2026-05-11 all_reduce debug 修改记录

已完成两处修改：

1. sglang:
   `/data/like/package/sglang_kernel_src/python/sglang/srt/layers/linear.py`

   在 `RowParallelLinear.forward` 的
   `debug_ppl_validate_all_reduce_safetensor_dir`
   分支里，不再调用 `tensor_model_parallel_all_reduce`，改成：

   ```python
   read_quant_method_out_reduce = read_quant_method_out_tensor.contiguous().clone()
   torch.distributed.all_reduce(
       read_quant_method_out_reduce,
       op=torch.distributed.ReduceOp.SUM,
       group=get_tp_group().device_group,
   )
   ```

   结果仍保存到：
   `validate_all_reduce_rank-{my_tp_rank}.sglang.safetensors`

2. vLLM:
   `/data/like/package/vllm-for-conda-simo/vllm/model_executor/layers/linear.py`

   增加 `get_tp_group` import，并在同一个 debug validate 分支里改成：

   ```python
   read_quant_method_out_reduce = read_quant_method_out_tensor.contiguous().clone()
   torch.distributed.all_reduce(
       read_quant_method_out_reduce,
       op=torch.distributed.ReduceOp.SUM,
       group=get_tp_group().device_group,
   )
   ```

   结果仍保存到：
   `validate_all_reduce_rank-{my_tp_rank}.vllm.safetensors`

已做语法检查：

```bash
/data/like/miniconda3/envs/simo_sglang/bin/python -m py_compile /data/like/package/sglang_kernel_src/python/sglang/srt/layers/linear.py
/data/like/miniconda3/envs/simo_vllm/bin/python -m py_compile /data/like/package/vllm-for-conda-simo/vllm/model_executor/layers/linear.py
```

两个检查都通过。

### 为了让 sglang / vLLM debug all_reduce bitwise 相同

如果两个程序都是读取同一批
`/data/like/temp/qdqx_2026_05_09___16_46_17_safetensors-online/rank-*-prefix-model.layers.0.self_attn.o_proj-forwardcount-0-1.safetensors`
再做 validate all_reduce，那么关键是让两边使用同一个 TP process group、同一个 rank 到 GPU 的映射、同一个 NCCL all_reduce 算法/协议。

建议在 sglang 和 vLLM 两边命令前都加同一组环境变量：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
NCCL_ALGO=Ring \
NCCL_PROTO=Simple \
NCCL_MIN_NCHANNELS=1 \
NCCL_MAX_NCHANNELS=1 \
NCCL_NVLS_ENABLE=0 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
...
```

说明：

- `CUDA_VISIBLE_DEVICES` 两边必须完全一致，保证 TP rank 0..7 映射到同一组物理 GPU。
- `NCCL_ALGO=Ring` 固定 collective 算法，避免一边选 Ring、一边选 Tree/NVLS。
- `NCCL_PROTO=Simple` 固定 NCCL protocol，避免 LL/LL128/Simple 的选择差异。
- `NCCL_MIN_NCHANNELS=1` 和 `NCCL_MAX_NCHANNELS=1` 固定 channel 数，减少 NCCL 自动调优带来的执行差异。
- `NCCL_NVLS_ENABLE=0` 避免 H100/NVLink SHARP 路径参与选择。

可选调试变量：

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL
```

这个只用于确认两边实际选择的 NCCL 算法/协议一致，不是 bitwise 必需项。

命令行参数方面：

- sglang 保持已有 `--disable-cuda-graph --enable-deterministic-inference`。
- vLLM 保持已有 `--enforce-eager`。
- 如果还要比较 debug dump 里的正常 forward 字段 `row_parallel_all_reduce_out`，建议两边都加 `--disable-custom-all-reduce`，让正常 forward 路径也尽量走 PyTorch/NCCL，而不是框架自定义 all_reduce。仅比较本次 validate 分支生成的 `read_quant_method_out_reduce` 时，这个参数不是必需的，因为 validate 分支已经显式调用 `torch.distributed.all_reduce`。

当前环境版本检查结果：

- sglang env: PyTorch `2.9.1+cu128`, CUDA `12.8`, NCCL `(2, 27, 5)`
- vLLM env: PyTorch `2.10.0+cu128`, CUDA `12.8`, NCCL `(2, 27, 5)`

由于 NCCL 版本相同，且 validate 分支读取的是同一批 safetensors 输入，固定上面的 NCCL 环境变量后，`validate_all_reduce_rank-*.sglang.safetensors` 和 `validate_all_reduce_rank-*.vllm.safetensors` 里的 `read_quant_method_out_reduce` 应该可以做到 bitwise 相同。若重新生成输入 safetensors 后再比较，则还要先保证 `row_parallel_quant_method_out` 本身已经 bitwise 相同；否则 all_reduce 不可能把不同输入规约成相同输出。

### `CUDA_DEVICE_ORDER=PCI_BUS_ID` 的作用

`CUDA_DEVICE_ORDER=PCI_BUS_ID` 控制 CUDA runtime 枚举 GPU 的顺序。

默认情况下，CUDA 可能按自己的默认策略枚举设备；设置成 `PCI_BUS_ID` 后，CUDA 会按 GPU 的 PCI bus id 排序，这个顺序通常和 `nvidia-smi` 看到的 GPU 编号更一致、更稳定。

它的关键影响是：决定程序里的逻辑设备号 `cuda:0`, `cuda:1`, ... 分别对应哪张物理 GPU。

例如加上：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

可以让 sglang 和 vLLM 两边都按同一种稳定顺序解释 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`，从而让 TP rank 到物理 GPU 的映射更容易一致。

它不会改变 all_reduce 算法，也不会直接保证数值 bitwise 相同；它只是先保证两边 `rank-0`、`rank-1` 等逻辑 rank 绑定到同一批、同一顺序的物理 GPU。如果两边 GPU 枚举顺序不同，那么即使命令里都写 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`，实际 rank 到物理 GPU 的映射也可能不同。

如果使用 GPU UUID 写 `CUDA_VISIBLE_DEVICES=GPU-...,...`，这个变量的重要性会降低，因为 UUID 本身已经明确指定物理 GPU。

## 2026-05-11 新增 `like-useful/torch_all_reduce.py`

已新增脚本：

`/data/like/package/simo_conda_sglang/like-useful/torch_all_reduce.py`

脚本行为：

1. 使用 `torchrun --nproc_per_node=8` 启动 8 个进程。
2. 每个 rank 读取自己的输入文件：

   ```text
   /data/like/temp/qdqx_2026_05_09___16_46_17_safetensors-online/rank-{rank}-prefix-model.layers.0.self_attn.o_proj-forwardcount-0-0.safetensors
   ```

3. 从 safetensor 中读取：

   ```python
   key = "row_parallel_quant_method_out"
   ```

4. 8 个 rank 对该 tensor 执行：

   ```python
   torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
   ```

5. rank 0 保存结果到：

   ```text
   /data/like/temp/torch.distributed.all_reduce.{suffix}.safetensors
   ```

   safetensor key 为：

   ```python
   "rank_0_all_reduce"
   ```

脚本参数里 `--suffix` 是必填项。

已做检查：

```bash
/data/like/miniconda3/envs/simo_sglang/bin/python -m py_compile like-useful/torch_all_reduce.py
/data/like/miniconda3/envs/simo_sglang/bin/python like-useful/torch_all_reduce.py --help
```

两个检查都通过。

运行命令示例：

```bash
cd /data/like/package/simo_conda_sglang

CUDA_DEVICE_ORDER=PCI_BUS_ID \
NCCL_ALGO=Ring \
NCCL_PROTO=Simple \
NCCL_MIN_NCHANNELS=1 \
NCCL_MAX_NCHANNELS=1 \
NCCL_NVLS_ENABLE=0 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/data/like/miniconda3/envs/simo_sglang/bin/torchrun \
  --nproc_per_node=8 \
  like-useful/torch_all_reduce.py \
  --suffix ring_simple_1ch
```

运行完成后会生成：

```text
/data/like/temp/torch.distributed.all_reduce.ring_simple_1ch.safetensors
```

注意：这份 `torch_all_reduce.py` 是纯 `torch.distributed.all_reduce` 路径。仓库里已有的 `like-useful/vllm_all_reduce.py` 是 vLLM `tensor_model_parallel_all_reduce` 路径。

### `dist.barrier()` 的作用

`torch.distributed.barrier()` 是一个进程同步点。

含义是：所有参与当前 process group 的 rank 都必须执行到这一行，程序才会继续往下走。只要有一个 rank 还没到达 barrier，已经到达的其他 rank 就会阻塞等待。

它不做 all_reduce，不传输 tensor 数据，也不改变 tensor 内容；它只保证多个进程在代码执行进度上对齐。

在 `like-useful/torch_all_reduce.py` 里有两个 barrier：

```python
dist.barrier()
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
...
dist.barrier()
```

第一个 barrier 的作用是确保 8 个 rank 都已经完成 safetensor 读取，并且都准备好进入 `all_reduce`。这样调试日志和失败位置更清晰，也避免某个 rank 还在读文件时其他 rank 已经进入 collective。

第二个 barrier 的作用是确保 rank 0 保存输出文件之前/之后，其他 rank 不会太早退出并销毁 process group。对于 NCCL collective 调试脚本，这能减少进程退出时序导致的干扰。

严格来说，`all_reduce` 本身就是 collective，所有 rank 都必须调用；所以第一个 barrier 不是数学上必需的。但在调试脚本里保留 barrier 通常更稳，能更早暴露某个 rank 没读到文件、shape 不一致、进程卡住等问题。

## 2026-05-11 分析：框架内 `torch.distributed.all_reduce` 仍然无法 bitwise 一致的原因

现象：

- `like-useful/torch_all_reduce.py` 已经改成读取：

  ```text
  rank-{rank}-prefix-model.layers.0.self_attn.o_proj-forwardcount-0-1.safetensors
  ```

- 脱离 sglang/vLLM 框架、只用 `torchrun + torch.distributed.all_reduce` 时：

  ```bash
  CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1 NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 like-useful/torch_all_reduce.py --suffix sglang
  CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1 NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 like-useful/torch_all_reduce.py --suffix vllm
  ```

  两个输出 bitwise 一致。

实际对比结果：

```text
torch_sglang shape (8, 8192) dtype torch.bfloat16 sum_f32 -0.8750482201576233
torch_vllm   shape (8, 8192) dtype torch.bfloat16 sum_f32 -0.8750482201576233
fw_sglang_r0 shape (8, 8192) dtype torch.bfloat16 sum_f32 -0.8789826035499573
fw_vllm_r0   shape (8, 8192) dtype torch.bfloat16 sum_f32 -0.8750482201576233
```

bitwise 关系：

```text
torch_sglang vs torch_vllm: bitwise=True
fw_sglang_r0 vs fw_vllm_r0: bitwise=False
torch_sglang vs fw_vllm_r0: bitwise=True
torch_vllm   vs fw_vllm_r0: bitwise=True
torch_sglang vs fw_sglang_r0: bitwise=False
```

框架内 sglang 和 vLLM 的差异规模：

```text
fw_sglang_r0 vs fw_vllm_r0:
  max_abs_diff = 0.00048828125
  nonzero_values = 38935
  mismatched_bytes = 39441
  first mismatch index=(0, 0)
    sglang = -0.00090789794921875
    vllm   = -0.000911712646484375
```

结论：

不是 `torch.distributed.all_reduce` 在 sglang/vLLM 两个 conda 环境里天然不一致。证据是独立 `torchrun` 的两个结果 bitwise 相同，而且 vLLM 框架内 debug 输出和独立 `torchrun` 输出 bitwise 相同。

真正的问题是：sglang 框架启动时把你在 shell 里设置的 `NCCL_ALGO=Ring` 覆盖掉了。

sglang 日志里有明确证据：

```text
WARNING:sglang.srt.server_args:NCCL_ALGO is set to 'allreduce:tree' and custom all reduce is disabled for deterministic inference when TP size > 1.
```

对应代码在：

```text
/data/like/package/sglang_kernel_src/python/sglang/srt/server_args.py
```

相关逻辑：

```python
if self.enable_deterministic_inference:
    ...
    os.environ["NCCL_ALGO"] = "allreduce:tree"
    self.disable_custom_all_reduce = True
```

也就是说，sglang 命令里虽然写了：

```bash
NCCL_ALGO=Ring
```

但因为同时传了：

```bash
--enable-deterministic-inference
```

sglang 在进程内部又改成了：

```bash
NCCL_ALGO=allreduce:tree
```

vLLM 这次命令没有同样的覆盖，仍然按外部环境变量走 `NCCL_ALGO=Ring`。

因此，两个框架内的 debug 分支虽然代码里都写的是：

```python
torch.distributed.all_reduce(..., op=torch.distributed.ReduceOp.SUM, group=get_tp_group().device_group)
```

但实际 NCCL reduction topology 不一样：

- sglang: `allreduce:tree`
- vLLM: `Ring`

对于 bf16/fp 浮点加法，规约顺序不同就可能导致 bitwise 不同。浮点加法不满足严格结合律，tree 和 ring 的求和顺序不同，所以结果可能只差 1 个或几个 bf16 ulp，但 bitwise 不一致。

这也解释了为什么：

- 独立 `torch_all_reduce.py` 两边 bitwise 一致：两边都真正使用 `NCCL_ALGO=Ring`。
- vLLM 框架内结果和独立 torchrun bitwise 一致：vLLM 没覆盖 `NCCL_ALGO=Ring`。
- sglang 框架内结果不同：sglang 因 `--enable-deterministic-inference` 改成了 `NCCL_ALGO=allreduce:tree`。

建议验证/修复方式：

1. 如果目标是让框架内 debug validate 分支和 `torch_all_reduce.py` 的结果一致，先让 sglang 不覆盖 `NCCL_ALGO`。

   最直接的测试是去掉 sglang 命令里的：

   ```bash
   --enable-deterministic-inference
   ```

   然后继续保留：

   ```bash
   NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1 NCCL_NVLS_ENABLE=0
   ```

2. 如果必须保留 `--enable-deterministic-inference`，需要临时 patch sglang：

   ```python
   os.environ["NCCL_ALGO"] = "allreduce:tree"
   ```

   改成尊重外部已经设置的 `NCCL_ALGO`，例如：

   ```python
   os.environ.setdefault("NCCL_ALGO", "allreduce:tree")
   ```

   或者为了本次 debug 直接注释掉这行覆盖。这样外部 `NCCL_ALGO=Ring` 才会真正生效。

3. 另一种方向是让 vLLM 也使用和 sglang 一样的算法：

   ```bash
   NCCL_ALGO=allreduce:tree
   ```

   然后重新跑 vLLM 和独立 torchrun 对比。这样有机会让 vLLM 框架内结果靠近当前 sglang 框架内结果。但如果最终目标是和已有 `torch_all_reduce.py --suffix sglang/vllm` 的 Ring 结果对齐，还是应该优先让 sglang 不覆盖 `NCCL_ALGO=Ring`。

4. 建议在两个 `RowParallelLinear.forward` 的 debug 分支临时多打几项日志，确认运行时环境：

   ```python
   logger.info(
       f"validate all_reduce env: rank={my_tp_rank}, "
       f"device={read_quant_method_out_reduce.device}, "
       f"NCCL_ALGO={os.environ.get('NCCL_ALGO')}, "
       f"NCCL_PROTO={os.environ.get('NCCL_PROTO')}, "
       f"NCCL_MIN_NCHANNELS={os.environ.get('NCCL_MIN_NCHANNELS')}, "
       f"NCCL_MAX_NCHANNELS={os.environ.get('NCCL_MAX_NCHANNELS')}, "
       f"NCCL_NVLS_ENABLE={os.environ.get('NCCL_NVLS_ENABLE')}"
   )
   ```

   预期当前 sglang 会打印 `NCCL_ALGO=allreduce:tree`，vLLM 会打印 `NCCL_ALGO=Ring`。

最终判断：

当前不一致的主因不是输入 safetensor、不是 conda 环境，也不是 `torch.distributed.all_reduce` API 本身，而是 sglang 在 `--enable-deterministic-inference` 下覆盖了 NCCL all_reduce 算法，导致 sglang 和 vLLM 实际执行的规约顺序不同。要 bitwise 一致，必须保证两个框架进程内实际生效的 `NCCL_ALGO/NCCL_PROTO/channel/NVLS` 完全一致，尤其是不要让 sglang 把 `Ring` 改成 `allreduce:tree`。

## 2026-05-11 复核：去掉 sglang `--enable-deterministic-inference` 后的结果

这次 sglang 离线命令已经去掉：

```bash
--enable-deterministic-inference
```

sglang 新日志：

```text
/data/like/package/simo_conda_sglang/templ/offline_batch_inference-qdqx_safetensor.online-quant.log.2026_05_11___17_53_15
```

日志里 `server_args` 显示：

```text
enable_deterministic_inference=False
disable_custom_all_reduce=False
```

并且这次没有再出现之前的警告：

```text
NCCL_ALGO is set to 'allreduce:tree'
```

这说明 sglang 这次没有再把外部传入的：

```bash
NCCL_ALGO=Ring
```

覆盖成：

```bash
NCCL_ALGO=allreduce:tree
```

当前磁盘上的文件时间：

```text
2026-05-11 17:53:51 validate_all_reduce_rank-0.sglang.safetensors
2026-05-11 16:29:26 validate_all_reduce_rank-0.vllm.safetensors
2026-05-11 17:18:31 torch.distributed.all_reduce.sglang.safetensors
2026-05-11 17:19:22 torch.distributed.all_reduce.vllm.safetensors
```

我重新比较了这四个结果：

```text
torch_sglang      shape (8, 8192) dtype torch.bfloat16 sum_f32 -0.8750482201576233
torch_vllm        shape (8, 8192) dtype torch.bfloat16 sum_f32 -0.8750482201576233
fw_sglang_r0_new  shape (8, 8192) dtype torch.bfloat16 sum_f32 -0.8750482201576233
fw_vllm_r0_old    shape (8, 8192) dtype torch.bfloat16 sum_f32 -0.8750482201576233
```

bitwise 比较结果：

```text
torch_sglang vs torch_vllm: bitwise=True
fw_sglang_r0_new vs fw_vllm_r0_old: bitwise=True
fw_sglang_r0_new vs torch_sglang: bitwise=True
fw_sglang_r0_new vs torch_vllm: bitwise=True
fw_vllm_r0_old vs torch_sglang: bitwise=True
fw_vllm_r0_old vs torch_vllm: bitwise=True
```

也就是说，当前这组文件里：

```text
/data/like/temp/qdqx_2026_05_09___16_46_17_safetensors-online-compare-all-reduce-out/validate_all_reduce_rank-0.sglang.safetensors
/data/like/temp/qdqx_2026_05_09___16_46_17_safetensors-online-compare-all-reduce-out/validate_all_reduce_rank-0.vllm.safetensors
```

已经 bitwise 一致。

我还比较了 rank 0..7 全部结果：

```text
rank 0: bitwise=True shape=(8, 8192) dtype=torch.bfloat16
rank 1: bitwise=True shape=(8, 8192) dtype=torch.bfloat16
rank 2: bitwise=True shape=(8, 8192) dtype=torch.bfloat16
rank 3: bitwise=True shape=(8, 8192) dtype=torch.bfloat16
rank 4: bitwise=True shape=(8, 8192) dtype=torch.bfloat16
rank 5: bitwise=True shape=(8, 8192) dtype=torch.bfloat16
rank 6: bitwise=True shape=(8, 8192) dtype=torch.bfloat16
rank 7: bitwise=True shape=(8, 8192) dtype=torch.bfloat16
all_ranks_bitwise=True
```

因此，基于当前磁盘上的文件，结论不是“仍旧无法保证 reduce 结果 bitwise 一致”，而是：

1. 之前不一致的原因确实是 sglang 的 `--enable-deterministic-inference` 把 `NCCL_ALGO=Ring` 覆盖成了 `allreduce:tree`。
2. 去掉 `--enable-deterministic-inference` 后，sglang 框架内 debug 分支、vLLM 框架内 debug 分支、以及脱离框架的 `like-useful/torch_all_reduce.py` 输出已经全部 bitwise 一致。

如果你在另一次运行里仍看到不一致，建议优先确认以下几点：

1. 先看 sglang 日志里是否又出现：

   ```text
   NCCL_ALGO is set to 'allreduce:tree'
   ```

   只要出现这个警告，就说明 sglang 又改了 all_reduce 算法。

2. 确认比较的是新生成的文件。当前 sglang 文件时间是 `2026-05-11 17:53:51`，vLLM 文件时间还是 `2026-05-11 16:29:26`。如果 rerun 了其中一边，最好先清理：

   ```bash
   rm -rf /data/like/temp/qdqx_2026_05_09___16_46_17_safetensors-online-compare-all-reduce-out
   ```

   然后 sglang/vLLM 都重新生成，避免新旧文件混比。

3. 建议在 sglang 和 vLLM 的 debug 分支里临时打印实际生效的环境变量：

   ```python
   logger.info(
       f"validate all_reduce env: rank={my_tp_rank}, "
       f"NCCL_ALGO={os.environ.get('NCCL_ALGO')}, "
       f"NCCL_PROTO={os.environ.get('NCCL_PROTO')}, "
       f"NCCL_MIN_NCHANNELS={os.environ.get('NCCL_MIN_NCHANNELS')}, "
       f"NCCL_MAX_NCHANNELS={os.environ.get('NCCL_MAX_NCHANNELS')}, "
       f"NCCL_NVLS_ENABLE={os.environ.get('NCCL_NVLS_ENABLE')}"
   )
   ```

4. 如果需要可重复复核，可以用下面的 Python 片段检查所有 rank 是否 bitwise 一致：

   ```bash
   /data/like/miniconda3/envs/simo_sglang/bin/python - <<'PY'
   from safetensors.torch import load_file
   import torch

   base = "/data/like/temp/qdqx_2026_05_09___16_46_17_safetensors-online-compare-all-reduce-out"
   key = "read_quant_method_out_reduce"
   all_ok = True
   for rank in range(8):
       s = load_file(f"{base}/validate_all_reduce_rank-{rank}.sglang.safetensors")[key].contiguous()
       v = load_file(f"{base}/validate_all_reduce_rank-{rank}.vllm.safetensors")[key].contiguous()
       ok = tuple(s.shape) == tuple(v.shape) and s.dtype == v.dtype and torch.equal(
           s.view(torch.uint8), v.view(torch.uint8)
       )
       all_ok = all_ok and ok
       print(f"rank {rank}: bitwise={ok} shape={tuple(s.shape)} dtype={s.dtype}")
       if not ok:
           diff = (s.float() - v.float()).abs()
           print(
               "  max_abs_diff", float(diff.max().item()),
               "nonzero_values", int((s != v).sum().item()),
               "mismatched_bytes", int((s.view(torch.uint8) != v.view(torch.uint8)).sum().item()),
           )
   print("all_ranks_bitwise=", all_ok)
   PY
   ```

当前复核结果是 `all_ranks_bitwise=True`。

## 2026-05-12 lm-eval Wikitext PPL 差异分析

结论：这次 `sglang ppl=27.2849`、`vllm ppl=10.1594` 的首要原因不是 `RowParallelLinear`、不是 all-reduce，也不是常规 sampling 参数不一致。`wikitext` 在 lm-eval 里走的是 `loglikelihood_rolling`，本质是在算 prompt token logprob，不是在采样生成。

我查到的硬差异是：当前 lm-eval adapter 给 sglang engine 和 vLLM engine 传入的 `input_ids` 不一样，原因是 `add_bos_token` 默认值不同。

1. SGLang adapter:

   `/data/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/lm_eval/models/sglang_causallms.py`

   - `add_bos_token` 默认是 `False`。
   - `tok_encode()` 默认会强制 `add_special_tokens=False`。
   - `loglikelihood_rolling()` 又会用 `prefix_token=self.prefix_token_id`，也就是 BOS，作为 rolling window 的 prefix。

2. vLLM adapter:

   `/data/like/miniconda3/envs/simo_vllm/lib/python3.12/site-packages/lm_eval/models/vllm_causallms.py`

   - `add_bos_token` 默认是 `None`。
   - `_add_special_kwargs(add_special_tokens=None, add_bos=None)` 返回 `{}`，也就是使用 tokenizer 自己的默认行为。
   - Llama3.1 的 tokenizer 默认 `tok(text)` 会带 `<|begin_of_text|>`。
   - `loglikelihood_rolling()` 同样还会额外加一个 `prefix_token_id=BOS`。

我用 tokenizer-only 脚本验证了这个差异：

```text
sglang_default tok_encode= [9906, 1917]
sglang_default engine_input= [128000, 9906, 1917]

vllm_default tok_encode= [128000, 9906, 1917]
vllm_default engine_input= [128000, 128000, 9906, 1917]
```

也就是说，在当前命令下，lm-eval 传给两边的 Wikitext 请求不是同一个 token 序列：

- sglang 当前打分的是 `[BOS] + text_tokens`
- vLLM 当前打分的是 `[BOS] + [BOS] + text_tokens`

这也解释了为什么离线 Engine 给定一条 prompt 的 layer dump 几乎 bitwise 一致，但 lm-eval PPL 差很多：离线脚本验证的是你给定 prompt 下的 transformer forward 中间层；lm-eval 的 Wikitext 走的是 adapter 里的 rolling loglikelihood、tokenizer/BOS 处理、prompt logprob 提取路径。两者不是同一个输入构造路径。

sampling parameter 这条线我认为不是主因：

- SGLang adapter 在 `generate=False` 时设置的是 `temperature=0, max_new_tokens=1`，并打开 `return_logprob=True`。
- vLLM adapter 设置的是 `SamplingParams(temperature=0, prompt_logprobs=1, max_tokens=1, detokenize=False)`。
- 这里的 `max_new_tokens=1/max_tokens=1` 只是为了让 engine 返回 logprob 结构，不是在评估采样输出。

建议先做这个对齐实验：

1. 如果想复现 vLLM 当前默认语义，在 sglang lm-eval 的 `model_args` 里显式加：

   ```json
   "add_bos_token": true
   ```

   这样 sglang 也会走 `[BOS] + [BOS] + text_tokens`，预期 PPL 会向 vLLM 当前结果靠近。

2. 如果想做更干净的单 BOS 公平对比，在两边都显式设置：

   ```json
   "add_bos_token": false
   ```

   这样两边都应走 `[BOS] + text_tokens`。注意这会改变 vLLM 当前默认 PPL，不能再直接和旧的 `10.1594` 比。

如果统一 `add_bos_token` 后 PPL 仍明显不一致，下一步再查 `lm_head/logits_processor/input_token_logprobs`。当前 `debug_ppl_compare_qdqx.py` 的 safetensor 对比主要覆盖 transformer layer 内部输出，没有覆盖最终 `lm_head` logits 和 log-softmax/logprob 提取，所以不能用它单独证明 lm-eval 的 PPL 路径已经一致。

## 2026-05-12 SGLang lm-eval `add_bos_token` 报错原因和修复

这次报错的直接原因是你把：

```python
"add_bos_token": True,
```

加到了 `self.model_args` 里。`self.model_args` 会原样传给：

```python
self.model = sgl.Engine(**self.model_args)
```

而 SGLang 的 `ServerArgs.__init__()` 没有 `add_bos_token` 这个参数，所以日志里报：

```text
TypeError: ServerArgs.__init__() got an unexpected keyword argument 'add_bos_token'
```

正确做法：不要把 `add_bos_token` 传给 `sgl.Engine`。它是 lm-eval adapter 层的 tokenizer 行为参数，只应该传给 `SGLangLM.__init__()`，最后用于 `self.add_bos_token` 和 `tok_encode()`。

先把这行从 `self.model_args` 删除：

```diff
 self.model_args = {
     "model_path": pretrained,
-    "add_bos_token": True,
     "tokenizer_path": tokenizer_path,
```

然后用下面两种方式之一。

方式 A：命令行显式传入，推荐用于实验对比：

```bash
cp /share/users/like/ipc.sglang.1.json /dev/shm/like/; rm -rf /dev/shm/debug_ppl_save_input_ipc_dir-online/ /data/like/temp/qdqx_2026_05_11___11_45_27_safetensors-online ; CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_MIN_NCHANNELS=1 NCCL_MAX_NCHANNELS=1 NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 debug_env_file=/dev/shm/like/ipc.sglang.1.json SIMO_SGLANG_REGISTER=1 lm-eval --model sglang --model_args '{"pretrained": "/data/like/hf-models/llama3.1-70B", "add_bos_token": true, "quantization": "simo", "json_model_override_args": "{\"quantization_config_file\": \"/data/like/package/simo_conda_sglang/simo/extensions/sglang_simo/example/simo_quantization_config/online_quantization/quant_config_w4a4_mxfp-copy-from-xuhaifeng.json\"}", "tp_size": 8, "dtype": "auto", "mem_fraction_static": 0.5, "skip_server_warmup": true, "disable_cuda_graph": true, "attention_backend": "fa3" }' --tasks wikitext --batch_size auto > templ/sglang.wikitext.l2norm.lm-eval-mxfp4.log.`nowstr.sh` 2>&1 &
```

这会让 `add_bos_token=True` 被 `SGLangLM.__init__()` 捕获，不会进入 `**kwargs`，也不会进入 `self.model_args`，因此不会传给 `sgl.Engine`。

方式 B：如果想硬编码默认行为，只改函数签名默认值，不改 `self.model_args`：

```diff
-        add_bos_token: Optional[bool] = False,
+        add_bos_token: Optional[bool] = True,
```

更建议用方式 A，因为你后面还要做两种公平对比：

1. 对齐 vLLM 当前默认行为：SGLang 显式 `"add_bos_token": true`，vLLM 维持当前命令。这样两边都会更接近 `[BOS] + [BOS] + text_tokens`。
2. 做单 BOS 公平对比：SGLang 和 vLLM 都显式 `"add_bos_token": false`。这样两边都会更接近 `[BOS] + text_tokens`。

注意：如果你只是把 `add_bos_token=True` 加进 `self.model_args`，它永远是错的，因为这是 SGLang Engine 的 server args，不是 lm-eval tokenizer args。

## 2026-05-12 `like-useful/tokenizer-only.py`

我已经写了脚本：

```text
like-useful/tokenizer-only.py
```

这个脚本不启动 sglang/vLLM engine，只用 HF tokenizer 模拟 lm-eval 里两个 adapter 的关键路径：

1. SGLang adapter 的 `tok_encode()` 行为。
2. vLLM adapter 的 `tok_encode()` 行为。
3. `loglikelihood_rolling()` 额外插入 `prefix_token_id=BOS` 后的 first rolling window。
4. 实际传给 engine 的 `engine_input ids`。
5. 实际被计入 prompt logprob 的 `scored token ids`。

运行当前默认差异复现：

```bash
/data/like/miniconda3/envs/simo_sglang/bin/python like-useful/tokenizer-only.py \
  --text 'Hello world' \
  --sglang-add-bos-token false \
  --vllm-add-bos-token none
```

关键输出会类似：

```text
== sglang simulated, add_bos_token=False ==
tok_encode ids: [9906, 1917]
engine input ids: [128000, 9906, 1917]
scored token ids: [9906, 1917]

== vllm simulated, add_bos_token=None, add_special_tokens=None ==
tok_encode ids: [128000, 9906, 1917]
engine input ids: [128000, 128000, 9906, 1917]
scored token ids: [128000, 9906, 1917]
```

也就是当前默认下：

- sglang: `[BOS] + text_tokens`
- vLLM: `[BOS] + [BOS] + text_tokens`

让脚本模拟 SGLang 的 `"add_bos_token": true`：

```bash
/data/like/miniconda3/envs/simo_sglang/bin/python like-useful/tokenizer-only.py \
  --text 'Hello world' \
  --sglang-add-bos-token true \
  --vllm-add-bos-token none
```

这时 SGLang 模拟输出会变成：

```text
== sglang simulated, add_bos_token=True ==
tok_encode ids: [128000, 9906, 1917]
engine input ids: [128000, 128000, 9906, 1917]
scored token ids: [128000, 9906, 1917]
```

所以答案是：这个脚本可以控制 SGLang adapter 层的 `"add_bos_token": true`，参数是：

```bash
--sglang-add-bos-token true
```

但注意它控制的是 lm-eval SGLang adapter 的 tokenizer 行为，不是 SGLang Engine 的 `ServerArgs`。真实跑 lm-eval 时，也应该把 `"add_bos_token": true` 放在 `--model_args` 顶层，让 `SGLangLM.__init__()` 接住它；不要放到 `self.model_args` 里传给 `sgl.Engine`。

做单 BOS 公平对比可以这样运行：

```bash
/data/like/miniconda3/envs/simo_vllm/bin/python like-useful/tokenizer-only.py \
  --text 'Hello world' \
  --sglang-add-bos-token false \
  --vllm-add-bos-token false
```

这时两边都会是：

```text
engine input ids: [128000, 9906, 1917]
scored token ids: [9906, 1917]
```

脚本已用下面命令检查过语法：

```bash
/data/like/miniconda3/envs/simo_sglang/bin/python -m py_compile like-useful/tokenizer-only.py
```

## 2026-05-12 为什么 `pwdx` 显示 `/sgl-workspace/sglang`，但宿主机 `ls /` 看不到

这不是 `pwdx` 错，也不是 `ls /` 漏显示。原因是进程 `2458670` 很可能运行在容器/namespace 里，它看到的 `/` 和你当前 shell 看到的宿主机 `/` 不是同一个 mount namespace。

你的 `/proc/2458670/status` 里有这个关键信号：

```text
NStgid: 2458670 60556
NSpid:  2458670 60556
```

这说明该进程在嵌套 PID namespace 里有另一个 pid：宿主机上是 `2458670`，namespace/container 里是 `60556`。这种情况下通常也会有独立的 mount namespace。`pwdx 2458670` 读的是该进程自己的当前工作目录，所以显示：

```text
2458670: /sgl-workspace/sglang
```

但你执行：

```bash
sudo ls /
```

列出来的是你当前 shell 所在的宿主机 mount namespace 的根目录。宿主机根目录里没有 `/sgl-workspace`，不代表目标进程的 mount namespace 里没有 `/sgl-workspace`。

可以用下面命令验证。

比较当前 shell 和目标进程的 mount namespace：

```bash
readlink /proc/$$/ns/mnt
sudo readlink /proc/2458670/ns/mnt
```

如果两个输出不同，例如：

```text
mnt:[4026531841]
mnt:[40265xxxxx]
```

就说明二者看到的是不同的挂载视图。

从目标进程自己的 root 去看它的 `/`：

```bash
sudo ls -la /proc/2458670/root/
sudo ls -la /proc/2458670/root/sgl-workspace/
sudo ls -la /proc/2458670/root/sgl-workspace/sglang
```

`/proc/2458670/root` 是 procfs 提供的 magic link，指向目标进程视角里的根目录。因此如果 `/sgl-workspace/sglang` 只存在于该进程的 mount namespace 里，宿主机可以通过：

```text
/proc/2458670/root/sgl-workspace/sglang
```

访问它。

也可以直接进入目标进程的 mount namespace 查看：

```bash
sudo nsenter -t 2458670 -m -- pwd
sudo nsenter -t 2458670 -m -- ls /
sudo nsenter -t 2458670 -m -- ls -la /sgl-workspace/sglang
```

如果还想知道 `/sgl-workspace/sglang` 在宿主机上对应哪个真实目录或 overlay/bind mount，可以看目标进程的 mountinfo：

```bash
sudo grep -E 'sgl-workspace|overlay|/data|/share' /proc/2458670/mountinfo
```

总结：`pwdx` 显示的是进程 `2458670` 自己 namespace 里的 cwd；`sudo ls /` 显示的是你当前 shell 的宿主机 `/`。二者 namespace 不同，所以路径可以不一致。

## 2026-05-12 如何找到 pid 2458670 属于哪个 Docker container

优先用 `/proc/<pid>/cgroup`。很多 Docker/containerd 环境会把 container id 写在 cgroup 路径里：

```bash
sudo cat /proc/2458670/cgroup
```

常见输出形态包括：

```text
0::/system.slice/docker-<64位container_id>.scope
0::/docker/<64位container_id>
0::/kubepods.slice/.../cri-containerd-<64位container_id>.scope
```

可以直接提取疑似 container id：

```bash
cid=$(sudo cat /proc/2458670/cgroup | grep -Eo '[0-9a-f]{64}' | head -n 1)
echo "$cid"
```

然后用 Docker 查：

```bash
docker ps --no-trunc | grep "$cid"
docker inspect "$cid" --format 'name={{.Name}} id={{.Id}} pid={{.State.Pid}} image={{.Config.Image}}'
```

如果 `cid` 只有 12 位短 id，也可以：

```bash
docker ps --no-trunc | grep "$(echo "$cid" | cut -c1-12)"
```

如果 `/proc/2458670/cgroup` 没有暴露 container id，可以用 namespace 反查。先看目标进程的 namespace：

```bash
sudo readlink /proc/2458670/ns/pid
sudo readlink /proc/2458670/ns/mnt
```

再遍历所有 Docker container 的 init pid，看哪个 namespace 一样：

```bash
target_pidns=$(sudo readlink /proc/2458670/ns/pid)
target_mntns=$(sudo readlink /proc/2458670/ns/mnt)

for cid in $(docker ps -q); do
  init_pid=$(docker inspect -f '{{.State.Pid}}' "$cid")
  pidns=$(sudo readlink /proc/$init_pid/ns/pid 2>/dev/null || true)
  mntns=$(sudo readlink /proc/$init_pid/ns/mnt 2>/dev/null || true)
  if [ "$pidns" = "$target_pidns" ] || [ "$mntns" = "$target_mntns" ]; then
    docker inspect "$cid" --format 'container={{.Name}} id={{.Id}} init_pid={{.State.Pid}} image={{.Config.Image}}'
  fi
done
```

你的 `/proc/2458670/status` 里有：

```text
NSpid:  2458670 60556
```

这表示宿主机 pid 是 `2458670`，container/内层 PID namespace 里的 pid 很可能是 `60556`。找到候选 container 后，可以进去确认：

```bash
docker exec -it <container_id_or_name> ps -ef | grep 60556
```

或者不用进入交互 shell：

```bash
docker exec <container_id_or_name> sh -lc 'ps -ef | grep 60556'
```

如果系统不是 Docker，而是 containerd/Kubernetes，`docker ps` 可能看不到。此时用：

```bash
sudo cat /proc/2458670/cgroup
sudo grep -Eo '([0-9a-f]{64})' /proc/2458670/cgroup
```

拿到 id 后查 containerd：

```bash
sudo crictl ps -a | grep "$(echo "$cid" | cut -c1-12)"
sudo crictl inspect "$cid"
```

最短路径通常是：

```bash
cid=$(sudo cat /proc/2458670/cgroup | grep -Eo '[0-9a-f]{64}' | head -n 1)
docker inspect "$cid" --format 'name={{.Name}} pid={{.State.Pid}} image={{.Config.Image}}'
```

如果这个拿不到 id，再用 namespace inode 反查。

## 2026-05-12 lm-eval 向 SGLang 按 batch size 1、固定顺序发送 Wikitext 请求

把命令里的：

```bash
--batch_size auto
```

改成：

```bash
--batch_size 1
```

并建议显式加固定 seed：

```bash
--seed 0,1234,1234,1234
```

如果还想让 SGLang engine 侧也尽量只同时跑一个请求，可以在 `model_args` 里加：

```json
"max_running_requests": 1
```

修改后的命令：

```bash
SIMO_SGLANG_REGISTER=1 lm-eval \
  --model sglang \
  --model_args '{"pretrained": "/data/like/hf-models/llama3.1-70B-strip-layers", "add_bos_token": true, "quantization": "simo", "json_model_override_args": "{\"quantization_config_file\": \"/data/like/package/simo_conda_sglang/simo/extensions/sglang_simo/example/simo_quantization_config/online_quantization/quant_config_w4a4_mxfp-copy-from-xuhaifeng.json\"}", "tp_size": 1, "dtype": "auto", "mem_fraction_static": 0.5, "skip_server_warmup": true, "disable_cuda_graph": true, "attention_backend": "fa3", "max_running_requests": 1}' \
  --tasks wikitext \
  --batch_size 1 \
  --seed 0,1234,1234,1234
```

如果要保留你原来的环境变量和日志重定向，可以写成：

```bash
SIMO_SGLANG_REGISTER=1 lm-eval --model sglang --model_args '{"pretrained": "/data/like/hf-models/llama3.1-70B-strip-layers", "add_bos_token": true, "quantization": "simo", "json_model_override_args": "{\"quantization_config_file\": \"/data/like/package/simo_conda_sglang/simo/extensions/sglang_simo/example/simo_quantization_config/online_quantization/quant_config_w4a4_mxfp-copy-from-xuhaifeng.json\"}", "tp_size": 1, "dtype": "auto", "mem_fraction_static": 0.5, "skip_server_warmup": true, "disable_cuda_graph": true, "attention_backend": "fa3", "max_running_requests": 1}' --tasks wikitext --batch_size 1 --seed 0,1234,1234,1234 > templ/sglang.wikitext.bs1.log.`nowstr.sh` 2>&1
```

为什么这样能固定请求顺序：

1. `wikitext` 在 lm-eval 里走 `loglikelihood_rolling`，不是采样生成。
2. `SGLangLM.loglikelihood_rolling()` 会先按数据集原始顺序构造 `all_windows`。
3. 当 `--batch_size 1` 时，外层循环每次只取一个 window：

   ```python
   batch_size = adaptive_batch_size or int(self.batch_size)
   for i in range(0, len(all_windows), batch_size):
       batch = all_windows[i : i + batch_size]
   ```

4. 因为每次传给 `_loglikelihood_tokens()` 的只有一个请求，里面的 `Collator` 即使按长度排序，也没有多个请求可重排。
5. `--seed 0,1234,1234,1234` 固定 python/numpy/torch/fewshot seed。对 Wikitext 这种 0-shot rolling loglikelihood 来说 seed 通常不影响样本顺序，但显式写上更利于复现实验。

注意：不要再用 `--batch_size auto`。`auto` 在 SGLang adapter 里会把 batch size 设成 `len(requests)`，等价于把很多 Wikitext request 一次性交给 adapter，然后 `_loglikelihood_tokens()` 里会按长度排序和批处理，不适合调试“逐条、固定顺序”的请求路径。

如果你需要的是“严格按 Wikitext 数据集原始 doc index 顺序”且后续又想用 `batch_size > 1`，那就需要改 `sglang_causallms.py`，禁用 `_loglikelihood_tokens()` 里的 `Collator` 长度排序；但在 `--batch_size 1` 下，不需要改代码。

## 2026-05-12 `rolling_token_windows = list(map(...))` 语法解释

代码：

```python
rolling_token_windows: List[Tuple[List[int], List[int]]] = list(
    map(
        make_disjoint_window,
        get_rolling_token_windows(
            token_list=self.tok_encode(string),
            prefix_token=self.prefix_token_id,
            # max_seq_len - (1 for context)
            max_seq_len=self.max_length - 1,
            context_len=1,
        ),
    )
)
```

这段代码等价于更展开的写法：

```python
token_list = self.tok_encode(string)

raw_windows = get_rolling_token_windows(
    token_list=token_list,
    prefix_token=self.prefix_token_id,
    max_seq_len=self.max_length - 1,
    context_len=1,
)

rolling_token_windows = []
for window in raw_windows:
    disjoint_window = make_disjoint_window(window)
    rolling_token_windows.append(disjoint_window)
```

各部分含义：

1. `rolling_token_windows: List[Tuple[List[int], List[int]]]`

   这是 Python 类型标注，表示 `rolling_token_windows` 预期是一个 list，list 里的每个元素是一个二元 tuple：

   ```python
   (context_tokens, continuation_tokens)
   ```

   其中 `context_tokens` 和 `continuation_tokens` 都是 `List[int]`，也就是 token id 列表。

2. `self.tok_encode(string)`

   把原始文本 `string` tokenize 成 token id 列表。例如：

   ```python
   "Hello world" -> [9906, 1917]
   ```

3. `get_rolling_token_windows(...)`

   这是一个 generator/迭代器函数，用来把很长的 token 序列切成多个 rolling window。每个 window 原始形态大致是：

   ```python
   (input_tokens, pred_tokens)
   ```

   对第一个 window，它会人为在开头加一个 `prefix_token`，通常是 BOS：

   ```python
   input_tokens = [BOS] + token_list[: first_seq_len - 1]
   pred_tokens  = token_list[:first_seq_len]
   ```

   它的目标是让模型可以给 `pred_tokens` 里的 token 算 logprob。

4. `map(make_disjoint_window, get_rolling_token_windows(...))`

   `map(func, iterable)` 的意思是：对 iterable 里的每个元素调用一次 `func`。

   这里就是：

   ```python
   for window in get_rolling_token_windows(...):
       make_disjoint_window(window)
   ```

   `map(...)` 本身也是 lazy iterator，不会立刻执行完。

5. `make_disjoint_window`

   `get_rolling_token_windows()` 产出的 `(input_tokens, pred_tokens)` 里面，`input_tokens` 和 `pred_tokens` 可能有重叠。`make_disjoint_window()` 会把重叠部分从 context 里裁掉，变成：

   ```python
   (context_tokens, continuation_tokens)
   ```

   lm-eval 后面会把：

   ```python
   context_tokens + continuation_tokens
   ```

   作为 engine input，然后只统计 `continuation_tokens` 的 logprob。

   对短文本 `"Hello world"`，如果 token 是：

   ```python
   token_list = [9906, 1917]
   prefix_token = 128000  # BOS
   ```

   则第一个 window 最终会变成：

   ```python
   context_tokens      = [128000]
   continuation_tokens = [9906, 1917]
   engine_input        = [128000, 9906, 1917]
   ```

6. 外层 `list(...)`

   因为 `map(...)` 和 `get_rolling_token_windows(...)` 都是 lazy iterator，外层 `list(...)` 会把它们一次性消费完，生成真正的 Python list。

所以这段代码一句话解释就是：

```text
把一条文本 tokenize，然后切成一个或多个 rolling loglikelihood 窗口；
每个窗口拆成 context 和 continuation；
最后得到 [(context_tokens, continuation_tokens), ...] 这个列表。
```

它服务于 Wikitext PPL 这种任务：模型不需要生成新文本，而是对数据集原文的每个 token 计算 logprob。`context_tokens` 是条件上下文，`continuation_tokens` 是要被打分的真实 token。

## 2026-05-12 `get_rolling_token_windows` 函数讲解

源码位置：

```text
/data/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/lm_eval/utils.py
```

函数源码：

```python
def get_rolling_token_windows(
    token_list: List[int], prefix_token: int, max_seq_len: int, context_len: int
) -> Generator[Tuple[List[int], List[int]], None, None]:
    assert 1 <= context_len <= max_seq_len
    if not token_list:
        return

    pred_len = max_seq_len - context_len + 1
    predicted = 0

    first_seq_len = min(max_seq_len, len(token_list))
    yield [prefix_token] + token_list[: first_seq_len - 1], token_list[:first_seq_len]
    predicted += first_seq_len

    while predicted < len(token_list):
        window_pred_len = min(len(token_list) - predicted, pred_len)
        window_end = predicted + window_pred_len

        yield (
            token_list[window_end - max_seq_len - 1 : window_end - 1],
            token_list[window_end - window_pred_len : window_end],
        )
        predicted += window_pred_len
```

这个函数的作用：

```text
把一串很长的 token_list 切成多个 rolling window；
每个 window 返回一个二元组：

(input_tokens, pred_tokens)
```

其中：

- `input_tokens`：送进模型、用于产生 logits 的 token。
- `pred_tokens`：这一段 window 里要被打分/预测的真实 token。
- `prefix_token`：第一个 token 没有上文，所以人为加一个 dummy token，通常是 BOS/EOS，让第一个真实 token 也能被打分。
- `max_seq_len`：单个 raw window 的最大长度。
- `context_len`：从上一段保留多少上下文 token，用来预测下一段。

注意它是 generator，因为里面用了 `yield`。调用 `get_rolling_token_windows(...)` 不会立刻生成全部结果，只有遍历它或套 `list(...)` 时才会逐个生成。

### 例子 1：短文本，一次 window 就够

假设：

```python
token_list = [10, 11, 12]
prefix_token = 0
max_seq_len = 5
context_len = 1
```

因为 `len(token_list)=3 <= max_seq_len=5`，只会 yield 一次：

```python
input_tokens = [0, 10, 11]
pred_tokens  = [10, 11, 12]
```

含义是：

```text
0  -> 预测 10
10 -> 预测 11
11 -> 预测 12
```

所以原始输出形式是：

```python
[
    ([0, 10, 11], [10, 11, 12])
]
```

lm-eval 后面还会对它调用 `make_disjoint_window`：

```python
make_disjoint_window(([0, 10, 11], [10, 11, 12]))
```

得到：

```python
context_tokens      = [0]
continuation_tokens = [10, 11, 12]
```

最终 engine input 是：

```python
[0, 10, 11, 12]
```

但只统计 `[10, 11, 12]` 的 logprob。

### 例子 2：长文本，需要多个 rolling window

假设：

```python
token_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
prefix_token = 0
max_seq_len = 5
context_len = 2
```

先计算：

```python
pred_len = max_seq_len - context_len + 1
         = 5 - 2 + 1
         = 4
```

意思是：第一个 window 后，每个后续 window 最多新增打分 4 个 token，并保留 2 个上下文 token。

这个函数原始 yield 结果是：

```python
[
    ([0, 1, 2, 3, 4], [1, 2, 3, 4, 5]),
    ([4, 5, 6, 7, 8], [6, 7, 8, 9]),
    ([5, 6, 7, 8, 9], [10]),
]
```

逐段看：

第一段：

```python
input_tokens = [0, 1, 2, 3, 4]
pred_tokens  = [1, 2, 3, 4, 5]
```

用 prefix `0` 开始，给 token `[1,2,3,4,5]` 打分。

第二段：

```python
input_tokens = [4, 5, 6, 7, 8]
pred_tokens  = [6, 7, 8, 9]
```

这里 `[4,5]` 是上下文，真正新增打分的是 `[6,7,8,9]`。

第三段：

```python
input_tokens = [5, 6, 7, 8, 9]
pred_tokens  = [10]
```

最后只剩 token `10` 需要打分。

如果再套 `make_disjoint_window`，就会变成 lm-eval 后续实际使用的 `(context_tokens, continuation_tokens)`：

```python
[
    ([0],    [1, 2, 3, 4, 5]),
    ([4, 5], [6, 7, 8, 9]),
    ([5, 6, 7, 8, 9], [10]),
]
```

对应 engine input 分别是：

```python
[0, 1, 2, 3, 4, 5]
[4, 5, 6, 7, 8, 9]
[5, 6, 7, 8, 9, 10]
```

每次只统计 continuation 部分的 logprob：

```python
[1, 2, 3, 4, 5]
[6, 7, 8, 9]
[10]
```

这样，整条 `token_list = [1,2,3,4,5,6,7,8,9,10]` 中每个 token 都被打分一次，不会漏，也不会重复计入。

### 在 SGLang adapter 里的特殊点

SGLang adapter 调用时写的是：

```python
max_seq_len=self.max_length - 1
context_len=1
```

这里 `max_seq_len` 故意减 1，是因为后面 `make_disjoint_window` 后会构造：

```python
context_tokens + continuation_tokens
```

第一个 window 会额外有一个 prefix/context token。用 `self.max_length - 1` 可以保证最终送进 engine 的 token 数不超过模型最大长度。

一句话总结：

```text
get_rolling_token_windows 把一段 token 序列切成“可逐段计算 logprob”的窗口；
每个窗口告诉后续代码：用哪些 token 做输入，以及这一窗口要给哪些真实 token 打分。
```

## 2026-05-13 SGLang lm-eval 时 `logger.info` 不打印，以及 62 条 Wikitext 为什么有 124 次 `_forward_raw`

### 1. 为什么 `logger.info` 看不到

你这次日志里已经能看到 SGLang Engine 的实际参数：

```text
server_args=ServerArgs(... log_level='error', ...)
```

所以在 SGLang worker/scheduler/model runner 进程里，普通：

```python
logger.info(...)
```

会被日志级别过滤掉。`model_runner.py` 里的 logger 是：

```python
logger = logging.getLogger(__name__)
```

它最终受 SGLang worker 进程的 root logger / server_args.log_level 控制。当前是 `error`，所以 `info` 级别不会出现在 lm-eval 重定向的日志里。

解决方法：在 `--model_args` 里显式加：

```json
"log_level": "info"
```

例如：

```bash
CUDA_VISIBLE_DEVICES=0 debug_env_file=/dev/shm/like/ipc.sglang.1.json SIMO_SGLANG_REGISTER=1 lm-eval --model sglang --model_args '{"pretrained": "/data/like/hf-models/llama3.1-70B-strip-layers/", "add_bos_token": true, "quantization": "simo", "json_model_override_args": "{\"quantization_config_file\": \"/data/like/package/simo_conda_sglang/simo/extensions/sglang_simo/example/simo_quantization_config/online_quantization/quant_config_w4a4_mxfp-copy-from-xuhaifeng.json\"}", "tp_size": 1, "dtype": "auto", "mem_fraction_static": 0.5, "skip_server_warmup": true, "disable_cuda_graph": true, "attention_backend": "fa3", "max_running_requests": 1, "watchdog_timeout": 2592000, "log_level": "info"}' --tasks wikitext --batch_size 1 > templ/sglang.wikitext.l2norm.lm-eval-mxfp4.log.`nowstr.sh` 2>&1 &
```

如果只是临时调试，也可以把调试日志改成：

```python
logger.warning(...)
```

或继续用你现在这种写文件/counter 的方式。对 `_forward_raw` 这种高频路径，写文件通常比刷 `logger.info` 更可靠，也更容易按 rank 做精确统计。

### 2. 为什么 62 条 Wikitext 数据会触发 124 次 `_forward_raw`

关键在 lm-eval 的 SGLang adapter：

```text
/data/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/lm_eval/models/sglang_causallms.py
```

里面 `_model_generate(generate=False)` 对 loglikelihood 请求写死了：

```python
if not generate:
    sampling_params = sampling_params if sampling_params else {}
    sampling_params.update(
        {
            "temperature": 0,
            "max_new_tokens": 1,
        }
    )
```

也就是说，虽然 Wikitext PPL 本质只需要 prompt token logprob，但 adapter 仍然让 SGLang engine 每个请求生成 1 个新 token。

SGLang 对每条这样的请求会走两段：

1. prefill/extend forward：处理完整 prompt，计算 `input_token_logprobs`。
2. decode forward：因为 `max_new_tokens=1`，还要生成 1 个 output token。

这两段都会进入 `ModelRunner._forward_raw`。

所以当前计数符合预期：

```text
62 条 Wikitext rolling loglikelihood 请求
每条请求 1 次 prefill + 1 次 decode
= 124 次 _forward_raw
```

你读到的 counter 是：

```python
array([123], dtype=uint64)
```

这也正好对应 124 次调用，因为你的 counter 代码在文件不存在时只创建 `[0]`，没有先加 1：

```python
if not os.path.exists(...):
    _forward_raw_counter = np.array([0], dtype=np.uint64)
    _forward_raw_counter.tofile(...)
else:
    _forward_raw_counter = np.fromfile(...)
    _forward_raw_counter[0] += 1
```

因此：

```text
counter 文件最终值 123
实际 _forward_raw 调用次数 = 123 + 1 = 124
```

日志里也能看到 `--batch_size 1` 后，每次都是单请求：

```text
Running loglikelihood requests: 100%|...| 1/1
```

这说明不是 batch 里多条请求导致的额外 forward，而是每条请求本身有 prefill + decode 两个阶段。

### 3. 如果只想算 prompt logprob，能不能避免 decode

理论上可以。因为 `_parse_logprobs()` 只使用：

```python
outputs["meta_info"]["input_token_logprobs"]
```

并没有使用生成出来的 output token。对纯 loglikelihood/PPL 来说，`max_new_tokens=1` 这次 decode 是多余的。

可以尝试把 SGLang lm-eval adapter 里的：

```python
"max_new_tokens": 1,
```

改成：

```python
"max_new_tokens": 0,
```

也就是：

```diff
 if not generate:
     sampling_params = sampling_params if sampling_params else {}
     sampling_params.update(
         {
             "temperature": 0,
-            "max_new_tokens": 1,
+            "max_new_tokens": 0,
         }
     )
```

SGLang 代码库里已有 `max_new_tokens=0` 的 scoring 用法，例如：

```text
/data/like/package/sglang_kernel_src/test/registered/core/test_srt_endpoint.py
```

里面有：

```python
score = run_generate(
    new_prompt, return_logprob=True, logprob_start_len=0, max_new_tokens=0
)
```

如果这个改法在你的 lm-eval 路径上正常工作，预期 `_forward_raw` 次数会从：

```text
62 * 2 = 124
```

降到接近：

```text
62 * 1 = 62
```

按你当前 counter 写法，最终文件值应接近：

```text
61
```

不过这是 adapter 行为修改，建议先用小模型或 `llama3.1-70B-strip-layers` 跑一次确认结果结构仍然包含 `meta_info["input_token_logprobs"]`。

## 2026-05-13 是谁把 SGLang `log_level` 从默认 `info` 变成 `error`

结论：不是 lm-eval 设置的，是 SGLang Python API `sgl.Engine(**kwargs)` 自己设置的。

`ServerArgs` 类里的默认值确实是：

```python
# /data/like/package/sglang_kernel_src/python/sglang/srt/server_args.py
log_level: str = "info"
```

但 lm-eval 使用的是 SGLang 的 offline Engine Python API：

```python
# /data/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/lm_eval/models/sglang_causallms.py
self.model = sgl.Engine(**self.model_args)
```

然后进入：

```python
# /data/like/package/sglang_kernel_src/python/sglang/srt/entrypoints/engine.py
def __init__(self, **kwargs):
    ...
    if "server_args" in kwargs:
        server_args = kwargs["server_args"]
    else:
        # Construct server_args from kwargs
        if "log_level" not in kwargs:
            # Do not print logs by default
            kwargs["log_level"] = "error"
        server_args = self.server_args_class(**kwargs)
```

关键就是这几行：

```python
if "log_level" not in kwargs:
    kwargs["log_level"] = "error"
```

所以流程是：

```text
lm-eval SGLangLM.__init__
  -> self.model_args 里没有 log_level
  -> sgl.Engine(**self.model_args)
  -> Engine.__init__ 发现 kwargs 没有 log_level
  -> 主动写入 kwargs["log_level"] = "error"
  -> ServerArgs(log_level="error")
```

这解释了为什么日志里最终看到：

```text
server_args=ServerArgs(... log_level='error', ...)
```

而不是 `ServerArgs` 类定义里的默认 `info`。

如果要阻止这个覆盖，必须在调用 `sgl.Engine` 之前显式传入 `log_level`。对 lm-eval 来说，就是把它放到 `--model_args` 顶层：

```json
"log_level": "info"
```

例如：

```bash
CUDA_VISIBLE_DEVICES=0 debug_env_file=/dev/shm/like/ipc.sglang.1.json SIMO_SGLANG_REGISTER=1 lm-eval --model sglang --model_args '{"pretrained": "/data/like/hf-models/llama3.1-70B-strip-layers/", "add_bos_token": true, "quantization": "simo", "json_model_override_args": "{\"quantization_config_file\": \"/data/like/package/simo_conda_sglang/simo/extensions/sglang_simo/example/simo_quantization_config/online_quantization/quant_config_w4a4_mxfp-copy-from-xuhaifeng.json\"}", "tp_size": 1, "dtype": "auto", "mem_fraction_static": 0.5, "skip_server_warmup": true, "disable_cuda_graph": true, "attention_backend": "fa3", "max_running_requests": 1, "watchdog_timeout": 2592000, "log_level": "info"}' --tasks wikitext --batch_size 1 > templ/sglang.wikitext.l2norm.lm-eval-mxfp4.log.`nowstr.sh` 2>&1 &
```

因为 `sglang_causallms.py` 里有：

```python
self.model_args.update(kwargs)
```

所以 `--model_args` 顶层的 `"log_level": "info"` 会进入 `self.model_args`，再传给 `sgl.Engine(**self.model_args)`。这时 `Engine.__init__` 看到 `kwargs` 里已经有 `log_level`，就不会再覆盖成 `error`。

一句话：`ServerArgs` 默认是 `info`，但 `sgl.Engine` Python API 为了 offline Engine 默认少打印日志，在没有显式传 `log_level` 时把它改成了 `error`。

## vLLM GroupCoordinator 里 rank、rank_in_group、local_rank 的区别

代码位置：

```python
/data/like/package/vllm-for-conda-simo/vllm/distributed/parallel_state.py
```

`GroupCoordinator` 里这几个字段的含义是：

```python
rank: int           # global rank
ranks: list[int]    # global ranks in the group
world_size: int     # size of the group
local_rank: int     # local rank used to assign devices
rank_in_group: int  # rank inside the group
```

核心区别：

```text
rank          = 当前进程在全局 torch.distributed world 里的 rank
rank_in_group = 当前进程在当前这个 GroupCoordinator.ranks 列表里的下标
local_rank    = 当前进程在本机 node 上的本地 rank，通常用于选择 cuda:{local_rank}
```

`GroupCoordinator.__init__` 里真正设置 `rank_in_group` 的代码是：

```python
self.rank = torch.distributed.get_rank()
self.local_rank = local_rank

for ranks in group_ranks:
    ...
    if self.rank in ranks:
        self.ranks = ranks
        self.world_size = len(ranks)
        self.rank_in_group = ranks.index(self.rank)
```

所以 `rank_in_group` 不是全局 rank，而是：

```python
rank_in_group = self.ranks.index(global_rank)
```

也就是当前 global rank 在当前通信组 `self.ranks` 里的位置。

注释里的例子是：

```text
Process | Node | Rank | Local Rank | Rank in Group
  0     |   0  |  0   |     0      |       0
  1     |   0  |  1   |     1      |       1
  2     |   1  |  2   |     0      |       2
  3     |   1  |  3   |     1      |       3
```

这个例子里 `Rank` 和 `Rank in Group` 都是 `0,1,2,3`，原因是这个 group 刚好包含了全局所有 rank，并且顺序也是 `[0, 1, 2, 3]`。在这种特殊情况下：

```text
self.ranks = [0, 1, 2, 3]
global rank 0 -> rank_in_group 0
global rank 1 -> rank_in_group 1
global rank 2 -> rank_in_group 2
global rank 3 -> rank_in_group 3
```

所以 `rank == rank_in_group` 只是巧合。这个注释主要想说明的是 `local_rank` 和 `rank_in_group` 的区别：在 2 个节点上，每个节点的 `local_rank` 都会从 0 重新开始，所以 node 1 上的全局 rank 2 的 `local_rank` 是 0，但它在这个 group 里的 `rank_in_group` 是 2。

更能看出差异的例子如下。

例子 1：全局 8 个进程，某个 TP group 是：

```python
self.ranks = [4, 5, 6, 7]
```

那么 global rank 6 这个进程上：

```text
rank = 6
rank_in_group = 2   # 因为 self.ranks.index(6) == 2
```

例子 2：group 不是连续 rank：

```python
self.ranks = [0, 2, 4, 6]
```

那么 global rank 4 这个进程上：

```text
rank = 4
rank_in_group = 2   # 因为 self.ranks.index(4) == 2
```

例子 3：同一个进程在不同 group 里的 `rank_in_group` 可以不同。

假设 global rank 6：

```text
TP group: [4, 5, 6, 7]  -> rank_in_group = 2
DP group: [0, 2, 4, 6]  -> rank_in_group = 3
PP group: [6, 7]        -> rank_in_group = 0
```

同一个进程的全局 `rank` 始终是 6，但是它在不同通信组里的 `rank_in_group` 不一样。

这也是为什么 vLLM 有些 API 参数写的是 group 内 rank，而真正调用 `torch.distributed` 时要转换成 global rank。例如：

```python
def broadcast(self, input_: torch.Tensor, src: int = 0):
    """Broadcast the input tensor.
    NOTE: `src` is the local rank of the source rank.
    """
    torch.distributed.broadcast(
        input_, src=self.ranks[src], group=self.device_group
    )
```

这里 `src` 是 group 内的 rank，也就是 `rank_in_group` 语义；但 `torch.distributed.broadcast` 需要的是全局 rank，所以代码用：

```python
self.ranks[src]
```

把 group 内 rank 转成 global rank。

再看 `next_rank`：

```python
rank_in_group = self.rank_in_group
world_size = self.world_size
return self.ranks[(rank_in_group + 1) % world_size]
```

它先用 `rank_in_group` 在 group 内找下一个位置，再从 `self.ranks[...]` 取出对应的 global rank 返回。

一句话总结：

```text
rank 是全局编号。
rank_in_group 是当前进程在某个通信子组里的编号。
local_rank 是当前进程在当前机器上的本地编号，主要用于绑 GPU。
```

注释里的 4 进程 2 节点例子里 `Rank` 和 `Rank in Group` 一样，只是因为那个 group 恰好是完整 world group `[0,1,2,3]`；在 TP/PP/DP/EP 这类子组里，它们经常不同。

## vLLM lm-eval 如何保证一次只处理一个 wikitext 请求

结论：你这个命令里的 `--batch_size 1` 可以让 lm-eval 的 vLLM 适配器一次只把 1 个 request/window 传给 `vllm.LLM.generate()`；但如果要在 vLLM Engine 调度层也限制“同一个调度 iteration 最多处理 1 条 sequence”，还应该在 `--model_args` 里加：

```json
"max_num_seqs": 1
```

vLLM 里最接近 SGLang `"max_running_requests": 1` 的参数是：

```text
max_num_seqs
```

代码依据：

```python
# /data/like/package/vllm-for-conda-simo/vllm/config/scheduler.py
max_num_seqs: int = Field(default=DEFAULT_MAX_NUM_SEQS, ge=1)
"""Maximum number of sequences to be processed in a single iteration."""
```

在 lm-eval 的 vLLM adapter 里：

```python
# /data/like/miniconda3/envs/simo_vllm/lib/python3.12/site-packages/lm_eval/models/vllm_causallms.py
self.model_args = {
    ...
    "max_num_seqs": kwargs.get("max_num_seqs", max_batch_size),
    ...
}
self.model_args.update(kwargs)
self.batch_size = int(batch_size)
self.model = LLM(**self.model_args)
```

所以在 `--model_args` JSON 里直接写：

```json
"max_num_seqs": 1
```

会传给 `vllm.LLM(...)`，再进入 vLLM scheduler config。

lm-eval 的 `--batch_size 1` 控制的是 adapter 层分批。对 wikitext 这种 `loglikelihood_rolling` 任务，vLLM adapter 先把每条文本切成 rolling windows：

```python
all_windows.extend((req_idx, window) for window in windows)
```

然后按 `self.batch_size` 切 batch：

```python
batch_size = adaptive_batch_size or int(self.batch_size)
for i in range(0, len(all_windows), batch_size):
    batch = all_windows[i : i + batch_size]
    batch_nlls = self._loglikelihood_tokens(requests=batch_windows)
```

`_loglikelihood_tokens()` 里面又会按 `self.batch_size` 分 chunk，然后调用：

```python
outputs = self._model_generate(requests=inputs, generate=False)
```

最后 `_model_generate()` 调用：

```python
outputs = self.model.generate(
    [TokensPrompt(prompt_token_ids=request) for request in requests],
    sampling_params=sampling_params,
    ...
)
```

因此，当 `--batch_size 1` 时，传给 `self.model.generate(...)` 的 `requests` 长度就是 1。也就是说，lm-eval 侧一次只向 vLLM 提交 1 个 window。

需要注意两个层次：

```text
--batch_size 1
    控制 lm-eval adapter 一次提交多少个请求/window 给 vLLM。

"max_num_seqs": 1
    控制 vLLM scheduler 一个 iteration 最多调度多少条 sequence。
```

如果只设置 `--batch_size 1`，在当前 lm-eval offline vLLM 用法下通常已经是一次提交一条；但为了排除 vLLM scheduler 内部继续合并多条 sequence 的可能性，建议同时设置 `"max_num_seqs": 1`。

推荐命令写成这样：

```bash
cp /share/users/like/ipc.vllm.1.json /dev/shm/like/ipc.vllm.1.json ; rm -rf /dev/shm/debug_ppl_save_input_ipc_dir-vllm/ /data/like/temp/qdqx_2026_05_11___11_45_27_safetensors-vllm ; CUDA_VISIBLE_DEVICES=0 debug_env_file=/dev/shm/like/ipc.vllm.1.json lm_eval --model vllm --model_args '{"pretrained": "/data/like/hf-models/llama3.1-70B-strip-layers", "quantization": "simo", "hf_overrides": {"quantization_config_file": "/data/like/package/simo_conda_sglang/simo/extensions/sglang_simo/example/simo_quantization_config/online_quantization/quant_config_w4a4_mxfp-copy-from-xuhaifeng.json"}, "tensor_parallel_size": 1, "dtype": "auto", "gpu_memory_utilization": 0.6, "enforce_eager": true, "max_num_seqs": 1}' --tasks wikitext --batch_size 1 > templ/vllm.wikitext.l2norm.lm-eval-mxfp4.log.`nowstr.sh` 2>&1 &
```

你原始命令里还有一个 shell 写法风险：

```bash
CUDA_VISIBLE_DEVICES=0
debug_env_file=/dev/shm/like/ipc.vllm.1.json lm_eval ...
```

如果真的换行执行，`CUDA_VISIBLE_DEVICES=0` 只是设置了当前 shell 变量，不一定 export 给后面的 `lm_eval` 进程。更稳妥的是写在同一条命令前面：

```bash
CUDA_VISIBLE_DEVICES=0 debug_env_file=/dev/shm/like/ipc.vllm.1.json lm_eval ...
```

或者显式：

```bash
export CUDA_VISIBLE_DEVICES=0
debug_env_file=/dev/shm/like/ipc.vllm.1.json lm_eval ...
```

关于“一个 wikitext 数据”的精确定义也要注意：wikitext 的 `loglikelihood_rolling` 会把一条文本按 `max_length` 切成一个或多个 rolling windows。`--batch_size 1` 保证的是一次处理 1 个 rolling window；如果某条 wikitext 文本很长，它仍然可能拆成多个 window 顺序处理。对于短文本，一条数据通常就是一个 window。

如果你还想固定请求顺序，保留：

```bash
--batch_size 1
```

并且不要使用 `--batch_size auto`、不要开 `data_parallel_size > 1`。vLLM adapter 在 batch 内有按长度排序的逻辑，但 batch size 为 1 时排序不会改变实际请求顺序。

## 为什么 SGLang wikitext 跑了 124 次 forward，而 vLLM 只跑了 62 次

先说明 counter 的读法：

```python
array([61], dtype=uint64)
```

表示实际执行了 62 次，因为第一次创建 counter 文件时写入的是 0，之后每次才 `+= 1`。同理，SGLang 之前的：

```python
array([123], dtype=uint64)
```

表示实际执行了 124 次。

这次 vLLM 的 lm-eval adapter 在 `generate=False` 时构造的是：

```python
# /data/like/miniconda3/envs/simo_vllm/lib/python3.12/site-packages/lm_eval/models/vllm_causallms.py
SamplingParams(
    temperature=0,
    prompt_logprobs=1,
    max_tokens=1,
    detokenize=False,
)
```

然后调用：

```python
self.model.generate(
    [TokensPrompt(prompt_token_ids=request) for request in requests],
    sampling_params=sampling_params,
)
```

`prompt_logprobs=1` 是关键。lm-eval 计算 wikitext PPL 需要的是 prompt token logprob，也就是输入 token 的 logprob，不需要真正使用生成出来的那个新 token。vLLM V1 的 `GPUModelRunner.execute_model()` 会在同一次执行里完成：

```text
1. prefill prompt
2. 用 prompt hidden states 计算 prompt_logprobs
3. 用最后一个位置的 logits sample 出 max_tokens=1 的输出 token
4. bookkeeping，判断请求 finished
```

所以 vLLM 不是“没有生成 token”，而是“第一个生成 token 可以直接由 prefill 的最后一个 logits 得到”，不需要再单独跑一次 decode forward。你的 vLLM 日志也能看到每个请求的 `execute_model input_ids` 是完整 prompt token 序列，因此 62 条 wikitext window 对应 62 次 `execute_model`。

SGLang lm-eval adapter 在 `generate=False` 时构造的是：

```python
# /data/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/lm_eval/models/sglang_causallms.py
sampling_params.update(
    {
        "temperature": 0,
        "max_new_tokens": 1,
    }
)

self.model.generate(
    input_ids=requests,
    sampling_params=sampling_params,
    return_logprob=True,
    top_logprobs_num=2,
    logprob_start_len=0,
)
```

按语义，SGLang 的 `max_new_tokens=1` 和 vLLM 的 `max_tokens=1` 都是“最多生成 1 个输出 token”。差异不是参数语义本身，而是 SGLang 默认启用了 overlap scheduler：

```python
# /data/like/package/sglang_kernel_src/python/sglang/srt/managers/scheduler.py
self.enable_overlap = not server_args.disable_overlap_schedule
```

你的 SGLang 日志里可以看到：

```text
disable_overlap_schedule=False
```

所以 SGLang 实际走的是 `event_loop_overlap()`。

overlap scheduler 的关键行为是：它会把上一个 prefill batch 先合入 `running_batch`，然后在处理上一个 prefill 结果之前，就可能先调度下一轮 decode。相关代码路径是：

```python
# /data/like/package/sglang_kernel_src/python/sglang/srt/managers/scheduler.py
def event_loop_overlap(self):
    ...
    batch = self.get_next_batch_to_run()
    ...
    if batch:
        batch_result = self.run_batch(batch)
        self.result_queue.append((batch.copy(), batch_result))
    ...
    if self.last_batch:
        pop_and_process()
```

也就是说，当前 batch 的结果不是立刻处理，而是先进 `result_queue`，下一轮才 `pop_and_process()`。

`get_next_batch_to_run()` 里面又会先把上一次的 extend/prefill batch 合入 running batch：

```python
if self.last_batch and self.last_batch.forward_mode.is_extend():
    ...
    if not self.last_batch.is_empty() and not self.last_batch.is_prefill_only:
        if self.running_batch.is_empty():
            self.running_batch = self.last_batch
        else:
            self.running_batch.merge_batch(self.last_batch)
```

因为 lm-eval adapter 传的是：

```text
max_new_tokens=1
```

所以这个请求不是 prefill-only：

```python
# /data/like/package/sglang_kernel_src/python/sglang/srt/managers/schedule_batch.py
def is_prefill_only(self) -> bool:
    return self.sampling_params.max_new_tokens == 0 and spec_alg is None
```

于是 SGLang 会把它放进 `running_batch`，并准备 decode：

```python
batch.prepare_for_decode()
```

这就解释了你看到的现象：

```text
第 N 轮：SGLang 跑 prefill forward
第 N+1 轮：在处理 prefill 结果前，overlap scheduler 已经把它作为 running decode batch 又跑了一次 decode forward
之后处理 prefill 结果，发现 output_ids 已经达到 max_new_tokens=1，request finished
再处理 decode 结果时，发现 req.finished()，这个 decode 结果被跳过/丢弃
```

SGLang 的代码里也有这个注释：

```python
# /data/like/package/sglang_kernel_src/python/sglang/srt/managers/schedule_batch.py
# If overlap schedule, we schedule one decode batch ahead so this gets called twice.
```

因此，SGLang 的 124 次 forward 不是因为 lm-eval 给了 124 条请求，也不是因为 `max_new_tokens=1` 在 SGLang 中表示“prefill + 1 次 decode 才算生成 1 token”。更准确地说：

```text
SGLang 默认 overlap scheduler 会提前调度一轮 decode。
max_new_tokens=1 的请求在 prefill 后本来已经可以 finished，
但由于 prefill 结果延后处理，decode 已经被提前发射到 GPU。
这个 decode forward 做了计算，但结果不会成为有效输出。
```

vLLM 这边没有观察到这个“一步提前调度”的行为；它在一次 `GPUModelRunner.execute_model()` 里完成 prefill、prompt_logprobs、sample、bookkeeping，所以 62 个 wikitext window 就是 62 次模型执行。

如果你想让 SGLang 在这个测试里也避免多跑那一次 decode，可以在 SGLang 的 lm-eval `--model_args` 顶层加：

```json
"disable_overlap_schedule": true
```

例如：

```bash
CUDA_VISIBLE_DEVICES=0 debug_env_file=/dev/shm/like/ipc.sglang.1.json SIMO_SGLANG_REGISTER=1 lm-eval --model sglang --model_args '{"pretrained": "/data/like/hf-models/llama3.1-70B-strip-layers/", "add_bos_token": true, "quantization": "simo", "json_model_override_args": "{\"quantization_config_file\": \"/data/like/package/simo_conda_sglang/simo/extensions/sglang_simo/example/simo_quantization_config/online_quantization/quant_config_w4a4_mxfp-copy-from-xuhaifeng.json\"}", "tp_size": 1, "dtype": "auto", "mem_fraction_static": 0.5, "skip_server_warmup": true, "disable_cuda_graph": true, "attention_backend": "fa3", "max_running_requests": 1, "watchdog_timeout": 2592000, "disable_overlap_schedule": true}' --tasks wikitext --batch_size 1 > templ/sglang.wikitext.l2norm.lm-eval-mxfp4.log.`nowstr.sh` 2>&1 &
```

如果目标只是算 prompt token logprob/PPL，而不是测试生成路径，还可以考虑把 SGLang lm-eval adapter 里 `generate=False` 的：

```python
"max_new_tokens": 1
```

改成：

```python
"max_new_tokens": 0
```

因为 SGLang 支持 `max_new_tokens=0, return_logprob=True, logprob_start_len=0` 的 prefill-only logprob 路径；代码里 `Req.is_prefill_only` 正是用 `max_new_tokens == 0` 判断的。这样 SGLang 会更接近“只算 prompt logprobs”的语义，也能避免为了一个被 lm-eval 忽略的输出 token 走生成调度路径。

建议排查顺序：

```text
1. 先只加 "disable_overlap_schedule": true，保持 max_new_tokens=1 不变。
   预期 SGLang forward 次数从 124 接近 62。

2. 如果只关心 PPL，再把 SGLang adapter 的 max_new_tokens 改成 0。
   这会走 prefill-only logprob 路径，更符合 wikitext loglikelihood_rolling 的用途。
```

一句话总结：SGLang 和 vLLM 对 `max_new_tokens/max_tokens=1` 的语义并没有本质不同，都是最多生成 1 个 token；你看到的 124 vs 62 主要来自 SGLang 默认 overlap scheduler 的“一步提前 decode”，而不是 lm-eval 多发了请求。
## DeepSeek `self.q_lora_rank is not None` 什么时候为 true

在 `/data/like/package/sglang_kernel_src/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:94` 的 `DeepseekMHAForwardMixin.forward_normal_prepare()` 里，`/data/like/package/sglang_kernel_src/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:101` 的判断：

```python
if self.q_lora_rank is not None:
```

为 true 的条件是：当前 DeepSeek 模型配置里存在 `q_lora_rank`，并且它的值不是 Python `None`。通常也就是 HuggingFace `config.json` 里有类似：

```json
"q_lora_rank": 1536
```

这样的整数值。

这个 `q_lora_rank` 不是运行时额外加载的 LoRA adapter，而是 DeepSeek MLA attention 结构里的 query 低秩投影配置。它决定 Q 投影走哪条实现路径。

来源链路如下：

`/data/like/package/sglang_kernel_src/python/sglang/srt/models/deepseek_v2.py:2259` 的 `DeepseekV2DecoderLayer.__init__()` 创建 attention：

```python
self.self_attn = DeepseekV2AttentionMLA(
    ...
    q_lora_rank=(
        config.q_lora_rank if hasattr(config, "q_lora_rank") else None
    ),
    kv_lora_rank=config.kv_lora_rank,
    ...
)
```

也就是说，`self.q_lora_rank` 来自模型 config。若 config 没有 `q_lora_rank` 字段，则传 `None`；若字段存在但值是 `null`，加载到 Python 后也是 `None`。

`/data/like/package/sglang_kernel_src/python/sglang/srt/models/deepseek_v2.py:1069` 的 `DeepseekV2AttentionMLA.__init__()` 里，`/data/like/package/sglang_kernel_src/python/sglang/srt/models/deepseek_v2.py:1096` 直接保存：

```python
self.q_lora_rank = q_lora_rank
```

然后 `/data/like/package/sglang_kernel_src/python/sglang/srt/models/deepseek_v2.py:1121` 的 `DeepseekV2AttentionMLA.__init__()` 根据同一个条件建不同的层：

如果 `self.q_lora_rank is not None`，会创建：

```python
self.fused_qkv_a_proj_with_mqa
self.q_a_layernorm
self.q_b_proj
```

这对应 `/data/like/package/sglang_kernel_src/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:101` 到 `/data/like/package/sglang_kernel_src/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:178` 的路径：先取 fused latent，再把 Q latent 经过 `q_a_layernorm` 和 `q_b_proj` 还原成实际 Q。

如果 `self.q_lora_rank is None`，会创建普通 Q 投影：

```python
self.q_proj
self.kv_a_proj_with_mqa
```

这对应 `/data/like/package/sglang_kernel_src/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:180` 到 `/data/like/package/sglang_kernel_src/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:184` 的路径：直接用 `self.q_proj(hidden_states)` 计算 Q，同时单独用 `self.kv_a_proj_with_mqa(hidden_states)` 计算 KV latent。

另外，`/data/like/package/sglang_kernel_src/python/sglang/srt/models/deepseek_v2.py:2785` 的 `DeepseekV2ForCausalLM.__init__()` 也使用同样条件。`/data/like/package/sglang_kernel_src/python/sglang/srt/models/deepseek_v2.py:2795` 到 `/data/like/package/sglang_kernel_src/python/sglang/srt/models/deepseek_v2.py:2797`：

```python
self.fuse_qkv_a_proj = (
    hasattr(config, "q_lora_rank") and config.q_lora_rank is not None
)
```

如果 `q_lora_rank` 非 None，权重加载时会把 `q_a_proj` 和 `kv_a_proj_with_mqa` 按输出维度融合成 `fused_qkv_a_proj_with_mqa`；如果是 None，则不会启用这个 Q/KV A projection fusion。

对于模型 `/data/like/hf-models/DeepSeek-V2-Lite-Chat-16B_A2.4B/`，结论是：这个 if 不会为 true。

原因是 `/data/like/hf-models/DeepSeek-V2-Lite-Chat-16B_A2.4B/config.json:34` 是：

```json
"q_lora_rank": null
```

JSON 的 `null` 加载到 Python config 后就是 `None`。所以在这个模型上，`/data/like/package/sglang_kernel_src/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:101` 的 `DeepseekMHAForwardMixin.forward_normal_prepare()` 判断结果是 false，会走 `/data/like/package/sglang_kernel_src/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:180` 的普通 `q_proj` 路径。

注意：这个模型仍然有 `/data/like/hf-models/DeepSeek-V2-Lite-Chat-16B_A2.4B/config.json:20`：

```json
"kv_lora_rank": 512
```

所以它仍然使用 KV 的 low-rank latent 路径；只是 Q 没有使用 `q_lora_rank` 对应的低秩 Q 路径。

## vLLM SIMO MLA KV cache 量化写入最终调用的 Triton kernel

这次 server 命令指定了：

```bash
--attention-config '{"backend": "TRITON_MLA"}'
```

日志 `/share/users/like/package/h100/package/simo_conda_sglang/temp/vllm.serve.log.2026_05_15___16_09_44` 里也能看到使用的是 `AttentionBackendEnum.TRITON_MLA`，并且 KV cache quant spec 是 `QuantizeSpecMX(dtype='mxfp8_e4m3', scale_mode='e8m0_floor', observer_mode='abs_max', block_size=32, group_size=32, axis=-1, is_dynamic=False)`。

因此，对 DeepSeek-V2-Lite-Chat-16B_A2.4B 这个 MLA 模型，`simo/extensions/vllm_simo/v1/attention/backends/simo_mla.py:154` 的 `SIMOMLAImpl.do_kv_cache_update()` 最终调用的 Triton kernel 是：

```text
simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:25
concat_and_cache_mla_kernel()
```

不是 GQA 路径里的 `reshape_and_cache_kernel_flash()`。`simo/extensions/vllm_simo/v1/attention/backends/simo_gqa.py` 是 `TRITON_ATTN`/GQA 后端使用的；你这条命令是 `TRITON_MLA`，入口在 `simo_mla.py`。

调用链如下。

`simo/extensions/vllm_simo/v1/attention/backends/simo_mla.py:154` 的 `SIMOMLAImpl.do_kv_cache_update()`：

```python
if self._kv_cache_quant_spec is not None:
    from simo.extensions.vllm_simo.v1.attention.ops.triton_concat_and_cache_mla import (
        concat_and_cache_mla,
    )
    ...
    concat_and_cache_mla(
        kv_c_normed,
        k_pe.squeeze(1),
        kv_cache,
        slot_mapping.flatten(),
        kv_cache_quant_spec=self._kv_cache_quant_spec,
    )
```

其中 `simo/extensions/vllm_simo/v1/attention/backends/simo_mla.py:166` 到 `simo/extensions/vllm_simo/v1/attention/backends/simo_mla.py:169` 导入 `concat_and_cache_mla()`；`simo/extensions/vllm_simo/v1/attention/backends/simo_mla.py:171` 到 `simo/extensions/vllm_simo/v1/attention/backends/simo_mla.py:177` 在配置要求时对 `kv_c_normed` 和 `k_pe` 做 Hadamard transform；`simo/extensions/vllm_simo/v1/attention/backends/simo_mla.py:180` 到 `simo/extensions/vllm_simo/v1/attention/backends/simo_mla.py:186` 调用 `concat_and_cache_mla()`。

`simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:183` 的 `concat_and_cache_mla()` 是 Python wrapper。它做几件事：

1. `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:203` 到 `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:212` 根据 `kv_cache_quant_spec` 判断量化格式。你当前配置是 MX 格式 `mxfp8_e4m3`，所以会走 `QuantizeSpecMX` 分支，得到 `mx_format_id`、`observer_mode`、`scale_rounding_mode`，`tile_size` 是 `kv_cache_quant_spec.block_size`，也就是 32。

2. `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:214` 到 `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:217` 要求 `kv_lora_rank` 和 `pe_dim` 都能被 `tile_size` 整除。DeepSeek MLA cache 里写入的是 `[kv_c, k_pe]`，它会按 32 个元素一组量化。

3. `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:219` 到 `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:225` 用 meta tensor 调 `get_downcast_kernel()`，计算每个 tile 量化后需要多少 packed bytes 和 scale bytes。

4. `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:227` 到 `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:232` 计算 cache entry 的布局大小：先放 `KV_C packed`，再放 `KV_C scales`，然后放 `K_PE packed`，最后放 `K_PE scales`。

5. `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:235` 把 `kv_c` 和 `k_pe` 拼成 `concatenated_kv = torch.cat([kv_c.contiguous(), k_pe.contiguous()], dim=1)`。

6. `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:241` 设置 Triton grid 为 `(num_tokens, total_tiles)`。

7. `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:243` 到 `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:268` 启动最终 kernel：

```python
concat_and_cache_mla_kernel[grid](...)
```

最终 kernel 的核心实现是 `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:25` 的 `concat_and_cache_mla_kernel()`。

`simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:56` 到 `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:59` 的注释说明了 kernel 粒度：grid 是 `(num_tokens, num_tiles_per_entry)`，每个 Triton program 处理一个 token 的一个量化 tile。

`simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:64` 到 `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:65`：

```python
token_idx = tl.program_id(0)
tile_idx = tl.program_id(1)
```

也就是第 0 维选 token，第 1 维选当前 token 内的第几个 32 元素 tile。

`simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:67` 到 `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:69` 读取 `slot_mapping[token_idx]`。如果 `slot_idx < 0`，这个 token 不需要写 cache，kernel 直接 return。

`simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:74` 到 `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:76` 从拼接后的 `[kv_c, k_pe]` 中加载一个 tile：

```python
head_size_offs = tile_idx * TILE_SIZE + tl.arange(0, TILE_SIZE)
kv_load = tl.load(src_ptr + head_size_offs, mask=head_size_offs < concat_dim)
```

当前配置 `TILE_SIZE=32`，所以每个 program 一次处理 32 个 bf16/fp16 元素。

`simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:79` 到 `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:81` 把一维 `slot_idx` 映射成 paged KV cache 里的 block 和 block 内 offset：

```python
block_idx = slot_idx // block_size
block_offset = slot_idx % block_size
slot_base = block_idx * stride_cache_0 + block_offset * stride_cache_1
```

`simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:83` 到 `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:93` 判断这个 tile 属于 `KV_C` 还是 `K_PE`。如果 `tile_idx * TILE_SIZE < kv_lora_rank`，写到 `KV_C` 区域；否则写到 `K_PE` 区域。两段区域的 packed 数据和 scale 数据是分开存的。

对于你当前的 `mxfp8_e4m3` 配置，`simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:95` 的 `if MX_FORMAT_ID > 0` 为 true，所以走 MX quant path。

`simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:97` 到 `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:104` 调用：

```python
_compute_and_pack_mxfmt(...)
```

这里会对当前 32 元素 tile 计算 shared scale，并把数据量化/pack 成 MX 格式。对 `mxfp8_e4m3` 来说，每个元素最终是 FP8 E4M3 表示，scale 默认是 E8M0 floor，observer 默认是 abs max。

`simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:133` 到 `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:143` 是非 FP6、非 4bit 的存储路径。`mxfp8_e4m3` 会走这里：把 packed FP8 数据 bitcast 成 `uint8`，写入 `kv_cache` 的 packed 区。

`simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:145` 到 `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:148` 写 scale。MX 格式每个 tile 存 1 byte scale：

```python
s_u8 = tl.reshape(kv_scale, (1,)).to(tl.uint8, bitcast=True)
tl.store(kv_cache_ptr + slot_base + scale_write_offset + scale_offs, s_u8)
```

`simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:150` 到 `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:180` 是另一条 per-group FP8/INT8 路径；当前 `quant_config_kvquant_mxfp8.json` 不走这条，因为它是 `mxfp8_e4m3`，属于 MX spec。

量化配置从这里进入：

`simo/extensions/vllm_simo/quantization/quantization_config.py:570` 的 `SIMOConfig.get_quant_method()` 检查 attention layer 和 `kv_cache_quant_algo`。`simo/extensions/vllm_simo/quantization/quantization_config.py:571` 到 `simo/extensions/vllm_simo/quantization/quantization_config.py:588` 会对 `MLAAttention` 检查后端必须是 `TRITON_MLA`，然后返回 `SIMOKVCacheMethod`。

`simo/extensions/vllm_simo/quantization/quantization_method.py:853` 的 `SIMOKVCacheMethod.create_weights()` 把 `kv_cache_quant_spec`、`kv_cache_downcast_kernel`、Hadamard 参数和 packed/scale head size 挂到 layer 上。

`simo/extensions/vllm_simo/quantization/quantization_method.py:874` 的 `SIMOKVCacheMethod.process_weights_after_loading()` 在 layer impl 创建之后，把 `self.kv_cache_spec` 写到 `layer.impl._kv_cache_quant_spec`，所以 `simo/extensions/vllm_simo/v1/attention/backends/simo_mla.py:166` 的 `if self._kv_cache_quant_spec is not None` 会为 true。

总结：这条 DeepSeek + `TRITON_MLA` + SIMO KV cache quant 的写 cache 路径是：

```text
simo/extensions/vllm_simo/v1/attention/backends/simo_mla.py:154
SIMOMLAImpl.do_kv_cache_update()
  -> simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:183
     concat_and_cache_mla()
       -> simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:25
          concat_and_cache_mla_kernel()
```

最终 kernel 做的事情是：对每个 token 的 `[kv_c, k_pe]` 按 32 元素 tile 切分，分别量化成 MXFP8 E4M3 packed bytes 和 1-byte scale，再按照 paged KV cache 的 `slot_mapping` 写入 `kv_cache[block_idx, block_offset, :]` 的对应 packed/scale 区域。

## vLLM SIMO MLA `kv_cache.shape=[21361, 16, 594]` 的来源

在 `vllm/model_executor/layers/attention/mla_attention.py:871` 的 `unified_mla_kv_cache_update()` 里：

```python
kv_cache = attn_layer.kv_cache[forward_context.virtual_engine]
```

你在 pudb 里看到：

```python
kv_cache.shape = torch.Size([21361, 16, 594])
```

这个 shape 对 SIMO + `TRITON_MLA` 来说含义是：

```text
[num_blocks, block_size, quantized_mla_cache_entry_size]
= [21361, 16, 594]
```

中间的 `16` 是 vLLM KV cache 的 page/block size，也就是每个 KV cache block 里有 16 个 token slot。它不是 head 数，也不是 MXFP8 的量化 block size。这里有两个容易混淆的 block size：

```text
vLLM KV cache block_size = 16      # 每个 paged KV block 里放多少 token
SIMO MXFP8 quant block_size = 32   # 每 32 个元素共享一个 MX scale
```

### `block_size=16` 从哪里来

你的 server 命令没有显式传 `--block-size`。在 CUDA 平台上，如果用户没有指定，vLLM 会把 cache block size 设成 16。

`vllm/platforms/cuda.py:168` 的 `CudaPlatform.check_and_update_config()` 里，`vllm/platforms/cuda.py:178` 到 `vllm/platforms/cuda.py:179`：

```python
if cache_config and cache_config.block_size is None:
    cache_config.block_size = 16
```

这个值随后进入 `MLAAttention`。

`vllm/model_executor/layers/attention/mla_attention.py:281` 的 `MLAAttention.__init__()` 里，`vllm/model_executor/layers/attention/mla_attention.py:314` 到 `vllm/model_executor/layers/attention/mla_attention.py:320`：

```python
if cache_config is not None:
    kv_cache_dtype = cache_config.cache_dtype
    block_size = cache_config.block_size
else:
    kv_cache_dtype = "auto"
    block_size = 16
```

所以 attention backend 初始化时拿到的 block size 是 16。

### MLA cache shape 是在哪里定义的

原生 MLA backend 的 shape 规则在 `vllm/model_executor/layers/attention/mla_attention.py:1073` 的 `MLACommonBackend.get_kv_cache_shape()`：

```python
return (num_blocks, block_size, head_size)
```

SIMO 覆盖了 `TRITON_MLA` backend。你的命令使用的是：

```bash
--attention-config '{"backend": "TRITON_MLA"}'
```

SIMO 注册的 MLA backend 在 `simo/extensions/vllm_simo/v1/attention/backends/simo_mla.py:41` 的 `SIMOMLABackend`。它自己的 shape 规则在 `simo/extensions/vllm_simo/v1/attention/backends/simo_mla.py:63` 的 `SIMOMLABackend.get_kv_cache_shape()`，`simo/extensions/vllm_simo/v1/attention/backends/simo_mla.py:71` 到 `simo/extensions/vllm_simo/v1/attention/backends/simo_mla.py:72`：

```python
# MLA: [num_blocks, block_size, head_size] (num_kv_heads=1 assumed)
return (num_blocks, block_size, head_size)
```

所以第二维 `16` 是由这里的 `block_size` 形参进入最终 tensor shape 的。

### `594` 是哪里来的

普通 vLLM MLA 里，`vllm/model_executor/layers/attention/mla_attention.py:281` 的 `MLAAttention.__init__()` 在 `vllm/model_executor/layers/attention/mla_attention.py:307` 设置：

```python
self.head_size = kv_lora_rank + qk_rope_head_dim
```

DeepSeek-V2-Lite-Chat-16B_A2.4B 的 MLA cache 原始 entry 是：

```text
kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576
```

但是你启用了 SIMO KV cache quant，所以 cache 里不是直接存 576 个 bf16，而是存 `uint8` packed data + scale。

SIMO 对 `MLAAttention.get_kv_cache_spec()` 做了 patch。入口在 `simo/extensions/vllm_simo/model_executor/layers/attention/attention.py:13` 的 `get_kv_cache_spec()`，`simo/extensions/vllm_simo/model_executor/layers/attention/attention.py:20` 到 `simo/extensions/vllm_simo/model_executor/layers/attention/attention.py:24`：

```python
if hasattr(self.attn_backend, "make_kv_cache_spec"):
    spec = self.attn_backend.make_kv_cache_spec(self, vllm_config)
    if spec is not None:
        return spec
return original_method(self, vllm_config)
```

因此 SIMO MLA 会走 `simo/extensions/vllm_simo/v1/attention/backends/simo_mla.py:75` 的 `SIMOMLABackend.make_kv_cache_spec()`。关键代码在 `simo/extensions/vllm_simo/v1/attention/backends/simo_mla.py:80` 到 `simo/extensions/vllm_simo/v1/attention/backends/simo_mla.py:88`：

```python
x_q, s = layer.kv_cache_downcast_kernel(torch.randn(128, layer.head_size, device="meta"))
packed = x_q.contiguous().view(torch.uint8).shape[-1]
scale = s.contiguous().view(torch.uint8).shape[-1]

return MLAAttentionSpec(
    block_size=vllm_config.cache_config.block_size,
    num_kv_heads=1,
    head_size=packed + scale,
    dtype=torch.uint8,
)
```

当前配置 `quant_config_kvquant_mxfp8.json` 里是 `mxfp8_e4m3`。SIMO 在 `simo/extensions/vllm_simo/quantization/quantization_config.py:63` 的 `parse_quantize_spec()` 里解析 quant spec；`simo/quantization/config.py:411` 的 `QuantizeSpecMX` 里，`simo/quantization/config.py:448` 默认：

```python
block_size: int = Field(default=32, ...)
```

所以 MXFP8 对 576 个元素按 32 个元素一组量化：

```text
packed bytes = 576
scale bytes  = 576 / 32 = 18
total        = 576 + 18 = 594
```

这就是第三维 `594` 的来源。

### `21361` 是哪里来的

第一维 `21361` 是 KV cache block 数，由 profiling 后可用 KV cache 显存算出来，不是模型 config 里的固定值。

日志里有：

```text
Available KV cache memory: 5.11 GiB
GPU KV cache size: 341,776 tokens
```

而：

```text
341,776 / 16 = 21,361
```

所以最终 `num_blocks=21361`。

计算链路如下。

`vllm/v1/engine/core.py:241` 的 `EngineCore._initialize_kv_caches()` 先在 `vllm/v1/engine/core.py:247` 调 `self.model_executor.get_kv_cache_specs()` 收集每层 KV cache spec，然后在 `vllm/v1/engine/core.py:261` 调 `self.model_executor.determine_available_memory()` 做显存 profiling，最后在 `vllm/v1/engine/core.py:272` 到 `vllm/v1/engine/core.py:274` 调：

```python
kv_cache_configs = get_kv_cache_configs(
    vllm_config, kv_cache_specs, available_gpu_memory
)
```

`vllm/v1/core/kv_cache_utils.py:1501` 的 `get_kv_cache_configs()` 会合并各 worker 的 spec、分组、检查显存，然后在 `vllm/v1/core/kv_cache_utils.py:1586` 到 `vllm/v1/core/kv_cache_utils.py:1589` 调 `get_kv_cache_config_from_groups()`。

`vllm/v1/core/kv_cache_utils.py:1078` 的 `get_kv_cache_config_from_groups()` 在 general case 里，`vllm/v1/core/kv_cache_utils.py:1131` 到 `vllm/v1/core/kv_cache_utils.py:1137` 用 page size 和可用显存算 block 数：

```python
page_size = get_uniform_page_size(...)
num_blocks = get_num_blocks(
    vllm_config, group_size, available_memory, page_size
)
```

具体公式在 `vllm/v1/core/kv_cache_utils.py:836` 的 `get_num_blocks()`，`vllm/v1/core/kv_cache_utils.py:848`：

```python
num_blocks = int(available_memory // page_size // num_layers)
```

对你这个 case，每层每个 KV block 的 page size 是：

```text
block_size * num_kv_heads * head_size * dtype_size
= 16 * 1 * 594 * 1
= 9504 bytes
```

这个公式来自 `vllm/v1/kv_cache_interface.py:191` 的 `MLAAttentionSpec`，`vllm/v1/kv_cache_interface.py:196` 到 `vllm/v1/kv_cache_interface.py:206` 的 `MLAAttentionSpec.real_page_size_bytes()`：

```python
return (
    self.block_size
    * self.num_kv_heads
    * self.head_size
    * get_dtype_size(self.dtype)
)
```

SIMO spec 里 `dtype=torch.uint8`，所以 `dtype_size=1`。

### 最终 tensor 是在哪里真正 reshape 成 `[21361, 16, 594]` 的

真正分配 raw buffer 的地方是 `vllm/v1/worker/gpu_model_runner.py:5826` 的 `GPUModelRunner._allocate_kv_cache_tensors()`，`vllm/v1/worker/gpu_model_runner.py:5840` 到 `vllm/v1/worker/gpu_model_runner.py:5845`：

```python
tensor = torch.zeros(
    kv_cache_tensor.size, dtype=torch.int8, device=self.device
)
for layer_name in kv_cache_tensor.shared_by:
    kv_cache_raw_tensors[layer_name] = tensor
```

这一步只是按字节数分配一维 raw tensor。

真正把 raw tensor view 成 MLA backend 需要的 shape，是 `vllm/v1/worker/gpu_model_runner.py:5867` 的 `GPUModelRunner._reshape_kv_cache_tensors()`。

关键代码在 `vllm/v1/worker/gpu_model_runner.py:5898` 到 `vllm/v1/worker/gpu_model_runner.py:5913`：

```python
num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes
kernel_block_size = kernel_block_sizes[group.kv_cache_group_id]
kv_cache_shape = attn_backend.get_kv_cache_shape(
    kernel_num_blocks,
    kernel_block_size,
    kv_cache_spec.num_kv_heads,
    kv_cache_spec.head_size,
    cache_dtype_str=self.cache_config.cache_dtype,
)
```

对 SIMO MLA 来说，`attn_backend.get_kv_cache_shape()` 就是 `simo/extensions/vllm_simo/v1/attention/backends/simo_mla.py:63` 的 `SIMOMLABackend.get_kv_cache_shape()`，返回：

```text
(21361, 16, 594)
```

随后 `vllm/v1/worker/gpu_model_runner.py:5933` 到 `vllm/v1/worker/gpu_model_runner.py:5938`：

```python
kv_caches[layer_name] = (
    kv_cache_raw_tensors[layer_name]
    .view(dtype)
    .view(kv_cache_shape)
    .permute(*inv_order)
)
```

这里把 raw buffer 真正变成你在 `unified_mla_kv_cache_update()` 里看到的 `torch.Size([21361, 16, 594])`。

最后 `vllm/v1/worker/gpu_model_runner.py:5998` 的 `GPUModelRunner.initialize_kv_cache_tensors()` 在 `vllm/v1/worker/gpu_model_runner.py:6045` 到 `vllm/v1/worker/gpu_model_runner.py:6050` 调 `bind_kv_cache()`，把这个 tensor 绑到 attention layer 上。

绑定逻辑在 `vllm/v1/worker/utils.py:271` 的 `bind_kv_cache()`。`vllm/v1/worker/utils.py:325` 到 `vllm/v1/worker/utils.py:328`：

```python
for layer_name, kv_cache in kv_caches.items():
    forward_context[layer_name].kv_cache = [kv_cache]
```

所以 `vllm/model_executor/layers/attention/mla_attention.py:871` 的 `unified_mla_kv_cache_update()` 在 `vllm/model_executor/layers/attention/mla_attention.py:884` 取到的：

```python
attn_layer.kv_cache[forward_context.virtual_engine]
```

就是前面 reshape 并 bind 进去的 `[21361, 16, 594]`。

完整链路可以概括为：

```text
vllm/platforms/cuda.py:168
CudaPlatform.check_and_update_config()
  -> cache_config.block_size = 16

vllm/model_executor/layers/attention/mla_attention.py:281
MLAAttention.__init__()
  -> self.head_size = kv_lora_rank + qk_rope_head_dim = 576

simo/extensions/vllm_simo/model_executor/layers/attention/attention.py:13
get_kv_cache_spec()
  -> delegate to SIMOMLABackend.make_kv_cache_spec()

simo/extensions/vllm_simo/v1/attention/backends/simo_mla.py:75
SIMOMLABackend.make_kv_cache_spec()
  -> MLAAttentionSpec(block_size=16, head_size=594, dtype=torch.uint8)

vllm/v1/core/kv_cache_utils.py:1501
get_kv_cache_configs()
  -> compute num_blocks from available KV memory
  -> num_blocks = 21361

vllm/v1/worker/gpu_model_runner.py:5867
GPUModelRunner._reshape_kv_cache_tensors()
  -> SIMOMLABackend.get_kv_cache_shape(21361, 16, 1, 594)
  -> shape = (21361, 16, 594)

vllm/v1/worker/utils.py:271
bind_kv_cache()
  -> forward_context[layer_name].kv_cache = [kv_cache]

vllm/model_executor/layers/attention/mla_attention.py:871
unified_mla_kv_cache_update()
  -> kv_cache = attn_layer.kv_cache[forward_context.virtual_engine]
```

## SIMO MLA `concat_and_cache_mla_kernel()` 写 KV cache 时 `kv_c` 和 `k_pe` 的内部布局

结论：每个 token 内部的 KV cache entry 布局是：

```text
[kv_c mx element, kv_c mx scale, k_pe mx element, k_pe mx scale]
```

更准确地说，是按区域连续排布：

```text
[KV_C packed bytes | KV_C scale bytes | K_PE packed bytes | K_PE scale bytes]
```

不是：

```text
[kv_c mx element, k_pe mx element, kv_c mx scale, k_pe mx scale]
```

代码里在 `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:4` 已经直接写了注释：

```python
Cache layout: Tensor-Interleaved [KV_C packed | KV_C scales | K_PE packed | K_PE scales].
```

### Python wrapper 先 concat，但 cache 写入不是按 concat 原始顺序直接存

`simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:183` 的 `concat_and_cache_mla()` 里，`simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:235` 确实先做了：

```python
concatenated_kv = torch.cat([kv_c.contiguous(), k_pe.contiguous()], dim=1)
```

这只是给 kernel 提供连续输入。真正写入 KV cache 的布局由 `concat_and_cache_mla_kernel()` 里的 `packed_write_offset` 和 `scale_write_offset` 决定。

### offset 常量的计算

`simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:183` 的 `concat_and_cache_mla()` 里，`simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:227` 到 `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:232`：

```python
kv_c_num_tiles = kv_lora_rank // tile_size
k_pe_num_tiles = pe_dim // tile_size
KV_C_PACKED_BYTES = kv_c_num_tiles * packed_tile_bytes
KV_C_SCALE_BYTES = kv_c_num_tiles * scale_tile_bytes
KV_C_TOTAL_BYTES = KV_C_PACKED_BYTES + KV_C_SCALE_BYTES
K_PE_PACKED_BYTES = k_pe_num_tiles * packed_tile_bytes
```

这些值被传给 `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:25` 的 `concat_and_cache_mla_kernel()`。

### kernel 里如何区分 `kv_c` 和 `k_pe`

`simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:25` 的 `concat_and_cache_mla_kernel()` 里，`simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:64` 到 `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:65`：

```python
token_idx = tl.program_id(0)
tile_idx = tl.program_id(1)
```

每个 Triton program 处理某个 token 的某个 tile。

`simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:83` 到 `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:93` 决定当前 tile 写到哪个区域：

```python
is_kv_c = (tile_idx * TILE_SIZE) < kv_lora_rank
kv_c_num_tiles: tl.constexpr = kv_lora_rank // TILE_SIZE
if is_kv_c:
  local_tile_idx = tile_idx
  packed_write_offset = local_tile_idx * PACKED_TILE_BYTES
  scale_write_offset = KV_C_PACKED_BYTES + local_tile_idx * SCALE_TILE_BYTES
else:
  local_tile_idx = tile_idx - kv_c_num_tiles
  packed_write_offset = KV_C_TOTAL_BYTES + local_tile_idx * PACKED_TILE_BYTES
  scale_write_offset = KV_C_TOTAL_BYTES + K_PE_PACKED_BYTES + local_tile_idx * SCALE_TILE_BYTES
```

这段 offset 计算说明：

```text
KV_C packed 起始 offset = 0
KV_C scale  起始 offset = KV_C_PACKED_BYTES
K_PE packed 起始 offset = KV_C_TOTAL_BYTES
K_PE scale  起始 offset = KV_C_TOTAL_BYTES + K_PE_PACKED_BYTES
```

所以布局就是：

```text
[KV_C packed][KV_C scales][K_PE packed][K_PE scales]
```

### store 的位置

当前配置是 `mxfp8_e4m3`，属于 MX format，所以 `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:95` 的 `if MX_FORMAT_ID > 0` 为 true。

`simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:97` 到 `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:104` 调 `_compute_and_pack_mxfmt()` 得到：

```python
packed_kv, kv_scale = _compute_and_pack_mxfmt(...)
```

对于 `mxfp8_e4m3`，不是 FP6，也不是 4bit，所以走 `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:133` 到 `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:143` 的 packed data 写入：

```python
tl.store(
  kv_cache_ptr + slot_base + packed_write_offset + packed_offs,
  packed_flat,
  mask=packed_offs < PACKED_TILE_BYTES,
)
```

scale 写入在 `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:145` 到 `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:148`：

```python
s_u8 = tl.reshape(kv_scale, (1,)).to(tl.uint8, bitcast=True)
scale_offs = tl.arange(0, 1)
tl.store(kv_cache_ptr + slot_base + scale_write_offset + scale_offs, s_u8)
```

packed data 和 scale data 都用同一套 `packed_write_offset` / `scale_write_offset`，所以 `kv_c` tile 的 scale 一定写在 `KV_C packed` 后面；`k_pe` tile 的 scale 一定写在 `K_PE packed` 后面。

### 用你这个 DeepSeek + MXFP8 配置展开

对 DeepSeek-V2-Lite-Chat-16B_A2.4B：

```text
kv_lora_rank = 512
pe_dim = qk_rope_head_dim = 64
tile_size = 32
```

当前 `mxfp8_e4m3` 下：

```text
packed_tile_bytes = 32
scale_tile_bytes = 1
kv_c_num_tiles = 512 / 32 = 16
k_pe_num_tiles = 64 / 32 = 2
```

所以每个 token 的 594 bytes entry 内部是：

```text
offset [0,   512) : KV_C packed bytes, 16 tiles * 32 bytes
offset [512, 528) : KV_C scales,       16 tiles * 1 byte
offset [528, 592) : K_PE packed bytes, 2 tiles  * 32 bytes
offset [592, 594) : K_PE scales,       2 tiles  * 1 byte
```

也就是：

```text
[KV_C packed 512B | KV_C scale 16B | K_PE packed 64B | K_PE scale 2B]
```

总大小：

```text
512 + 16 + 64 + 2 = 594 bytes
```

这和你前面看到的 `kv_cache.shape = [21361, 16, 594]` 的第三维完全对应。

## DeepseekV2AttentionMLA 的 w_kc 何时填充、调用链路和作用

结论：`w_kc` 不是 checkpoint 里直接以 `w_kc` 名字保存的参数。它在模型初始化时先是 `None`，真正填充发生在 checkpoint 权重加载完成后的 post-load 阶段，由 `kv_b_proj.weight` 切分派生出来。也就是说，它是在 server 启动加载模型期间填充的，在接收请求之前已经准备好。

### 填充 w_kc 的调用链路

在你这条 `python3 -m sglang.launch_server ... --model-path /data/like/hf-models/DeepSeek-V2-Lite-Chat-16B_A2.4B --attention-backend triton ...` 的普通模型加载路径下，调用链路是：

```text
python/sglang/srt/model_loader/loader.py:653 DefaultModelLoader.load_model
  -> python/sglang/srt/model_loader/loader.py:685 DefaultModelLoader.load_weights_and_postprocess
  -> python/sglang/srt/models/deepseek_v2.py:2944 DeepseekV2ForCausalLM.load_weights
  -> python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:96 DeepseekV2WeightLoaderMixin.do_load_weights
  -> python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:416 DeepseekV2WeightLoaderMixin.post_load_weights
```

关键代码点如下：

`python/sglang/srt/models/deepseek_v2.py:1178 DeepseekV2AttentionMLA.__init__` 创建 `kv_b_proj`：

```python
self.kv_b_proj = ColumnParallelLinear(
    self.kv_lora_rank,
    self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
    ...
)
```

`python/sglang/srt/models/deepseek_v2.py:1248 DeepseekV2AttentionMLA.__init__` 初始化：

```python
self.w_kc = None
self.w_vc = None
```

`python/sglang/srt/model_loader/loader.py:685 DefaultModelLoader.load_weights_and_postprocess` 调用：

```python
model.load_weights(weights)
```

`python/sglang/srt/models/deepseek_v2.py:2944 DeepseekV2ForCausalLM.load_weights` 又调用：

```python
self.do_load_weights(weights, is_nextn)
```

`python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:96 DeepseekV2WeightLoaderMixin.do_load_weights` 负责把 checkpoint 权重加载进 `named_parameters()`。普通参数路径在 `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:364 DeepseekV2WeightLoaderMixin.do_load_weights` 到 `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:374 DeepseekV2WeightLoaderMixin.do_load_weights` 调用 `weight_loader/default_weight_loader`，这里会把 checkpoint 里的 `model.layers.*.self_attn.kv_b_proj.weight` 加载到 `self_attn.kv_b_proj.weight`。

等所有异步/同步 weight load 都完成后，`python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:380 DeepseekV2WeightLoaderMixin.do_load_weights` 调用：

```python
self.post_load_weights(is_nextn=is_nextn, weight_names=weight_names)
```

真正填充 `w_kc` 的位置在 `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:416 DeepseekV2WeightLoaderMixin.post_load_weights`。这个函数注释在 `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:421 DeepseekV2WeightLoaderMixin.post_load_weights` 到 `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:426 DeepseekV2WeightLoaderMixin.post_load_weights` 已经写明：它会 post-process `kv_b_proj`，包括把权重切分成 `w_kc` 和 `w_vc`。

具体逻辑是：

`python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:445 DeepseekV2WeightLoaderMixin.post_load_weights` 到 `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:450 DeepseekV2WeightLoaderMixin.post_load_weights` 找到每一层的 `self_attn`。

`python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:465 DeepseekV2WeightLoaderMixin.post_load_weights` 到 `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:466 DeepseekV2WeightLoaderMixin.post_load_weights` 取出：

```python
w = self_attn.kv_b_proj.weight
```

如果是 fp8/int8/AWQ 等量化权重，`python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:452 DeepseekV2WeightLoaderMixin.post_load_weights` 到 `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:570 DeepseekV2WeightLoaderMixin.post_load_weights` 会先做对应 dequant/requant/scale 处理。

然后 `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:572 DeepseekV2WeightLoaderMixin.post_load_weights` 到 `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:574 DeepseekV2WeightLoaderMixin.post_load_weights` 把 `kv_b_proj.weight` 切成 K 部分和 V 部分：

```python
w_kc, w_vc = w.unflatten(
    0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
```

最后 `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:585 DeepseekV2WeightLoaderMixin.post_load_weights` 到 `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:588 DeepseekV2WeightLoaderMixin.post_load_weights` 写入：

```python
self_attn.w_kc = bind_or_assign(
    self_attn.w_kc, w_kc.transpose(1, 2).contiguous().transpose(1, 2)
)
```

如果走 DeepGEMM BMM 分支，写入点是 `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:622 DeepseekV2WeightLoaderMixin.post_load_weights` 到 `python/sglang/srt/models/deepseek_common/deepseek_weight_loader.py:624 DeepseekV2WeightLoaderMixin.post_load_weights`。

对 `/data/like/hf-models/DeepSeek-V2-Lite-Chat-16B_A2.4B/config.json` 这个模型，`num_attention_heads=16`、`kv_lora_rank=512`、`qk_nope_head_dim=128`、`v_head_dim=128`。在 `--tp-size 1` 下，`kv_b_proj.weight` 逻辑上会被拆成：

```text
w_kc: [num_heads, qk_nope_head_dim, kv_lora_rank] = [16, 128, 512]
w_vc: [num_heads, kv_lora_rank, v_head_dim]       = [16, 512, 128]  # 写入前后有 transpose
```

### w_kc 的作用

`w_kc` 是 `kv_b_proj` 里用于生成 non-RoPE key 的那一半 up-projection 权重。它的作用不是单独算一个普通线性层输出，而是在 MLA absorb 路径里把 query 的 non-RoPE 部分投影到压缩 KV latent 空间，从而避免显式 materialize/cache full K。

使用点在 `python/sglang/srt/models/deepseek_v2.py:1525 DeepseekV2AttentionMLA.forward_absorb_prepare`。

`python/sglang/srt/models/deepseek_v2.py:1643 DeepseekV2AttentionMLA.forward_absorb_prepare` 到 `python/sglang/srt/models/deepseek_v2.py:1648 DeepseekV2AttentionMLA.forward_absorb_prepare` 先算：

```python
q = self.q_proj(hidden_states)[0].view(...)
latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
k_nope = latent_cache[..., : self.kv_lora_rank]
k_nope = self.kv_a_layernorm(k_nope).unsqueeze(1)
```

`python/sglang/srt/models/deepseek_v2.py:1650 DeepseekV2AttentionMLA.forward_absorb_prepare` 到 `python/sglang/srt/models/deepseek_v2.py:1651 DeepseekV2AttentionMLA.forward_absorb_prepare` 把 query 和 latent cache 切开：

```python
q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
k_pe = latent_cache[..., self.kv_lora_rank :].unsqueeze(1)
```

然后 `python/sglang/srt/models/deepseek_v2.py:1720 DeepseekV2AttentionMLA.forward_absorb_prepare` 的普通路径是：

```python
q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)
```

fp8/DeepGEMM/ROCm 分支在 `python/sglang/srt/models/deepseek_v2.py:1653 DeepseekV2AttentionMLA.forward_absorb_prepare` 到 `python/sglang/srt/models/deepseek_v2.py:1718 DeepseekV2AttentionMLA.forward_absorb_prepare` 做的是同一件事，只是换了 kernel/scale 处理。

后续 `python/sglang/srt/models/deepseek_v2.py:1749 DeepseekV2AttentionMLA.forward_absorb_core` 进入 attention core。对支持 absorb core 的后端，`python/sglang/srt/models/deepseek_v2.py:1772 DeepseekV2AttentionMLA.forward_absorb_core` 到 `python/sglang/srt/models/deepseek_v2.py:1781 DeepseekV2AttentionMLA.forward_absorb_core` 调用：

```python
attn_output = self.attn_mqa(
    q_nope_out,
    k_nope,
    k_nope,
    forward_batch,
    q_rope=q_pe,
    k_rope=k_pe,
    ...
)
```

也就是 attention 的 non-RoPE key/value 主体用的是压缩 latent `k_nope`，不是完整 per-head K/V。`q_nope_out` 已经提前乘过 `w_kc`，所以 attention score 和先 materialize full K 再做点积在代数上等价。

attention 输出仍在 latent 空间，随后 `w_vc` 用来把 latent-space attention output 展开回 value head 维度。普通路径在 `python/sglang/srt/models/deepseek_v2.py:1924 DeepseekV2AttentionMLA.forward_absorb_core` 到 `python/sglang/srt/models/deepseek_v2.py:1930 DeepseekV2AttentionMLA.forward_absorb_core`：

```python
torch.bmm(
    attn_output.transpose(0, 1),
    self.w_vc,
    out=...
)
```

最后 `python/sglang/srt/models/deepseek_v2.py:1931 DeepseekV2AttentionMLA.forward_absorb_core` 调用：

```python
output, _ = self.o_proj(attn_bmm_output)
```

### 和 DeepSeek-V2 论文“矩阵吸收”的关系

有关系，而且 `w_kc` 正是 SGLang 里实现 MLA matrix absorption 的关键张量之一。

DeepSeek-V2 论文的 MLA 部分说明：MLA 通过把 K/V 共同压缩到 latent vector 降低 KV cache；推理时 K 的 up-projection 可以被吸收到 Q 侧，V 的 up-projection 可以被吸收到 O 侧，因此不需要显式计算完整 keys/values 做 attention。论文还说明 RoPE 是位置相关的，所以需要 decoupled RoPE，否则 K 侧的矩阵无法直接吸收。参考：DeepSeek-V2 paper, Sec. 2.1.2/2.1.3, https://arxiv.org/abs/2405.04434 ，HTML 版对应 https://ar5iv.labs.arxiv.org/html/2405.04434 。

对应到 SGLang：

```text
kv_a_proj_with_mqa + kv_a_layernorm 产生压缩 latent: c_kv / k_nope
kv_b_proj.weight 的 K 部分 -> post_load 后变成 self_attn.w_kc
kv_b_proj.weight 的 V 部分 -> post_load 后变成 self_attn.w_vc
```

如果不做吸收，概念上会先算：

```text
K_full = c_kv @ W_UK
V_full = c_kv @ W_UV
score = Q_nope @ K_full^T
```

吸收后，利用矩阵乘法结合律改成：

```text
score = (Q_nope @ W_UK^T) @ c_kv^T
```

SGLang 代码里的 `w_kc` 就是这里的 `W_UK`/转置约定后的 K up-projection 权重。`python/sglang/srt/models/deepseek_v2.py:1720 DeepseekV2AttentionMLA.forward_absorb_prepare` 先算 `q_nope_out = q_nope @ w_kc`，然后 attention 直接和 latent `k_nope` 做计算。

需要注意：SGLang 没有把 `w_kc` 永久合并进 `q_proj.weight` 生成一个新的 checkpoint 权重；它是在 forward 的 absorb path 里显式执行 `q_nope @ w_kc`。这在代数意义上就是论文里的 K up-projection 吸收到 Q 侧。

同理，`w_vc` 对应 V up-projection。SGLang 没有把完整 V 写进 KV cache，而是在 attention 得到 latent-space output 后再乘 `w_vc`，再交给 `o_proj`。这对应论文里 V up-projection / output projection 一侧的吸收思想。

### 你这条 `--attention-backend triton` 命令下的一个细节

`w_kc` 在模型加载后总是会被填充，但不是每个 forward 都一定使用它。

`python/sglang/srt/models/deepseek_v2.py:1406 DeepseekV2AttentionMLA.forward_prepare` 会通过 `dispatch_attn_forward_method` 选择 MHA 还是 MLA。

对 `--attention-backend triton`，选择逻辑在 `python/sglang/srt/models/deepseek_common/attention_backend_handler.py:158 handle_attention_triton`：

```python
if (
    forward_batch.forward_mode.is_extend_without_speculative()
    and sum(forward_batch.extend_prefix_lens_cpu) == 0
):
    return AttnForwardMethod.MHA
else:
    return _dispatch_mla_subtype(attn, forward_batch)
```

所以在 `triton` 后端下，首个没有 prefix 的 extend/prefill 可能走 MHA 路径；decode 或带 prefix 的请求会走 MLA 路径。`python/sglang/srt/models/deepseek_v2.py:1419 DeepseekV2AttentionMLA.forward_prepare` 到 `python/sglang/srt/models/deepseek_v2.py:1421 DeepseekV2AttentionMLA.forward_prepare` 只有当 `attn_forward_method == AttnForwardMethod.MLA` 时才会进入 `forward_absorb_prepare`，也才会实际使用 `w_kc`。

## 这条 SGLang 启动命令下 MLA absorb 是否能避免 latent 显式升维开销

结论：在你这条命令下，SGLang 可以在 MLA 路径上实现 DeepSeek-V2 论文里 matrix absorption 的主要效果，但不是所有 forward 都走这个路径。具体说：

```text
decode / 带 prefix 的请求：走 MLA absorb，能避免把历史 latent cache 显式升维成 full K/V。
首个无 prefix 的 prefill/extend：在 triton 后端默认走 MHA，会显式执行 kv_b_proj(kv_a)，不能完全体现 absorb 避免升维 GEMM 的效果。
```

### 分支选择决定是否真正走 absorb

入口在 `python/sglang/srt/models/deepseek_v2.py:1406 DeepseekV2AttentionMLA.forward_prepare`：

```python
attn_forward_method = self.dispatch_attn_forward_method(forward_batch)
```

如果选择到 MHA，`python/sglang/srt/models/deepseek_v2.py:1407 DeepseekV2AttentionMLA.forward_prepare` 到 `python/sglang/srt/models/deepseek_v2.py:1410 DeepseekV2AttentionMLA.forward_prepare` 会调用：

```python
inner_state = self.forward_normal_prepare(...)
```

如果选择到 MLA，`python/sglang/srt/models/deepseek_v2.py:1419 DeepseekV2AttentionMLA.forward_prepare` 到 `python/sglang/srt/models/deepseek_v2.py:1421 DeepseekV2AttentionMLA.forward_prepare` 会调用：

```python
inner_state = self.forward_absorb_prepare(...)
```

你启动命令里是 `--attention-backend triton`，对应选择逻辑在 `python/sglang/srt/models/deepseek_common/attention_backend_handler.py:158 handle_attention_triton`：

```python
if (
    forward_batch.forward_mode.is_extend_without_speculative()
    and sum(forward_batch.extend_prefix_lens_cpu) == 0
):
    return AttnForwardMethod.MHA
else:
    return _dispatch_mla_subtype(attn, forward_batch)
```

也就是说，在没有 `--enable-deterministic-inference` 的情况下，`triton` 后端对“无 prefix 的 extend/prefill”默认返回 `MHA`；其他情况返回 MLA subtype。`python/sglang/srt/models/deepseek_common/attention_backend_handler.py:162 handle_attention_triton` 到 `python/sglang/srt/models/deepseek_common/attention_backend_handler.py:164 handle_attention_triton` 还写了一个例外：如果打开 deterministic inference，会直接用 MLA。

### MHA 路径会显式做 latent 升维

MHA 路径实现来自 mixin：`python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:94 DeepseekMHAForwardMixin.forward_normal_prepare`。

对 DeepSeek-V2-Lite 这个 `q_lora_rank = null` 的模型，`python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:180 DeepseekMHAForwardMixin.forward_normal_prepare` 到 `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:185 DeepseekMHAForwardMixin.forward_normal_prepare` 先算：

```python
q = self.q_proj(hidden_states)[0].view(...)
latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
```

`python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:186 DeepseekMHAForwardMixin.forward_normal_prepare` 到 `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:208 DeepseekMHAForwardMixin.forward_normal_prepare` 会取出 `kv_a` 和 `k_pe`，并对 `kv_a` 做 RMSNorm。

关键是 `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:235 DeepseekMHAForwardMixin.forward_normal_prepare` 到 `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:240 DeepseekMHAForwardMixin.forward_normal_prepare`：

```python
kv = self.kv_b_proj(kv_a)[0]
```

然后 `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:241 DeepseekMHAForwardMixin.forward_normal_prepare` 到 `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:243 DeepseekMHAForwardMixin.forward_normal_prepare` 把它 reshape/split 成 full K/V：

```python
kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
k_nope = kv[..., : self.qk_nope_head_dim]
v = kv[..., self.qk_nope_head_dim :]
```

`python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:245 DeepseekMHAForwardMixin.forward_normal_prepare` 再拼出 full K：

```python
k = self._concat_and_cast_mha_k(k_nope, k_pe, forward_batch)
```

最后 `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:248 DeepseekMHAForwardMixin.forward_normal_core` 到 `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:257 DeepseekMHAForwardMixin.forward_normal_core` 走普通 MHA attention：

```python
attn_output = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
...
output, _ = self.o_proj(attn_output)
```

所以首个无 prefix prefill/extend 如果被 `handle_attention_triton` 分到 MHA，确实会显式做 `kv_b_proj(kv_a)`，也就是会把 latent 升维成每个 head 的 K/V 来做这次 attention。这一段不能说“完全避免了 latent 显式升维 GEMM”。

不过注意一个细节：即使 MHA 路径显式算了当前 batch 的 full K/V，它仍然会把 latent 形式写入 MLA KV cache。`python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:213 DeepseekMHAForwardMixin.forward_normal_prepare` 调用：

```python
self._set_mla_kv_buffer(latent_cache, kv_a, k_pe, forward_batch)
```

`python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:389 DeepseekMHAForwardMixin._set_mla_kv_buffer` 到 `python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py:400 DeepseekMHAForwardMixin._set_mla_kv_buffer` 在 CUDA 路径保存的是 `kv_a` 和 `k_pe`：

```python
forward_batch.token_to_kv_pool.set_mla_kv_buffer(
    self.attn_mha, forward_batch.out_cache_loc, kv_a.unsqueeze(1), k_pe
)
```

所以 MHA prefill 的问题主要是“这次 prefill attention 计算里显式升维了”，不是“KV cache 也保存成 full K/V 了”。KV cache 仍然是 latent cache。

### MLA absorb 路径可以实现论文里的避免显式升维效果

MLA 路径在 `python/sglang/srt/models/deepseek_v2.py:1525 DeepseekV2AttentionMLA.forward_absorb_prepare`。

对 `q_lora_rank = null` 的 DeepSeek-V2-Lite，`python/sglang/srt/models/deepseek_v2.py:1643 DeepseekV2AttentionMLA.forward_absorb_prepare` 到 `python/sglang/srt/models/deepseek_v2.py:1648 DeepseekV2AttentionMLA.forward_absorb_prepare` 只生成 latent：

```python
q = self.q_proj(hidden_states)[0].view(...)
latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
k_nope = latent_cache[..., : self.kv_lora_rank]
k_nope = self.kv_a_layernorm(k_nope).unsqueeze(1)
```

这里没有调用 `self.kv_b_proj(kv_a)` 来把 `k_nope` 显式升维成 full per-head K/V。

K 侧 absorption 的关键在 `python/sglang/srt/models/deepseek_v2.py:1650 DeepseekV2AttentionMLA.forward_absorb_prepare` 到 `python/sglang/srt/models/deepseek_v2.py:1720 DeepseekV2AttentionMLA.forward_absorb_prepare`：

```python
q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
...
q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)
```

这一步把 `kv_b_proj` 的 K up-projection 权重吸收到 Q 侧：不算 `K_full = latent @ W_UK`，而是算 `Q_absorbed = Q_nope @ W_UK^T`。

对 `triton` 来说，`FORWARD_ABSORB_CORE_ATTENTION_BACKENDS` 不包含 `"triton"`。列表在 `python/sglang/srt/models/deepseek_v2.py:205` 到 `python/sglang/srt/models/deepseek_v2.py:212`，只有：

```python
["fa3", "nsa", "flashinfer", "cutlass_mla", "trtllm_mla", "ascend"]
```

所以 `triton` 的 MLA core 会走 `python/sglang/srt/models/deepseek_v2.py:1782 DeepseekV2AttentionMLA.forward_absorb_core` 之后的 else 分支。`python/sglang/srt/models/deepseek_v2.py:1810 DeepseekV2AttentionMLA.forward_absorb_core` 到 `python/sglang/srt/models/deepseek_v2.py:1811 DeepseekV2AttentionMLA.forward_absorb_core` 做的是：

```python
q = torch.cat([q_nope_out, q_pe], dim=-1)
k = torch.cat([k_nope, k_pe], dim=-1)
```

这里的 `k_nope` 仍然是 latent 维度，不是 `kv_b_proj` 升维后的 full no-RoPE key。随后 `python/sglang/srt/models/deepseek_v2.py:1817 DeepseekV2AttentionMLA.forward_absorb_core` 到 `python/sglang/srt/models/deepseek_v2.py:1824 DeepseekV2AttentionMLA.forward_absorb_core` 调用：

```python
attn_output = self.attn_mqa(
    q,
    k,
    k_nope,
    forward_batch,
    save_kv_cache=save_kv_cache,
    ...
)
```

这说明 MLA 路径里 attention 的 value 也是 latent `k_nope`，不是 full V。

V 侧 expansion 被推迟到 attention 之后。`python/sglang/srt/models/deepseek_v2.py:1919 DeepseekV2AttentionMLA.forward_absorb_core` 到 `python/sglang/srt/models/deepseek_v2.py:1930 DeepseekV2AttentionMLA.forward_absorb_core` 的普通路径：

```python
torch.bmm(
    attn_output.transpose(0, 1),
    self.w_vc,
    out=...
)
```

这个 GEMM 仍然存在，但它只对 attention output 做一次，不是对所有历史 KV cache token 显式生成 full V 再参与 attention。最后 `python/sglang/srt/models/deepseek_v2.py:1931 DeepseekV2AttentionMLA.forward_absorb_core` 再做：

```python
output, _ = self.o_proj(attn_bmm_output)
```

因此，MLA absorb 路径的计算结构确实符合论文的核心目标：避免对历史 latent cache 显式执行 `kv_b_proj` 升维成完整 K/V，KV cache 里保存 latent，attention 直接在 absorbed query + latent key/value 上做。

### 对你这条命令的实际判断

你的命令没有 `--enable-deterministic-inference`，并且用了 `--attention-backend triton`。按当前代码：

```text
1. 第一次无 prefix prefill/extend:
   handle_attention_triton -> MHA
   forward_normal_prepare -> self.kv_b_proj(kv_a)
   这次会显式 latent 升维。

2. decode 或有 prefix 的请求:
   handle_attention_triton -> MLA
   forward_absorb_prepare / forward_absorb_core
   不会显式对历史 latent cache 做 kv_b_proj 升维。

3. KV cache:
   即使 MHA prefill 显式算了当前 full K/V，_set_mla_kv_buffer 仍然保存 latent kv_a + k_pe，而不是保存 full K/V。
```

所以答案不是简单的“可以”或“不可以”：

```text
对 decode 阶段和 prefix cache 重用阶段：可以，实现了论文里 absorb 避免历史 latent 显式升维的效果。
对首个无 prefix prefill：默认不可以，因为 triton 分支主动选择 MHA，会显式调用 kv_b_proj(kv_a)。
```

如果你想让 `triton` 后端的无 prefix prefill 也走 MLA absorb，从代码看可以尝试在启动命令里加：

```bash
--enable-deterministic-inference
```

因为 `python/sglang/srt/models/deepseek_common/attention_backend_handler.py:162 handle_attention_triton` 到 `python/sglang/srt/models/deepseek_common/attention_backend_handler.py:164 handle_attention_triton` 明确写了 deterministic inference 会直接返回 MLA subtype。不过这会改变调度/性能路径，是否更快需要用你的 workload 实测。

## 对 forward_absorb_core 里 FORWARD_ABSORB_CORE_ATTENTION_BACKENDS 判断的更正说明

你的质疑是对的。对你这条启动命令：

```bash
--attention-backend triton --disable-cuda-graph
```

`python/sglang/srt/models/deepseek_v2.py:1763 DeepseekV2AttentionMLA.forward_absorb_core` 这一句：

```python
if self.current_attention_backend in FORWARD_ABSORB_CORE_ATTENTION_BACKENDS:
```

在正常情况下不是 True，而是 False。

### 为什么这个 if 不是 True

`FORWARD_ABSORB_CORE_ATTENTION_BACKENDS` 定义在 `python/sglang/srt/models/deepseek_v2.py:205` 到 `python/sglang/srt/models/deepseek_v2.py:212`：

```python
FORWARD_ABSORB_CORE_ATTENTION_BACKENDS = [
    "fa3",
    "nsa",
    "flashinfer",
    "cutlass_mla",
    "trtllm_mla",
    "ascend",
]
```

这里没有 `"triton"`。

而 `self.current_attention_backend` 在 `python/sglang/srt/models/deepseek_v2.py:1323 DeepseekV2AttentionMLA.dispatch_attn_forward_method` 里设置。关键代码在 `python/sglang/srt/models/deepseek_v2.py:1327 DeepseekV2AttentionMLA.dispatch_attn_forward_method` 到 `python/sglang/srt/models/deepseek_v2.py:1340 DeepseekV2AttentionMLA.dispatch_attn_forward_method`：

```python
if forward_batch.forward_mode.is_decode_or_idle():
    attention_backend = get_global_server_args().decode_attention_backend
...
else:
    attention_backend = get_global_server_args().prefill_attention_backend
self.current_attention_backend = attention_backend
```

你的日志 `/share/users/like/package/h100/package/simo_conda_sglang/temp/sglang_server.2026_05_14___17_45_37` 里 `server_args` 显示：

```text
attention_backend='triton'
decode_attention_backend=None
prefill_attention_backend=None
enable_deterministic_inference=False
disable_cuda_graph=True
```

所以在这份代码里，`self.current_attention_backend` 很可能是 `None`，不是 `"triton"`。即使你显式把 `prefill_attention_backend/decode_attention_backend` 也设置成 `"triton"`，它也不在 `FORWARD_ABSORB_CORE_ATTENTION_BACKENDS` 里。因此 `python/sglang/srt/models/deepseek_v2.py:1763 DeepseekV2AttentionMLA.forward_absorb_core` 的 if 对这条命令不会走 True 分支。

这里还有一个容易混淆的点：虽然 `current_attention_backend` 可能是 `None`，handler 仍然会 fallback 到 triton。`python/sglang/srt/models/deepseek_common/attention_backend_handler.py:21 AttentionBackendRegistry.get_handler` 到 `python/sglang/srt/models/deepseek_common/attention_backend_handler.py:22 AttentionBackendRegistry.get_handler`：

```python
return cls._handlers.get(backend_name, cls._handlers.get("triton"))
```

所以 `backend_name=None` 时也会调用 triton handler，但 `current_attention_backend` 仍不是 `"fa3"/"flashinfer"/...`，因此 absorb core 的 True 分支仍不成立。

### 哪个 attn_mqa 会被调用

`python/sglang/srt/models/deepseek_v2.py:1772 DeepseekV2AttentionMLA.forward_absorb_core` 这个 True 分支里的调用：

```python
attn_output = self.attn_mqa(
    q_nope_out,
    k_nope,
    k_nope,
    forward_batch,
    q_rope=q_pe,
    k_rope=k_pe,
    ...
)
```

对你这条 `--attention-backend triton` 命令不会被调用。

但是，如果已经进入了 `forward_absorb_core`，else 分支里的 `attn_mqa` 会被调用。对应 `python/sglang/srt/models/deepseek_v2.py:1810 DeepseekV2AttentionMLA.forward_absorb_core` 到 `python/sglang/srt/models/deepseek_v2.py:1824 DeepseekV2AttentionMLA.forward_absorb_core`：

```python
q = torch.cat([q_nope_out, q_pe], dim=-1)
k = torch.cat([k_nope, k_pe], dim=-1)

attn_output = self.attn_mqa(
    q,
    k,
    k_nope,
    forward_batch,
    save_kv_cache=save_kv_cache,
    ...
)
```

所以准确说：

```text
1. 首个无 prefix prefill/extend:
   handle_attention_triton -> MHA
   不进入 forward_absorb_core
   因此 1772 和 1817 两个 attn_mqa 都不会在这个 MLA core 里调用。

2. decode 或带 prefix 的请求:
   handle_attention_triton -> MLA
   会进入 forward_absorb_core
   1763 的 if 为 False
   调用的是 1817 这一处 self.attn_mqa(q, k, k_nope, ...)
   不是 1772 这一处 self.attn_mqa(q_nope_out, k_nope, k_nope, q_rope=..., k_rope=...)
```

### 日志也能验证这一点

日志里有这些行：

```text
triton_backend.py:824 forward_extend ... q.shape:torch.Size([8, 16, 192])
triton_backend.py:1028 forward_decode ... q.shape:torch.Size([1, 16, 576])
```

对 DeepSeek-V2-Lite：

```text
qk_nope_head_dim = 128
qk_rope_head_dim = 64
kv_lora_rank = 512
```

`192 = 128 + 64`，这是 MHA/full QK 的维度，符合首个 no-prefix extend/prefill 走 MHA。

`576 = 512 + 64`，这是 MLA absorb 路径里 `q = cat([q_nope_out, q_pe])` 后的维度，符合 `python/sglang/srt/models/deepseek_v2.py:1810 DeepseekV2AttentionMLA.forward_absorb_core` 的 else 分支，而不是 `python/sglang/srt/models/deepseek_v2.py:1772 DeepseekV2AttentionMLA.forward_absorb_core` 的 True 分支。

### 我之前讲解里需要更正的地方

之前如果把 `python/sglang/srt/models/deepseek_v2.py:1772 DeepseekV2AttentionMLA.forward_absorb_core` 那个 `self.attn_mqa(q_nope_out, k_nope, k_nope, q_rope=..., k_rope=...)` 说成你这条 `triton` 命令下会执行，那是不准确的。

更准确的说法是：

```text
对 fa3/flashinfer/cutlass_mla/trtllm_mla/nsa/ascend 这些 backend：
  forward_absorb_core 走 1763 True 分支，调用 1772 的 attn_mqa。

对你这条 --attention-backend triton 命令：
  首个 no-prefix prefill 走 MHA，不进 forward_absorb_core；
  decode/带 prefix 走 MLA，但 1763 为 False，调用 1817 的 attn_mqa。
```

不过前面关于“triton 的 MLA absorb 路径不会显式执行 `kv_b_proj(kv_a)` 升维历史 latent cache”的结论仍然成立。因为 else 分支里 `python/sglang/srt/models/deepseek_v2.py:1810 DeepseekV2AttentionMLA.forward_absorb_core` 到 `python/sglang/srt/models/deepseek_v2.py:1811 DeepseekV2AttentionMLA.forward_absorb_core` 使用的是：

```python
q = torch.cat([q_nope_out, q_pe], dim=-1)
k = torch.cat([k_nope, k_pe], dim=-1)
```

这里的 `k_nope` 是 latent 维度 `kv_lora_rank=512`，不是 `kv_b_proj` 升维后的 `qk_nope_head_dim=128` full K。然后 `python/sglang/srt/models/deepseek_v2.py:1817 DeepseekV2AttentionMLA.forward_absorb_core` 用 `self.attn_mqa(q, k, k_nope, ...)` 做 attention。

## TRITON_INTERPRET=1 调试 decode_attention_fwd 的报错原因和解决办法

你的命令是：

```bash
TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES=0 python3 ../sglang_kernel_src/like-useful/load_sgl_src_decode_attention_fwd.py
```

日志 `temp/load_sgl_src_decode_attention_fwd.py.deepseek.log.2026_05_20___18_49_18` 的核心报错是：

```text
python/sglang/srt/layers/attention/triton_ops/decode_attention.py:338
for start_n in range(split_kv_start, split_kv_end, BLOCK_N):

TypeError: only 0-dimensional arrays can be converted to Python scalars
```

这不是 H100、FP8 tensor、safetensors 读取、也不是 `decode_attention_fwd` 入口参数本身的错误。根因是：这个 Triton kernel 在 `TRITON_INTERPRET=1` 解释执行模式下，不能处理由 runtime tensor 决定边界的 Python `range(...)`。

### 实际调用链路

你的脚本在 `like-useful/load_sgl_src_decode_attention_fwd.py:41` 调用：

```python
decode_attention_fwd(...)
```

然后进入 SGLang 的 decode attention：

```text
python/sglang/srt/layers/attention/triton_ops/decode_attention.py:719 decode_attention_fwd
  -> python/sglang/srt/layers/attention/triton_ops/decode_attention.py:761 decode_attention_fwd
  -> python/sglang/srt/layers/attention/triton_ops/decode_attention.py:676 decode_attention_fwd_grouped
  -> python/sglang/srt/layers/attention/triton_ops/decode_attention.py:426 _decode_grouped_att_m_fwd
  -> python/sglang/srt/layers/attention/triton_ops/decode_attention.py:478 _decode_grouped_att_m_fwd
  -> python/sglang/srt/layers/attention/triton_ops/decode_attention.py:285 _fwd_grouped_kernel_stage1
```

为什么走 grouped path：`python/sglang/srt/layers/attention/triton_ops/decode_attention.py:739 decode_attention_fwd` 计算：

```python
kv_group_num = q.shape[1] // v_buffer.shape[1]
```

日志里：

```text
q.shape       = [2, 16, 576]
k_buffer.shape = [1752252, 1, 576]
v_buffer.shape = [1752252, 1, 512]
```

所以：

```text
kv_group_num = 16 // 1 = 16
```

`python/sglang/srt/layers/attention/triton_ops/decode_attention.py:741 decode_attention_fwd` 到 `python/sglang/srt/layers/attention/triton_ops/decode_attention.py:761 decode_attention_fwd` 会选择 GQA/MQA/MLA 的 grouped kernel，这是 DeepSeek MLA decode 的预期路径。

生产路径里，SGLang triton backend 是在 `python/sglang/srt/layers/attention/triton_backend.py:1015 TritonAttnBackend.forward_decode` 进入 decode，最后 `python/sglang/srt/layers/attention/triton_backend.py:1066 TritonAttnBackend.forward_decode` 到 `python/sglang/srt/layers/attention/triton_backend.py:1081 TritonAttnBackend.forward_decode` 调用：

```python
self.decode_attention_fwd(...)
```

所以你的 standalone 脚本复现的是生产 decode attention 的核心调用。

### 报错触发点

`python/sglang/srt/layers/attention/triton_ops/decode_attention.py:303 _fwd_grouped_kernel_stage1` 到 `python/sglang/srt/layers/attention/triton_ops/decode_attention.py:326 _fwd_grouped_kernel_stage1` 里，`split_kv_start` 和 `split_kv_end` 是通过 `tl.load` 和 `tl.program_id` 算出来的 runtime Triton tensor：

```python
cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
kv_splits = tl.load(num_kv_splits + cur_batch)

kv_len_per_split = (
    tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
)
split_kv_start = kv_len_per_split * split_kv_id
split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)
```

然后 `python/sglang/srt/layers/attention/triton_ops/decode_attention.py:338 _fwd_grouped_kernel_stage1` 使用 Python 内置 `range`：

```python
for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
```

在正常 Triton JIT 编译执行时，这种写法会被 Triton 编译器处理。但 `TRITON_INTERPRET=1` 是 Python/Numpy 解释执行。解释器需要把 `split_kv_start` / `split_kv_end` 转成 Python `int` 传给 `range`，于是走到 Triton interpreter 里的 `tensor.__index__`，最终报：

```text
TypeError: only 0-dimensional arrays can be converted to Python scalars
```

也就是说，这个 kernel 当前写法对正常编译执行可用，但对 Triton interpreter 不兼容。

同类问题在 normal/MHA stage1 也存在：`python/sglang/srt/layers/attention/triton_ops/decode_attention.py:109 _fwd_kernel_stage1` 也有：

```python
for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
```

而 stage2 的 `python/sglang/srt/layers/attention/triton_ops/decode_attention.py:555 _fwd_kernel_stage2` 是：

```python
for split_kv_id in range(0, MAX_KV_SPLITS):
```

这里 `MAX_KV_SPLITS` 是 constexpr/Python 静态值，所以不是这次报错的来源。

### 还有一个脚本问题：time_prefix 被覆盖

`like-useful/load_sgl_src_decode_attention_fwd.py:6` 到 `like-useful/load_sgl_src_decode_attention_fwd.py:7`：

```python
time_prefix="1779268229.8098829"  # dsv2 lite
time_prefix="1776764594.4215453"  # llama
```

第二行会覆盖第一行。日志里的 tensor shape 是：

```text
q.shape = [2, 16, 576]
kv_indptr.shape = [3]
kv_indices.shape = [18]
```

这对应实际加载的 `1776764594.4215453` 这份数据。另一份 `1779268229.8098829` 是：

```text
q.shape = [1, 16, 576]
kv_indptr.shape = [2]
kv_indices.shape = [9]
```

两份都是 DeepSeek MLA 形状，因为 `576 = kv_lora_rank 512 + qk_rope_head_dim 64`。但如果你想调单请求那份 DeepSeek 数据，应该注释掉第二个赋值：

```python
time_prefix = "1779268229.8098829"  # dsv2 lite, batch=1
# time_prefix = "1776764594.4215453"
```

不过这个覆盖不是当前 `TRITON_INTERPRET` 报错的根因。即使用 batch=1 那份数据，`python/sglang/srt/layers/attention/triton_ops/decode_attention.py:338 _fwd_grouped_kernel_stage1` 仍然会遇到同类 interpreter 限制。

### 如何解决

分两类目标。

#### 目标 1：只想确认 kernel 正常跑

不要开 `TRITON_INTERPRET=1`，直接跑正常 JIT 编译路径：

```bash
CUDA_VISIBLE_DEVICES=0 /data/like/miniconda3/envs/simo_sglang/bin/python ../sglang_kernel_src/like-useful/load_sgl_src_decode_attention_fwd.py
```

这可以验证输入数据和 SGLang decode attention 正常，但不能进入 Triton interpreter 单步调试。

#### 目标 2：必须用 TRITON_INTERPRET 调试 stage1

需要把 `decode_attention.py` 的 stage1 kernel 改成 interpreter 友好的写法：不要用 runtime tensor 作为 Python `range` 的 start/end。具体思路是把动态范围：

```python
for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
```

改成“静态 Python 上界 + mask”的形式，例如调试版本增加一个 constexpr：

```python
INTERPRET_MAX_KV_LEN_PER_SPLIT: tl.constexpr
```

然后在 `TRITON_INTERPRET=1` 时走静态循环：

```python
for rel_n in range(0, INTERPRET_MAX_KV_LEN_PER_SPLIT, BLOCK_N):
    start_n = split_kv_start + rel_n
    offs_n = start_n + tl.arange(0, BLOCK_N)
    # 后面的 tl.load / mask 仍然使用 offs_n < split_kv_end
```

`INTERPRET_MAX_KV_LEN_PER_SPLIT` 可以在 Python wrapper 里根据当前输入算出来。对 `_decode_grouped_att_m_fwd`，位置是 `python/sglang/srt/layers/attention/triton_ops/decode_attention.py:426 _decode_grouped_att_m_fwd`。可以在 launch 前用 torch 计算：

```python
seq_lens = kv_indptr[1:] - kv_indptr[:-1]
per_split = ((seq_lens + num_kv_splits - 1) // num_kv_splits)
per_split = ((per_split + _MIN_BLOCK_KV - 1) // _MIN_BLOCK_KV) * _MIN_BLOCK_KV
INTERPRET_MAX_KV_LEN_PER_SPLIT = int(per_split.max().item())
```

然后在 `python/sglang/srt/layers/attention/triton_ops/decode_attention.py:478 _decode_grouped_att_m_fwd` launch `_fwd_grouped_kernel_stage1` 时把它作为 constexpr 传进去。

为了不影响生产 kernel，建议只在 `TRITON_INTERPRET=1` 时启用这个 debug 分支，例如在 Python wrapper 中：

```python
import os
interpret_max_kv_len_per_split = 0
if os.getenv("TRITON_INTERPRET") == "1":
    seq_lens = kv_indptr[1:] - kv_indptr[:-1]
    per_split = ((seq_lens + num_kv_splits - 1) // num_kv_splits)
    per_split = ((per_split + _MIN_BLOCK_KV - 1) // _MIN_BLOCK_KV) * _MIN_BLOCK_KV
    interpret_max_kv_len_per_split = int(per_split.max().item())
```

kernel 里用 constexpr 分支：

```python
if INTERPRET_MAX_KV_LEN_PER_SPLIT > 0:
    for rel_n in range(0, INTERPRET_MAX_KV_LEN_PER_SPLIT, BLOCK_N):
        start_n = split_kv_start + rel_n
        ...
else:
    for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
        ...
```

这样正常运行时仍走原来的动态循环；只有 interpreter 调试时走静态循环。

这次保存的数据里 sequence length 很小：

```text
1779268229.8098829: kv_indices length = 9
1776764594.4215453: kv_indices length = 18, batch=2，平均每条 9
```

`MIN_BLOCK_KV=32`，所以 debug 静态上界会是 32。也就是说这份数据在 stage1 里每个 split 实际只需要一个 `BLOCK_N=32` tile。最小化调试时，甚至可以先把 debug kernel 写死：

```python
for rel_n in range(0, 32, BLOCK_N):
    start_n = split_kv_start + rel_n
    ...
```

但这个只适合当前 safetensor，不适合作为通用修复。

### 建议的排查顺序

1. 先修正 `like-useful/load_sgl_src_decode_attention_fwd.py:6` 到 `like-useful/load_sgl_src_decode_attention_fwd.py:7` 的 `time_prefix` 覆盖问题，明确你要调 batch=1 还是 batch=2 的 DeepSeek 数据。

2. 不带 `TRITON_INTERPRET=1` 跑一次，确认 saved tensor + wrapper 参数本身能正常执行。

3. 如果必须 interpreter 单步调试，就给 `python/sglang/srt/layers/attention/triton_ops/decode_attention.py:285 _fwd_grouped_kernel_stage1` 增加 debug-only 静态循环分支；必要时也给 `python/sglang/srt/layers/attention/triton_ops/decode_attention.py:45 _fwd_kernel_stage1` 做同样处理，因为 normal/MHA path 也有同样的动态 `range` 写法。

4. 不建议把这个静态循环无条件替换到生产路径。对长上下文，静态上界会增加解释执行/编译代码体积或循环次数；最好只在 `TRITON_INTERPRET=1` 下启用。

## 能否在 stage1 kernel 里把 split_kv_start/end 转成 shape=0 scalar 再给 range 用

结论：在当前 `/data/like/miniconda3/envs/simo_sglang/` 里的 Triton 3.5.1 interpreter 下，不能靠在 `stage1` kernel 里写一行正常 Triton 代码来稳定解决。理论上 Triton 有 `tensor.item()` / `reshape(())` 这种 single-element tensor 转 scalar 的接口，但当前 interpreter 实现里它不会变成 Python `range()` 需要的 0-d numpy scalar，仍然会是 shape=(1,) 的对象，所以还是会报同类错误。

### 报错位置回顾

报错发生在 grouped MLA/GQA stage1：

`python/sglang/srt/layers/attention/triton_ops/decode_attention.py:285 _fwd_grouped_kernel_stage1` 里：

```python
cur_batch = tl.program_id(0)
cur_head_id = tl.program_id(1)
split_kv_id = tl.program_id(2)
```

`python/sglang/srt/layers/attention/triton_ops/decode_attention.py:303 _fwd_grouped_kernel_stage1` 到 `python/sglang/srt/layers/attention/triton_ops/decode_attention.py:326 _fwd_grouped_kernel_stage1` 计算：

```python
cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
kv_splits = tl.load(num_kv_splits + cur_batch)

kv_len_per_split = (
    tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
)
split_kv_start = kv_len_per_split * split_kv_id
split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)
```

然后 `python/sglang/srt/layers/attention/triton_ops/decode_attention.py:338 _fwd_grouped_kernel_stage1`：

```python
for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
```

在 `TRITON_INTERPRET=1` 下，Python `range()` 会尝试调用 Triton tensor 的 `__index__`，把 `split_kv_start` / `split_kv_end` 转成 Python int。当前错误说明它们在 interpreter 里是 size=1 但 shape=(1,) 的 numpy array，而不是 0-d scalar。

### 直接在 kernel 里 `.item()` 是否可行

直觉上可能会想这样改：

```python
split_kv_start = split_kv_start.item()
split_kv_end = split_kv_end.item()
for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
    ...
```

或者：

```python
split_kv_start = split_kv_start.reshape(())
split_kv_end = split_kv_end.reshape(())
```

但在当前 Triton 3.5.1 interpreter 里，这个方向不可靠。

原因在 Triton 自己的实现。`/data/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/triton/language/core.py:1885 item` 到 `/data/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/triton/language/core.py:1891 item` 写的是：

```python
def item(input, _semantic=None, _generator=None):
    """
    Converts a single-element tensor into a scalar.
    """
    return _unsplat(input, _semantic=_semantic, _generator=_generator)
```

`/data/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/triton/language/core.py:1896 reshape` 到 `/data/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/triton/language/core.py:1914 reshape` 里，`reshape(())` 也会走 `_unsplat`：

```python
if len(shape) == 0:
    return _unsplat(input, _semantic=_semantic, _generator=_generator)
```

但是 interpreter 的 unsplat 实现在 `/data/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/triton/runtime/interpreter.py:668 create_unsplat` 到 `/data/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/triton/runtime/interpreter.py:669 create_unsplat`：

```python
def create_unsplat(self, arg):
    return TensorHandle(np.full((1, ), arg.data[0], dtype=_get_np_dtype(arg.dtype)), arg.dtype.scalar)
```

注意这里返回的是 `np.full((1,), ...)`，也就是 shape=(1,)；不是 shape=()。

而 Python `range()` 触发的 `__index__` 在 `/data/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/triton/runtime/interpreter.py:818 _patch_lang_tensor`：

```python
tensor.__index__ = lambda self: int(self.handle.data)
```

对 numpy 2.x 来说，`int(np.array([123]))` 会报：

```text
TypeError: only 0-dimensional arrays can be converted to Python scalars
```

所以即使用 `.item()` / `reshape(())`，在这个 interpreter 实现下也不能保证 `range()` 拿到真正的 shape=0 scalar。

### 能不能改成 `tl.static_range` 或 `tl.range`

也不适合直接解决这个问题。

`tl.static_range`/Python `range` 都要求循环边界是编译期静态值，适合 `MAX_KV_SPLITS` 这种 constexpr。这里 `split_kv_start` / `split_kv_end` 来自：

```text
kv_indptr / num_kv_splits / tl.program_id(2)
```

它们是每个 batch、每个 split 不同的 runtime 值，不是 constexpr。

`tl.range` 在正常 JIT/IR 里可以表达一些 runtime loop，但你现在的问题是 `TRITON_INTERPRET=1` 的 Python interpreter 不能执行这个动态边界循环；换成 `tl.range(split_kv_start, split_kv_end, BLOCK_N)` 也不一定能解决 interpreter 单步调试的问题，而且可能改变正常 JIT 的编译行为。

### 真正可行的几种办法

#### 办法 1：推荐，stage1 kernel 改 debug-only 静态上界循环

这个是最稳的办法：不要让 Python `range()` 的 start/end 依赖 `split_kv_start/end`，而是用 constexpr 上界循环，再用 mask 限制真实范围。

在 `python/sglang/srt/layers/attention/triton_ops/decode_attention.py:285 _fwd_grouped_kernel_stage1` 增加一个 constexpr，例如：

```python
INTERPRET_MAX_KV_LEN_PER_SPLIT: tl.constexpr,
```

然后把 `python/sglang/srt/layers/attention/triton_ops/decode_attention.py:338 _fwd_grouped_kernel_stage1` 的动态循环改成 debug-only 分支：

```python
if INTERPRET_MAX_KV_LEN_PER_SPLIT > 0:
    for rel_n in range(0, INTERPRET_MAX_KV_LEN_PER_SPLIT, BLOCK_N):
        start_n = split_kv_start + rel_n
        offs_n = start_n + tl.arange(0, BLOCK_N)
        # 原来的 tl.load/mask 逻辑继续用 offs_n < split_kv_end
else:
    for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        # 原逻辑
```

在 `python/sglang/srt/layers/attention/triton_ops/decode_attention.py:426 _decode_grouped_att_m_fwd` 的 Python wrapper 里，只在 `TRITON_INTERPRET=1` 时计算这个 constexpr：

```python
import os

interpret_max_kv_len_per_split = 0
if os.getenv("TRITON_INTERPRET") == "1":
    seq_lens = kv_indptr[1:] - kv_indptr[:-1]
    per_split = (seq_lens + num_kv_splits - 1) // num_kv_splits
    per_split = ((per_split + _MIN_BLOCK_KV - 1) // _MIN_BLOCK_KV) * _MIN_BLOCK_KV
    interpret_max_kv_len_per_split = int(per_split.max().item())
```

然后在 `python/sglang/srt/layers/attention/triton_ops/decode_attention.py:478 _decode_grouped_att_m_fwd` launch `_fwd_grouped_kernel_stage1` 时传入：

```python
INTERPRET_MAX_KV_LEN_PER_SPLIT=interpret_max_kv_len_per_split,
```

这样生产路径 `INTERPRET_MAX_KV_LEN_PER_SPLIT=0`，仍走原来的动态循环；interpreter 调试路径走静态循环，不再触发 `range(split_kv_start, split_kv_end, ...)`。

#### 办法 2：只为当前小样本临时写死 32

你当前 safetensor 里 decode 的 seq len 很短，`kv_indices` 是 9 或 18，`MIN_BLOCK_KV=32`，`BLOCK_N=32`。如果只是临时单步调试当前数据，可以在 `python/sglang/srt/layers/attention/triton_ops/decode_attention.py:338 _fwd_grouped_kernel_stage1` 附近临时改成：

```python
for rel_n in range(0, 32, BLOCK_N):
    start_n = split_kv_start + rel_n
    offs_n = start_n + tl.arange(0, BLOCK_N)
    ...
```

这能绕过 interpreter 的 `__index__`，但只适合当前短序列数据，不是通用修复。

#### 办法 3：patch Triton interpreter，仅用于本地调试

如果你坚持“不改 SGLang kernel”，可以本地临时 patch Triton interpreter，让 `__index__` 接受 size=1 的 numpy array。

位置是：

```text
/data/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/triton/runtime/interpreter.py:818 _patch_lang_tensor
```

把：

```python
tensor.__index__ = lambda self: int(self.handle.data)
```

临时改成：

```python
def _tensor_index(self):
    data = self.handle.data
    if getattr(data, "size", None) == 1:
        return int(data.item())
    return int(data)

tensor.__index__ = _tensor_index
```

这个 patch 会让原来的：

```python
for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
```

在 `split_kv_start/end` 是 size=1 tensor 时更可能跑起来。

但这不是 SGLang 代码层面的修复，也不是我最推荐的方式，因为它改的是 conda env 里的 Triton runtime，全局影响所有 `TRITON_INTERPRET=1` 的 Triton kernel。只建议作为本地临时 debug hack，并且改完要记录清楚。

### 最终建议

如果你的目标是调 `python/sglang/srt/layers/attention/triton_ops/decode_attention.py:285 _fwd_grouped_kernel_stage1` 的逻辑，我建议用“办法 1”：在 SGLang `decode_attention.py` 里加 `TRITON_INTERPRET=1` 才启用的静态上界循环。

不建议在 kernel 里尝试：

```python
split_kv_start = split_kv_start.item()
split_kv_end = split_kv_end.item()
```

因为在当前 Triton interpreter 实现下，它并不能可靠变成 Python `range()` 接受的 0-d scalar；即使某个版本能工作，也是在依赖 interpreter 的实现细节，不如显式把循环改成静态上界 + mask。

## stage1 里把动态 range 改成 while 是否可行

结论：对 `TRITON_INTERPRET=1` 调试来说，这个改法是可行方向，而且比 `.item()` / `reshape(())` 更靠谱。

也就是把 `python/sglang/srt/layers/attention/triton_ops/decode_attention.py:338 _fwd_grouped_kernel_stage1` 的：

```python
for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
    ...
```

改成：

```python
start_n = split_kv_start
while start_n < split_kv_end:
    ...
    start_n += BLOCK_N
```

这个写法避免了 Python `range()` 对 `split_kv_start` / `split_kv_end` 调用 `__index__`，因此可以绕过当前报错：

```text
TypeError: only 0-dimensional arrays can be converted to Python scalars
```

### 为什么 while 可以绕过这个错误

原来的错误点是 `python/sglang/srt/layers/attention/triton_ops/decode_attention.py:338 _fwd_grouped_kernel_stage1`：

```python
for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
```

这里的 `split_kv_start` 和 `split_kv_end` 来自 `python/sglang/srt/layers/attention/triton_ops/decode_attention.py:322 _fwd_grouped_kernel_stage1` 到 `python/sglang/srt/layers/attention/triton_ops/decode_attention.py:326 _fwd_grouped_kernel_stage1`：

```python
kv_len_per_split = (
    tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
)
split_kv_start = kv_len_per_split * split_kv_id
split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)
```

在 Triton interpreter 里它们是 Triton tensor，底层 data 是 size=1 的 numpy array。Python `range()` 必须先把参数转成 Python int，于是触发 `/data/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/triton/runtime/interpreter.py:818 _patch_lang_tensor`：

```python
tensor.__index__ = lambda self: int(self.handle.data)
```

`int(np.array([x]))` 在当前 numpy/Triton interpreter 组合下会报错。

但是 `while start_n < split_kv_end:` 不需要 `__index__`。它走的是 Triton tensor 的比较和 bool 转换：

`/data/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/triton/language/core.py:1028 tensor.__lt__` 到 `/data/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/triton/language/core.py:1032 tensor.__lt__`：

```python
def __lt__(self, other, _semantic=None):
    other = _semantic.to_tensor(other)
    return _semantic.less_than(self, other)
```

然后 interpreter 对 bool 的处理在 `/data/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/triton/runtime/interpreter.py:804 _patch_lang_tensor` 到 `/data/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/triton/runtime/interpreter.py:819 _patch_lang_tensor`：

```python
def _get_bool(self):
    data = self.handle.data
    return bool(data) if data.size == 1 else True

tensor.__bool__ = lambda self: _get_bool(self)
```

所以 size=1 的比较结果可以转成 Python bool，`while` 能执行。

我用当前 conda env 做了一个最小复现：

```python
@triton.jit
def while_kernel(x, y):
    start = tl.load(x + 0)
    end = tl.load(x + 1)
    acc = tl.full((), 0, tl.int32)
    while start < end:
        acc += start
        start += 1
    tl.store(y, acc)
```

在 `TRITON_INTERPRET=1` 下可以跑出结果；同样逻辑换成 `for i in range(start, end, 1)` 会复现你遇到的 `__index__` 报错。

### 建议的具体改法

对 grouped MLA path，改 `python/sglang/srt/layers/attention/triton_ops/decode_attention.py:338 _fwd_grouped_kernel_stage1` 附近即可。核心结构应该是：

```python
start_n = split_kv_start
while start_n < split_kv_end:
    offs_n = start_n + tl.arange(0, BLOCK_N)
    kv_loc = tl.load(
        kv_indices + cur_batch_kv_start_idx + offs_n,
        mask=offs_n < split_kv_end,
        other=0,
    )

    # 原来 for 循环体里的 qk / softmax / acc 逻辑保持不变

    start_n += BLOCK_N
```

这个改法的语义和原来的：

```python
for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
```

是一致的：每次处理一个 `BLOCK_N` tile，最后一个 tile 仍靠 `mask=offs_n < split_kv_end` 保护越界。

### 也要处理 normal/MHA path

如果你后面调的不是 DeepSeek MLA grouped path，而是普通 MHA/GQA path，同类动态 `range` 还存在于 `python/sglang/srt/layers/attention/triton_ops/decode_attention.py:109 _fwd_kernel_stage1`：

```python
for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
```

那里也可以用同样方式改成：

```python
start_n = split_kv_start
while start_n < split_kv_end:
    ...
    start_n += BLOCK_N
```

否则 normal path 在 `TRITON_INTERPRET=1` 下仍可能遇到同类报错。

### 注意事项

1. 这个改法是为了 interpreter 调试友好。正常 Triton JIT 是否保持性能，需要实际 benchmark。`while` 是 runtime loop，编译器处理方式可能和 `range` 不完全一样。

2. 如果只是为了调试，我更建议加在 debug 分支里，或者先改本地分支，不要直接作为性能路径提交。

3. `while start_n < split_kv_end` 必须保证 `start_n` 每轮递增：

```python
start_n += BLOCK_N
```

否则 interpreter/JIT 都会死循环。

4. 不要把 `while` 条件写成 block tensor 条件。这里 `split_kv_start/end` 是 single-element scalar-like tensor，所以可行；如果条件是 shape 大于 1 的 block tensor，interpreter 的 bool 语义会变成 `data.size != 1` 时直接 True，不适合控制循环。

### 最终判断

你的改法：

```python
start_n = split_kv_start
while start_n < split_kv_end:
    computing ...
    start_n = start_n + BLOCK_N
```

可以作为解决 `TRITON_INTERPRET=1` 报错的直接方案。它比尝试把 `split_kv_start/end` 转 shape=0 scalar 再给 `range()` 更靠谱。

如果目标是长期保留到 SGLang 代码里，我建议只对 `TRITON_INTERPRET=1` 或本地 debug 分支启用，正常生产路径最好保留原 `range` 写法，除非你 benchmark 确认 `while` 版本没有性能/编译退化。

## SIMOConfig 的 is_checkpoint_quant_serialized 参数在哪个 commit 删除

结论：删除发生在这个提交：

```text
62d9622b14916b35312666bea386e4bf7ea32981
Dehua/update vllm 0.15.0 (model-opt/simo!142)
Author: dehua <hchu@siorigin.com>
Date: 2026-02-02T11:24:51+00:00
```

我用下面命令确认过：

```bash
git log --oneline -S'is_checkpoint_quant_serialized' -- simo/extensions/vllm_simo/quantization/quantization_config.py
```

结果只有两个相关提交：

```text
62d9622 Dehua/update vllm 0.15.0 (model-opt/simo!142)
0e2d30f [RENAME]update torchmo -> simo (model-opt/torchmo!104)
```

其中 `0e2d30f` 是引入/rename 后存在该字段；`62d9622` 是删除该字段。并且：

```bash
git log --oneline 62d9622..main -S'is_checkpoint_quant_serialized' -- simo/extensions/vllm_simo/quantization/quantization_config.py
```

没有输出，说明 `62d9622` 之后 main 上没有再引入/删除这个字符串。

### 删除前的代码位置

在你给的旧提交 `c1ea7eecb06c9dd103b9100626009e2364eb2d08`，字段还存在。

`simo/extensions/vllm_simo/quantization/quantization_config.py:153 SIMOConfig.__init__` 到 `simo/extensions/vllm_simo/quantization/quantization_config.py:160 SIMOConfig.__init__`：

```python
def __init__(
  self,
  is_checkpoint_quant_serialized: bool = False,
  target_to_specs: Optional[dict[str, Any]] = None,
  excludes: Optional[list[str]] = None,
  per_quant_opt: Optional[list[str]] = None,
  kv_cache_quant_algo: Optional[dict[str, Any]] = None,
  flash_comm: Optional[bool] = None,
) -> None:
```

`simo/extensions/vllm_simo/quantization/quantization_config.py:175 SIMOConfig.__init__` 到 `simo/extensions/vllm_simo/quantization/quantization_config.py:179 SIMOConfig.__init__`：

```python
if is_checkpoint_quant_serialized:
  self._load_from_serialized_checkpoint(
    target_to_specs, excludes, per_quant_opt, kv_cache_quant_algo, flash_comm
  )
  self._loaded_from_external = True
```

旧版还在 `simo/extensions/vllm_simo/quantization/quantization_config.py:261 SIMOConfig.__repr__` 到 `simo/extensions/vllm_simo/quantization/quantization_config.py:270 SIMOConfig.__repr__` 打印：

```python
f"  is_checkpoint_quant_serialized={self.is_checkpoint_quant_serialized},\n"
```

并在 `simo/extensions/vllm_simo/quantization/quantization_config.py:273 SIMOConfig._load_from_serialized_checkpoint` 到 `simo/extensions/vllm_simo/quantization/quantization_config.py:288 SIMOConfig._load_from_serialized_checkpoint` 里设置：

```python
self.is_checkpoint_quant_serialized = True
```

`simo/extensions/vllm_simo/quantization/quantization_config.py:545 SIMOConfig.from_config` 到 `simo/extensions/vllm_simo/quantization/quantization_config.py:552 SIMOConfig.from_config` 调用构造函数时也传了：

```python
return cls(
  is_checkpoint_quant_serialized=True,
  target_to_specs=target_to_specs,
  excludes=excludes,
  per_quant_opt=per_quant_opt,
  kv_cache_quant_algo=kv_cache_quant_algo,
  flash_comm=flash_comm,
)
```

### 删除 commit 前一版和后一版对比

在 `62d9622^`，也就是删除提交的父提交，字段仍然存在。

`simo/extensions/vllm_simo/quantization/quantization_config.py:156 SIMOConfig.__init__` 到 `simo/extensions/vllm_simo/quantization/quantization_config.py:163 SIMOConfig.__init__`：

```python
def __init__(
  self,
  is_checkpoint_quant_serialized: bool = False,
  target_to_specs: Optional[dict[str, Any]] = None,
  excludes: Optional[list[str]] = None,
  per_quant_opt: Optional[list[str]] = None,
  kv_cache_quant_algo: Optional[dict[str, Any]] = None,
  flash_comm: Optional[str] = None,
) -> None:
```

到 `62d9622` 之后，构造函数变成：

`simo/extensions/vllm_simo/quantization/quantization_config.py:107 SIMOConfig.__init__` 到 `simo/extensions/vllm_simo/quantization/quantization_config.py:114 SIMOConfig.__init__`：

```python
def __init__(
  self,
  target_to_specs: Optional[dict[str, Any]] = None,
  excludes: Optional[list[str]] = None,
  per_quant_opt: Optional[list[str]] = None,
  kv_cache_quant_algo: Optional[dict[str, Any]] = None,
  flash_comm: Optional[str] = None,
) -> None:
```

也就是说，`is_checkpoint_quant_serialized` 参数是在 `62d9622b14916b35312666bea386e4bf7ea32981` 这个 commit 里从 `SIMOConfig.__init__` 签名中删除的。

同一个提交还删除了 `SIMOConfig._load_from_serialized_checkpoint` 这条配置加载路径，并把 `SIMOConfig.from_config` 的构造调用改成不再传 `is_checkpoint_quant_serialized=True`：

`simo/extensions/vllm_simo/quantization/quantization_config.py:371 SIMOConfig.from_config` 到 `simo/extensions/vllm_simo/quantization/quantization_config.py:377 SIMOConfig.from_config`：

```python
return cls(
  target_to_specs=target_to_specs,
  excludes=excludes,
  per_quant_opt=per_quant_opt,
  kv_cache_quant_algo=kv_cache_quant_algo,
  flash_comm=flash_comm,
)
```

### 当前 main 状态

当前 `main` 是：

```text
4913f0f [feat] sglang online real quantize (model-opt/simo!166)
```

在当前 `main` 上：

```bash
git grep -n 'is_checkpoint_quant_serialized' main -- simo/extensions/vllm_simo/quantization/quantization_config.py
```

没有结果，说明这个字段在当前 main 的 `simo/extensions/vllm_simo/quantization/quantization_config.py` 里已经完全不存在。

## `pip install -e . --no-build-isolation` 报错原因

日志 `temp/ins.log.2026_05_25___17_54_18` 的根因是 CUDA toolkit 版本和 PyTorch 编译时 CUDA 版本不一致。

关键报错在日志里：

```text
RuntimeError: ('The detected CUDA version (%s) mismatches the version that was used to compilePyTorch (%s). Please make sure to use the same CUDA versions.', '12.8', '13.0')
```

也就是：

```text
当前构建时检测到的 CUDA / nvcc 版本: 12.8
当前 conda 环境里的 PyTorch 编译 CUDA 版本: 13.0
```

我在 `/data/like/miniconda3/envs/simo_sglang` 里确认到：

```text
torch = 2.11.0+cu130
torch.version.cuda = 13.0
torch.utils.cpp_extension.CUDA_HOME = /usr/local/cuda
/usr/local/cuda -> /usr/local/cuda-12.8
/usr/local/cuda/bin/nvcc --version = release 12.8, V12.8.93
```

所以虽然 pip 依赖里显示已经安装了 `cuda-toolkit==13.0.2`、`nvidia-cuda-runtime==13.0.*`、`nvidia-cuda-nvrtc==13.0.*` 等 Python wheel 包，但当前环境里没有可被 `torch.utils.cpp_extension` 使用的 CUDA 13.0 `nvcc`。PyTorch 扩展构建最终使用了系统 `/usr/local/cuda-12.8`，于是触发版本检查失败。

### 为什么安装会走到 CUDA 编译

`setup.py:7` 到 `setup.py:12` 从 PyTorch 导入 CUDA extension 构建工具：

```python
from torch.utils.cpp_extension import (
  CUDA_HOME,
  BuildExtension,
  CppExtension,
  CUDAExtension,
)
```

`setup.py:37` 到 `setup.py:46` 决定是否启用 CUDA 扩展：

```python
if not torch.cuda.is_available():
  print("PyTorch GPU support is not available. Skipping compilation of CUDA extensions")
if CUDA_HOME is None and torch.cuda.is_available():
  print("CUDA toolkit is not available. Skipping compilation of CUDA extensions")

use_cuda = torch.cuda.is_available() and CUDA_HOME is not None
extension = CUDAExtension if use_cuda else CppExtension
```

你当前机器上 GPU 可用，且 `CUDA_HOME=/usr/local/cuda` 不为空，所以 `use_cuda=True`，`setup.py` 使用 `CUDAExtension`。

`setup.py:67` 到 `setup.py:72` 会把 CUDA 源文件加入编译：

```python
sources = list(glob.glob(os.path.join(extension_dir, "**/*.cpp"), recursive=True))
cuda_sources = list(glob.glob(os.path.join(extension_dir, "**/*.cu"), recursive=True))

if use_cuda:
  sources += cuda_sources
  extra_compile_args["nvcc"].extend(["-DUSE_CUDA", "--extended-lambda"])
```

`setup.py:89` 到 `setup.py:95` 调用 `BuildExtension`：

```python
setup(
  ...
  ext_modules=get_extensions(),
  cmdclass={"build_ext": BuildExtension},
  ...
)
```

日志里的调用栈也对应这个过程：

```text
running build_ext
...
torch/utils/cpp_extension.py build_extensions
torch/utils/cpp_extension.py _check_cuda_version
raise RuntimeError(CUDA_MISMATCH_MESSAGE, cuda_str_version, torch.version.cuda)
```

因此这不是 pip 解析依赖失败，也不是 `--no-build-isolation` 本身的问题；失败发生在构建 `simo._C` CUDA extension 之前的 PyTorch CUDA 版本一致性检查。

### 推荐解决方案 1：安装/切换到 CUDA 13.0 nvcc

如果要保留当前 PyTorch `2.11.0+cu130`，构建扩展时也必须使用 CUDA 13.0 的 toolkit/nvcc。

如果系统上有 CUDA 13.0，例如 `/usr/local/cuda-13.0`，可以这样跑：

```bash
conda activate /data/like/miniconda3/envs/simo_sglang
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

python - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME
print("torch.version.cuda =", torch.version.cuda)
print("CUDA_HOME =", CUDA_HOME)
PY

pip install -e . --no-build-isolation
```

如果系统没有 CUDA 13.0，需要先安装 CUDA 13.0 toolkit，至少要有匹配的 `nvcc`。用 conda 的话方向是：

```bash
conda activate /data/like/miniconda3/envs/simo_sglang
conda install -c nvidia cuda-nvcc=13.0 cuda-toolkit=13.0
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
pip install -e . --no-build-isolation
```

安装后先确认：

```bash
$CUDA_HOME/bin/nvcc --version
python -c "import torch; print(torch.version.cuda)"
```

两边都应该是 13.0。

### 推荐解决方案 2：换成 CUDA 12.8 版本的 PyTorch

如果你必须使用系统 `/usr/local/cuda-12.8` 编译，那么 PyTorch 也要换成 CUDA 12.8 构建版本。目标是让：

```text
torch.version.cuda == 12.8
/usr/local/cuda/bin/nvcc --version == 12.8
```

当前环境是 `torch 2.11.0+cu130`，和系统 CUDA 12.8 不匹配。需要安装与 CUDA 12.8 匹配的 PyTorch build，或者换到已有的 CUDA 12.8 PyTorch conda 环境。

### 临时绕过：只装 Python editable，不编 CUDA 扩展

如果只是想让 Python 包以 editable 方式挂进去，不需要 `simo._C` CUDA 扩展，可以临时让 `torch.cuda.is_available()` 变成 false，从而走 `CppExtension`/CPU-only 路径：

```bash
CUDA_VISIBLE_DEVICES="" pip install -e . --no-build-isolation
```

但这只是临时绕过。因为 `setup.py:45` 的 `use_cuda` 会变成 false，`setup.py:70` 到 `setup.py:72` 不会加入 `.cu` 源文件。后续如果运行需要 CUDA extension 的 SIMO 功能，仍然会缺少或不能使用对应 CUDA kernel。

### 最终判断

这次安装失败的直接原因是：

```text
构建 simo._C 时，PyTorch 发现 nvcc/CUDA_HOME 是 CUDA 12.8，
但当前 PyTorch 是 CUDA 13.0 编译的 torch 2.11.0+cu130，
因此 torch.utils.cpp_extension 主动中止 build_ext。
```

最稳的修复是让构建用的 CUDA toolkit/nvcc 版本和 `torch.version.cuda` 完全一致。对当前 `/data/like/miniconda3/envs/simo_sglang` 来说，就是改用 CUDA 13.0 的 `nvcc`，或者把 PyTorch 换成 CUDA 12.8 构建版本。

## SGLang 启动日志 `torchvision::nms does not exist` 报错原因

日志 `temp/sglang_server.30121.2026_05_25___22_36_43` 里只有一个导入期错误，服务还没有进入模型加载阶段，所以这不是 `/data/like/hf-models/DeepSeek-V2-Lite-Chat-16B_A2.4B-strip-layers/` 模型目录的问题，也不是 `--kv-cache-dtype fp8_e4m3` 或 `--attention-backend triton` 的问题。

直接错误是：

```text
RuntimeError: operator torchvision::nms does not exist
```

调用链是：

```text
python -m sglang.launch_server
  -> python/sglang/__init__.py
  -> python/sglang/srt/utils/common.py
  -> torchvision
  -> torchvision/_meta_registrations.py
  -> torch.library.register_fake("torchvision::nms")
```

关键位置：

`python/sglang/__init__.py:29` 导入 SGLang patch：

```python
from sglang.srt.utils.hf_transformers_patches import apply_all as _apply_hf_patches
```

`python/sglang/srt/utils/common.py:91` 在模块 import 阶段直接导入 torchvision：

```python
from torchvision.io import decode_jpeg
```

然后进入当前 conda 环境的 torchvision。`/data/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/torchvision/__init__.py:7` 到 `/data/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/torchvision/__init__.py:8`：

```python
from . import extension  # load _C extension first
from torchvision import _meta_registrations, datasets, io, models, ops, transforms, utils
```

`/data/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/torchvision/_meta_registrations.py:163`：

```python
@torch.library.register_fake("torchvision::nms")
```

这里要注册 `torchvision::nms` 的 fake/meta 实现。但这个 op 应该先由 torchvision 的 C++ extension `_C.so` 注册；现在 `_C.so` 没加载成功，所以 dispatcher 里没有 `torchvision::nms` 这个 operator，最终报：

```text
RuntimeError: operator torchvision::nms does not exist
```

### 根因：torch 和 torchvision 的 CUDA build 不匹配

当前 `/data/like/miniconda3/envs/simo_sglang` 里实际版本是：

```text
torch = 2.11.0+cu128
torch.version.cuda = 12.8
torchvision = 0.26.0+cu130
```

`/data/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/torchvision/version.py` 显示：

```python
__version__ = '0.26.0+cu130'
```

也就是 torchvision 是 CUDA 13.0 构建；但 torch 是 CUDA 12.8 构建。

进一步验证：直接加载 torchvision C++ extension 会失败：

```python
import torch
torch.ops.load_library(
  "/data/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/torchvision/_C.so"
)
```

结果是：

```text
OSError: Could not load this library: .../torchvision/_C.so
```

`ldd /data/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/torchvision/_C.so` 也能看到它依赖：

```text
libcudart.so.13 => not found
```

虽然环境里存在一些 `nvidia/cu13` 文件，但当前 torch 运行时是 cu128，torchvision `_C.so` 作为 cu130 扩展不能和当前 torch/cu128 正常匹配加载。因此 `_C.so` 没有注册 `torchvision::nms`，后续 `_meta_registrations.py` 注册 fake nms 就失败。

### 为什么会在纯文本 DeepSeek 模型启动时触发 torchvision

即使你启动的是文本模型 DeepSeek-V2，`python/sglang/srt/utils/common.py:91` 也在 import 阶段全局执行：

```python
from torchvision.io import decode_jpeg
```

所以只要 `python -m sglang.launch_server` 导入 `sglang`，就会导入 torchvision。当前错误发生在这个 import 阶段，和是否真的处理图像无关。

### 解决办法

最稳的修复是让 torch 和 torchvision 使用同一个 CUDA build。

当前 torch 是：

```text
2.11.0+cu128
```

所以应安装匹配的 torchvision：

```text
torchvision 0.26.0+cu128
```

可以先卸载当前 cu130 torchvision，再安装 cu128 版本：

```bash
conda activate /data/like/miniconda3/envs/simo_sglang
pip uninstall -y torchvision
pip install --no-cache-dir --force-reinstall torchvision==0.26.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
```

如果你使用的是内部镜像，需要确保镜像里拿到的是 `0.26.0+cu128`，而不是没有 local version 标记的 `0.26.0` 或 `0.26.0+cu130`。

安装后验证：

```bash
/data/like/miniconda3/envs/simo_sglang/bin/python - <<'PY'
import torch
import torchvision
print("torch", torch.__version__, torch.version.cuda)
print("torchvision", torchvision.__version__)
from torchvision.io import decode_jpeg
print("decode_jpeg import ok")
PY
```

期望输出里版本类似：

```text
torch 2.11.0+cu128 12.8
torchvision 0.26.0+cu128
decode_jpeg import ok
```

另一个方向是把 torch 也换成 CUDA 13.0 build，让它和当前 `torchvision 0.26.0+cu130` 匹配。但你当前 SGLang 环境已经是 `torch 2.11.0+cu128`，通常改 torchvision 风险更小。

### 临时绕过思路

如果你只是跑纯文本模型，不想处理图像，可以考虑把 `python/sglang/srt/utils/common.py:91` 的 torchvision 导入改成 lazy import，只在真正调用 JPEG decode 时再导入。但这是代码改动，不是环境根因修复；而且 SGLang 其他多模态路径仍可能导入 torchvision。

所以推荐优先修复环境：

```text
torch 2.11.0+cu128 需要匹配 torchvision 0.26.0+cu128
```

### 最终判断

本次 SGLang 启动失败的根因是：

```text
当前 conda 环境里 torch 和 torchvision 的 CUDA build 不一致：
torch 是 2.11.0+cu128，
torchvision 是 0.26.0+cu130。

torchvision/_C.so 加载失败，导致 torchvision::nms 没有注册，
随后 torchvision/_meta_registrations.py 注册 fake nms 时报
RuntimeError: operator torchvision::nms does not exist。
```

## 2026-05-25 23:03:19 SGLang 新日志分析

日志文件：

```text
temp/sglang_server.30121.2026_05_25___23_03_19
```

### 结论

`torchvision` 的 CUDA 版本已经修好了：

```text
torch       2.11.0+cu128, torch.version.cuda = 12.8
torchvision 0.26.0+cu128
```

这次的新问题不是 `torchvision::nms`，而是环境里还有两个 CUDA/PyTorch 二进制扩展没有和当前 `torch 2.11.0+cu128` 对齐：

1. `sgl-deep-gemm/deep_gemm` 仍然带着 CUDA 13 依赖，启动时加载 `deep_gemm/_C.so` 失败。
2. `/data/like/package/sglang_kernel_src/sgl-kernel/python/sgl_kernel/sm90/common_ops.abi3.so` 和当前 PyTorch ABI 不匹配，加载时报 undefined symbol。

### 第一处致命错误：deep_gemm 仍依赖 CUDA 13

traceback 的主错误是：

```text
RuntimeError: Failed to load dynamic shared library
/data/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/deep_gemm/_C.so
libcudart.so.13: cannot open shared object file: No such file or directory
```

我检查了这个 so 的动态依赖：

```bash
ldd /data/like/miniconda3/envs/simo_sglang/lib/python3.12/site-packages/deep_gemm/_C.so
```

关键输出是：

```text
libcudart.so.13    => not found
libnvrtc.so.13     => not found
libcublasLt.so.13  => not found
libcublas.so.13    => not found
```

但当前环境和系统 CUDA 是：

```text
torch.version.cuda = 12.8
/usr/local/cuda -> /usr/local/cuda-12.8
nvcc release 12.8
```

所以当前 `deep_gemm/_C.so` 是 CUDA 13 版本，和现在的 cu128 环境不一致。

触发链路是：

```text
python/sglang/launch_server.py:66:<module>
  -> python/sglang/srt/server_args.py:7643:prepare_server_args
  -> python/sglang/srt/server_args.py:6935:ServerArgs.from_cli_args
  -> python/sglang/srt/server_args.py:902:ServerArgs.__post_init__
  -> python/sglang/srt/server_args.py:1299:ServerArgs._handle_piecewise_cuda_graph
  -> python/sglang/srt/server_args.py:6981:ServerArgs.get_model_config
  -> python/sglang/srt/configs/model_config.py:27:<module>
  -> python/sglang/srt/layers/quantization/fp8_kernel.py:31:<module>
  -> python/sglang/srt/layers/deep_gemm_wrapper/configurer.py:32:<module>
  -> python/sglang/srt/layers/deep_gemm_wrapper/configurer.py:17:_compute_enable_deep_gemm
  -> python/sglang/srt/layers/deep_gemm_wrapper/configurer.py:25:import deep_gemm
```

`python/sglang/srt/layers/deep_gemm_wrapper/configurer.py:24-29:_compute_enable_deep_gemm` 现在的逻辑是：

```python
try:
    import deep_gemm
except ImportError:
    return False

return envs.SGLANG_ENABLE_JIT_DEEPGEMM.get()
```

这里有一个坑：它只 catch `ImportError`，但这次 `deep_gemm` 不是找不到包，而是包存在、`_C.so` 加载时抛了 `RuntimeError`。所以 SGLang 不会自动禁用 DeepGEMM，而是直接启动失败。

另外，`SGLANG_ENABLE_JIT_DEEPGEMM=0` 也不能绕过这个错误，因为代码是先 `import deep_gemm`，再读取 `SGLANG_ENABLE_JIT_DEEPGEMM`。我实际验证过，带这个环境变量仍然会在同一处因为 `libcudart.so.13` 报错。

### 第二处问题：sgl_kernel/common_ops ABI 不匹配

日志里还有这一段：

```text
[sgl_kernel] Found architecture-specific library:
/data/like/package/sglang_kernel_src/sgl-kernel/python/sgl_kernel/sm90/common_ops.abi3.so

ImportError:
undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib
```

加载逻辑在：

```text
sgl-kernel/python/sgl_kernel/__init__.py:18-23:<module>
sgl-kernel/python/sgl_kernel/load_utils.py:48:_load_architecture_specific_ops
sgl-kernel/python/sgl_kernel/load_utils.py:72-99:_load_architecture_specific_ops
```

`undefined symbol: c10::cuda::c10_cuda_check_implementation...` 说明这个 `common_ops.abi3.so` 编译时使用的 PyTorch/CUDA ABI 和当前运行时的 `torch 2.11.0+cu128` 不一致。它不只是缺少 `libcudart.so`，而是 C++/PyTorch 符号对不上。

所以即使先修好 `deep_gemm`，后面大概率还会继续卡在 `sgl_kernel` 的 `common_ops.abi3.so`。

### 建议修复顺序

优先把整个环境统一到 cu128，而不是只修 `torchvision`：

1. 处理 `sgl-deep-gemm/deep_gemm`。

当前包名是：

```text
sgl-deep-gemm 0.1.0
```

如果这次启动不需要 DeepGEMM，最直接的验证方式是先让 `deep_gemm` 包不可导入，使 `configurer.py:24-27:_compute_enable_deep_gemm` 走 `ImportError -> return False`。例如可以卸载：

```bash
conda activate /data/like/miniconda3/envs/simo_sglang
pip uninstall -y sgl-deep-gemm
```

如果后续需要 DeepGEMM，则应该安装或重新构建与 `torch 2.11.0+cu128`、CUDA 12.8 匹配的 `sgl-deep-gemm`，不能继续使用当前这个依赖 `libcudart.so.13` 的 `_C.so`。

2. 重新安装或重编译 `sgl_kernel`。

日志实际加载的是源码树里的：

```text
/data/like/package/sglang_kernel_src/sgl-kernel/python/sgl_kernel/sm90/common_ops.abi3.so
```

这个 so 需要按当前环境重建，或者安装一个和当前 `torch 2.11.0+cu128` 兼容的 `sglang-kernel/sgl-kernel` wheel。否则会继续报：

```text
undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib
```

3. 验证当前环境的二进制一致性。

可以先跑：

```bash
/data/like/miniconda3/envs/simo_sglang/bin/python - <<'PY'
import torch
import torchvision
print("torch", torch.__version__, torch.version.cuda)
print("torchvision", torchvision.__version__)

try:
    import deep_gemm
    print("deep_gemm import ok")
except Exception as e:
    print("deep_gemm import failed:", repr(e))

try:
    import sgl_kernel
    print("sgl_kernel import ok")
except Exception as e:
    print("sgl_kernel import failed:", repr(e))
PY
```

期望至少达到：

```text
torch 2.11.0+cu128 12.8
torchvision 0.26.0+cu128
deep_gemm 要么 import ok，要么因为包不存在而由 SGLang 禁用
sgl_kernel import ok
```

### 最终判断

`temp/sglang_server.30121.2026_05_25___23_03_19` 的根因是：换成 cu128 的 `torch/torchvision` 后，环境里仍残留了不匹配的二进制扩展。

当前第一个 fatal error 是：

```text
deep_gemm/_C.so 依赖 CUDA 13 的 libcudart.so.13/libnvrtc.so.13/libcublas*.so.13，
但当前环境是 CUDA 12.8 / torch 2.11.0+cu128。
```

后续还需要修：

```text
sgl-kernel/python/sgl_kernel/sm90/common_ops.abi3.so
和当前 PyTorch 2.11.0+cu128 ABI 不匹配。
```

所以这次不是单点的 `torchvision` 问题，而是 CUDA 13 -> CUDA 12.8 切换后，`deep_gemm` 和 `sgl_kernel` 也必须同步重装或重编译。

## main 分支 extend_attention.py 新增 k_scale/v_scale 参数分析

对比对象：

```text
old: main-2026_05_25___14_36_25
new: main
file: python/sglang/srt/layers/attention/triton_ops/extend_attention.py
```

### 结论

`_fwd_kernel` 新增 `k_scale`、`v_scale` 是为了让 Triton attention backend 支持 FP8 KV cache。

这个改动来自 commit：

```text
07b8d763e feat: Add FP8 KV cache support for Triton attention backend (#18882)
```

核心原因是：KV cache 如果用 FP8 保存，写入 cache 前会把原始 K/V 除以 scale 再转成 FP8；attention 从 cache 读 K/V 时，必须把 scale 乘回来，否则 prefix KV 的 QK logits 和 PV 输出都会少一个反量化因子。

所以：

```text
k_scale: 读 FP8 K cache 后用于恢复 K 的反量化 scale，影响 QK logits。
v_scale: 读 FP8 V cache 后用于恢复 V 的反量化 scale，影响 attention output。
```

非 FP8 KV cache 路径下，这两个值传 `1.0`，行为和旧版本一致。

### 旧版本的问题

旧分支 `main-2026_05_25___14_36_25` 的 `python/sglang/srt/layers/attention/triton_ops/extend_attention.py:220:_fwd_kernel` 只有 `sm_scale`，没有 `k_scale/v_scale`。

旧逻辑里 prefix 阶段从 KV cache 读 K/V：

```text
python/sglang/srt/layers/attention/triton_ops/extend_attention.py:327:_fwd_kernel
```

旧分支对应逻辑是：

```text
qk = dot(q, k)
qk *= sm_scale
acc += dot(p, v)
```

也就是默认 `K_Buffer`、`V_Buffer` 里读出来的值已经是原始数值。这个假设在 bf16/fp16 KV cache 下成立，但在 FP8 KV cache 下不成立。

### 新版本的具体改动

新分支 `main` 中，`python/sglang/srt/layers/attention/triton_ops/extend_attention.py:228:_fwd_kernel` 的函数签名新增：

```text
python/sglang/srt/layers/attention/triton_ops/extend_attention.py:242:_fwd_kernel  sm_scale
python/sglang/srt/layers/attention/triton_ops/extend_attention.py:243:_fwd_kernel  k_scale
python/sglang/srt/layers/attention/triton_ops/extend_attention.py:244:_fwd_kernel  v_scale
```

在 prefix KV 阶段，K 从 cache 读出后参与 QK：

```text
python/sglang/srt/layers/attention/triton_ops/extend_attention.py:374:_fwd_kernel
python/sglang/srt/layers/attention/triton_ops/extend_attention.py:380:_fwd_kernel
python/sglang/srt/layers/attention/triton_ops/extend_attention.py:385:_fwd_kernel
python/sglang/srt/layers/attention/triton_ops/extend_attention.py:398:_fwd_kernel
```

新逻辑是：

```python
qk = tl.dot(q.to(k.dtype), k)
qk *= sm_scale * k_scale
```

含义是：

```text
真实 K = FP8_cache_K * k_scale
所以 QK logits = Q @ (FP8_cache_K * k_scale)^T * sm_scale
            = (Q @ FP8_cache_K^T) * sm_scale * k_scale
```

V 从 cache 读出后参与 PV：

```text
python/sglang/srt/layers/attention/triton_ops/extend_attention.py:416:_fwd_kernel
python/sglang/srt/layers/attention/triton_ops/extend_attention.py:421:_fwd_kernel
python/sglang/srt/layers/attention/triton_ops/extend_attention.py:427:_fwd_kernel
```

新逻辑是：

```python
acc = acc * re_scale[:, None] + tl.dot(p, v) * v_scale
```

含义是：

```text
真实 V = FP8_cache_V * v_scale
所以 PV = P @ (FP8_cache_V * v_scale)
      = (P @ FP8_cache_V) * v_scale
```

### 为什么只在 stage 1 prefix 部分乘 scale

`python/sglang/srt/layers/attention/triton_ops/extend_attention.py:330:_fwd_kernel` 注释写的是：

```text
stage 1: compute scores with prefix
```

这一段读的是历史 prefix KV cache：

```text
K_Buffer / V_Buffer
```

这些 buffer 可能是 FP8 cache，所以必须乘 `k_scale/v_scale`。

但是 `python/sglang/srt/layers/attention/triton_ops/extend_attention.py:431:_fwd_kernel` 开始的 stage 2 处理的是当前 extend chunk 内部的三角注意力，读的是：

```text
K_Extend / V_Extend
```

这些是当前 forward 刚算出来的 K/V，本身还是计算 dtype，不是从 FP8 KV cache 读出的历史值。因此 stage 2 仍然保持：

```text
qk *= sm_scale
acc += dot(p, v)
```

不额外乘 `k_scale/v_scale`。

换句话说，这个 kernel 里有两类 K/V：

```text
prefix K/V: 来自 KV cache，可能被 FP8 量化，需要 descale。
extend K/V: 当前 forward 的原始 K/V，不需要 descale。
```

### 调用方如何传入 k_scale/v_scale

`python/sglang/srt/layers/attention/triton_ops/extend_attention.py:559:extend_attention_fwd` 的 Python wrapper 也新增了两个参数：

```text
python/sglang/srt/layers/attention/triton_ops/extend_attention.py:573:extend_attention_fwd  k_scale
python/sglang/srt/layers/attention/triton_ops/extend_attention.py:574:extend_attention_fwd  v_scale
```

并在 launch `_fwd_kernel` 时传进去：

```text
python/sglang/srt/layers/attention/triton_ops/extend_attention.py:630:extend_attention_fwd  sm_scale
python/sglang/srt/layers/attention/triton_ops/extend_attention.py:631:extend_attention_fwd  k_scale
python/sglang/srt/layers/attention/triton_ops/extend_attention.py:632:extend_attention_fwd  v_scale
```

上层调用在：

```text
python/sglang/srt/layers/attention/triton_backend.py:900:TritonAttnBackend.forward_extend
```

这里先决定是否有 layer 级别的 scale：

```text
python/sglang/srt/layers/attention/triton_backend.py:1002:TritonAttnBackend.forward_extend
python/sglang/srt/layers/attention/triton_backend.py:1003:TritonAttnBackend.forward_extend
python/sglang/srt/layers/attention/triton_backend.py:1004:TritonAttnBackend.forward_extend
python/sglang/srt/layers/attention/triton_backend.py:1006:TritonAttnBackend.forward_extend
python/sglang/srt/layers/attention/triton_backend.py:1007:TritonAttnBackend.forward_extend
```

逻辑是：

```python
if layer.k_scale is not None and layer.v_scale is not None:
    k_descale = layer.k_scale_float
    v_descale = layer.v_scale_float
else:
    k_descale = 1.0
    v_descale = 1.0
```

然后传给 `extend_attention_fwd`：

```text
python/sglang/srt/layers/attention/triton_backend.py:1009:TritonAttnBackend.forward_extend
python/sglang/srt/layers/attention/triton_backend.py:1023:TritonAttnBackend.forward_extend
python/sglang/srt/layers/attention/triton_backend.py:1024:TritonAttnBackend.forward_extend
```

### k_scale/v_scale 从哪里来

`RadixAttention` 初始化时默认没有 scale：

```text
python/sglang/srt/layers/radix_attention.py:54:RadixAttention
python/sglang/srt/layers/radix_attention.py:90:RadixAttention.__init__
python/sglang/srt/layers/radix_attention.py:91:RadixAttention.__init__
python/sglang/srt/layers/radix_attention.py:92:RadixAttention.__init__
python/sglang/srt/layers/radix_attention.py:93:RadixAttention.__init__
```

默认值是：

```python
self.k_scale = None
self.v_scale = None
self.k_scale_float = None
self.v_scale_float = None
```

如果启用了 KV cache quantization，量化方法会创建这两个参数：

```text
python/sglang/srt/layers/quantization/kv_cache.py:18:BaseKVCacheMethod
python/sglang/srt/layers/quantization/kv_cache.py:32:BaseKVCacheMethod.create_weights
python/sglang/srt/layers/quantization/kv_cache.py:39:BaseKVCacheMethod.create_weights
python/sglang/srt/layers/quantization/kv_cache.py:42:BaseKVCacheMethod.create_weights
```

加载权重后，`process_weights_after_loading` 会把 checkpoint 里的 scale 或默认 scale 写回 layer：

```text
python/sglang/srt/layers/quantization/kv_cache.py:51:BaseKVCacheMethod.process_weights_after_loading
python/sglang/srt/layers/quantization/kv_cache.py:52:BaseKVCacheMethod.process_weights_after_loading
python/sglang/srt/layers/quantization/kv_cache.py:59:BaseKVCacheMethod.process_weights_after_loading
python/sglang/srt/layers/quantization/kv_cache.py:82:BaseKVCacheMethod.process_weights_after_loading
python/sglang/srt/layers/quantization/kv_cache.py:83:BaseKVCacheMethod.process_weights_after_loading
python/sglang/srt/layers/quantization/kv_cache.py:84:BaseKVCacheMethod.process_weights_after_loading
python/sglang/srt/layers/quantization/kv_cache.py:85:BaseKVCacheMethod.process_weights_after_loading
```

关键语义在这个文件注释里已经说明：

```text
python/sglang/srt/layers/quantization/kv_cache.py:20:BaseKVCacheMethod
python/sglang/srt/layers/quantization/kv_cache.py:22:BaseKVCacheMethod
python/sglang/srt/layers/quantization/kv_cache.py:23:BaseKVCacheMethod
python/sglang/srt/layers/quantization/kv_cache.py:24:BaseKVCacheMethod
```

即 `k_scale/v_scale` 用于：

```text
1. 写 KV cache 前量化 K/V。
2. 从 KV cache 取出 K/V 后反量化 K/V。
```

KV cache 写入处在：

```text
python/sglang/srt/mem_cache/memory_pool.py:1087:MHATokenToKVPool.set_kv_buffer
python/sglang/srt/mem_cache/memory_pool.py:1131:MHATokenToKVPool.set_kv_buffer
python/sglang/srt/mem_cache/memory_pool.py:1132:MHATokenToKVPool.set_kv_buffer
python/sglang/srt/mem_cache/memory_pool.py:1133:MHATokenToKVPool.set_kv_buffer
python/sglang/srt/mem_cache/memory_pool.py:1134:MHATokenToKVPool.set_kv_buffer
python/sglang/srt/mem_cache/memory_pool.py:1135:MHATokenToKVPool.set_kv_buffer
python/sglang/srt/mem_cache/memory_pool.py:1136:MHATokenToKVPool.set_kv_buffer
python/sglang/srt/mem_cache/memory_pool.py:1137:MHATokenToKVPool.set_kv_buffer
```

逻辑是：

```python
if cache_k.dtype != self.dtype:
    if k_scale is not None:
        cache_k.div_(k_scale)
    if v_scale is not None:
        cache_v.div_(v_scale)
    cache_k = cache_k.to(self.dtype)
    cache_v = cache_v.to(self.dtype)
```

如果 `self.dtype` 是 FP8，写入 cache 的就是：

```text
K_cache_fp8 = cast_fp8(K_original / k_scale)
V_cache_fp8 = cast_fp8(V_original / v_scale)
```

因此 attention 读取 cache 后要恢复：

```text
K_original ≈ K_cache_fp8 * k_scale
V_original ≈ V_cache_fp8 * v_scale
```

这正是 `extend_attention.py` 新增参数的用途。

### 和 decode attention 的一致性

这次改动也让 extend 路径和 decode 路径一致。

decode 路径在：

```text
python/sglang/srt/layers/attention/triton_ops/decode_attention.py:745:decode_attention_fwd
python/sglang/srt/layers/attention/triton_ops/decode_attention.py:756:decode_attention_fwd
python/sglang/srt/layers/attention/triton_ops/decode_attention.py:757:decode_attention_fwd
python/sglang/srt/layers/attention/triton_ops/decode_attention.py:758:decode_attention_fwd
```

decode 已经把 `k_scale` 合入 QK scale：

```text
python/sglang/srt/layers/attention/triton_ops/decode_attention.py:784:decode_attention_fwd
python/sglang/srt/layers/attention/triton_ops/decode_attention.py:803:decode_attention_fwd
```

并在 reduce V 阶段乘回 `v_scale`：

```text
python/sglang/srt/layers/attention/triton_ops/decode_attention.py:522:_fwd_kernel_stage2
python/sglang/srt/layers/attention/triton_ops/decode_attention.py:527:_fwd_kernel_stage2
python/sglang/srt/layers/attention/triton_ops/decode_attention.py:590:_fwd_kernel_stage2
python/sglang/srt/layers/attention/triton_ops/decode_attention.py:592:_fwd_kernel_stage2
```

新版本的 extend attention 相当于把这套逻辑补到了 prefill/extend 访问 prefix KV cache 的路径上。

### unified deterministic kernel 也同步改了

新分支里 `python/sglang/srt/layers/attention/triton_ops/extend_attention.py:705:_fwd_kernel_unified` 也增加了等价逻辑。

wrapper 在：

```text
python/sglang/srt/layers/attention/triton_ops/extend_attention.py:956:extend_attention_fwd_unified
python/sglang/srt/layers/attention/triton_ops/extend_attention.py:961:extend_attention_fwd_unified
python/sglang/srt/layers/attention/triton_ops/extend_attention.py:962:extend_attention_fwd_unified
```

launch 时把 `sm_scale * k_scale` 传进 kernel：

```text
python/sglang/srt/layers/attention/triton_ops/extend_attention.py:1042:extend_attention_fwd_unified
python/sglang/srt/layers/attention/triton_ops/extend_attention.py:1043:extend_attention_fwd_unified
```

kernel 内部：

```text
python/sglang/srt/layers/attention/triton_ops/extend_attention.py:903:_fwd_kernel_unified
python/sglang/srt/layers/attention/triton_ops/extend_attention.py:951:_fwd_kernel_unified
```

即：

```python
qk *= sm_scale_withk
tl.store(O + offs_o, acc / deno[:, None] * v_scale, ...)
```

`TritonAttnBackend._forward_extend_unified` 里同样取 `layer.k_scale_float/layer.v_scale_float`：

```text
python/sglang/srt/layers/attention/triton_backend.py:1034:TritonAttnBackend._forward_extend_unified
python/sglang/srt/layers/attention/triton_backend.py:1138:TritonAttnBackend._forward_extend_unified
python/sglang/srt/layers/attention/triton_backend.py:1139:TritonAttnBackend._forward_extend_unified
python/sglang/srt/layers/attention/triton_backend.py:1140:TritonAttnBackend._forward_extend_unified
python/sglang/srt/layers/attention/triton_backend.py:1142:TritonAttnBackend._forward_extend_unified
python/sglang/srt/layers/attention/triton_backend.py:1143:TritonAttnBackend._forward_extend_unified
python/sglang/srt/layers/attention/triton_backend.py:1146:TritonAttnBackend._forward_extend_unified
python/sglang/srt/layers/attention/triton_backend.py:1151:TritonAttnBackend._forward_extend_unified
python/sglang/srt/layers/attention/triton_backend.py:1152:TritonAttnBackend._forward_extend_unified
```

### 最终理解

这个改动不是为了改变普通 bf16/fp16 attention 的数学，而是为了修正 FP8 KV cache 下 Triton extend attention 的数值。

等价公式是：

```text
FP8 cache 写入：
K_cache = quantize(K / k_scale)
V_cache = quantize(V / v_scale)

attention 读取 prefix cache：
logits = Q @ (K_cache * k_scale)^T * sm_scale
       = (Q @ K_cache^T) * sm_scale * k_scale

output = softmax(logits) @ (V_cache * v_scale)
       = (softmax(logits) @ V_cache) * v_scale
```

旧分支缺少这两个乘法，所以一旦 Triton extend attention 读的是 FP8 KV cache，prefix 部分的 attention 会按被压缩后的 K/V 数值计算，结果偏小或错误。新分支通过 `k_scale/v_scale` 把 FP8 cache 的反量化补回来了。

---

## Q: sglang 的 triton_backend.py 中，`attn_logits = self.forward_metadata.swa_attn_logits` 什么时候触发？

### 代码位置

`python/sglang/srt/layers/attention/triton_backend.py:1247-1252`，在 `forward_decode` 方法中：

```python
attn_logits = self.forward_metadata.attn_logits
if (
    self.forward_metadata.swa_attn_logits is not None
    and layer.v_head_dim == self.swa_v_head_dim
):
    attn_logits = self.forward_metadata.swa_attn_logits
```

### 触发条件

这行代码的触发需要**两个条件同时满足**：

#### 条件 1：`sliding_window_size` 不为 None 且 SWA 与 full-attention 的 v_head_dim 不同

在 `TritonAttnBackend.__init__` 中（`triton_backend.py:135-137`）：

```python
full_v_head_dim = model_runner.model_config.v_head_dim
swa_v_head_dim = model_runner.model_config.swa_v_head_dim
if self.sliding_window_size is not None and swa_v_head_dim != full_v_head_dim:
    self.v_head_dim = full_v_head_dim
    self.swa_v_head_dim = swa_v_head_dim
```

`self.swa_v_head_dim` 只有在**模型同时存在 sliding window 层和 full-attention 层，且它们的 v_head_dim 不同**时才会被设为非 None 值。

当 `self.swa_v_head_dim is not None` 时，在 `init_forward_metadata` 的 decode 路径（`triton_backend.py:348-355`）会额外分配一个 `swa_attn_logits` buffer：

```python
if self.swa_v_head_dim is not None:
    swa_attn_logits = torch.empty(
        (bs, self.num_head, self.max_kv_splits, self.swa_v_head_dim),
        dtype=torch.float32, device=self.device,
    )
else:
    swa_attn_logits = None
```

#### 条件 2：当前 layer 恰好是 SWA 层（`layer.v_head_dim == self.swa_v_head_dim`）

在 `forward_decode` 中，每一层都会根据自身的 `v_head_dim` 决定使用哪个 buffer：
- 如果当前层是 **full-attention 层**（`v_head_dim == self.v_head_dim`），使用 `self.forward_metadata.attn_logits`
- 如果当前层是 **SWA 层**（`v_head_dim == self.swa_v_head_dim`），使用 `self.forward_metadata.swa_attn_logits`

### 为什么需要两个 buffer？

代码注释（`triton_backend.py:127-132`）解释：

```python
# The decode triton kernel derives attn_lse offsets from attn_logits
# strides via integer division by v_head_dim (the "// Lv" trick in
# _fwd_kernel_stage1/stage2), so attn_logits.shape[-1] must exactly
# match the layer's v_head_dim. For hybrid SWA models where SWA and
# full-attention layers use different v_head_dim (e.g. Gemma 4:
# swa=256, full=512), we allocate a second buffer for SWA layers.
```

Triton decode kernel 内部通过 `attn_logits` 的 stride 除以 `v_head_dim` 来计算 `attn_lse` 的偏移量（`// Lv` trick）。因此 `attn_logits.shape[-1]` 必须**精确匹配**该层实际的 `v_head_dim`，否则计算出的 offset 会出错。对于 SWA 和 full-attention 层共用同一 batch 但 v_head_dim 不同的模型，就需要两个不同 shape 的 buffer。

### 哪些模型满足条件？

核心场景是 **Gemma 4** 系列（如 `google/gemma-4-27b`）。

在 `python/sglang/srt/utils/hf_transformers/config.py:136-159` 中，对 Gemma 4 做了特殊处理：

```python
if config.model_type in ("gemma4", "gemma4_assistant"):
    text_config = config.text_config
    # SWA 层: head_dim = 256
    swa_head_dim = text_config.head_dim       # ← 例如 256
    text_config.swa_v_head_dim = swa_head_dim  # ← 256

    # Full-attention 层: head_dim = 512
    if global_head_dim is not None:
        text_config.head_dim = global_head_dim  # ← 例如 512，覆盖了原始值
```

Gemma 4 的 HuggingFace config 中，基础属性（`head_dim`、`num_key_value_heads`）描述的是 **SWA 层**，全局属性（`global_head_dim`、`num_global_key_value_heads`）描述的是 **full-attention 层**。SGLang 的约定相反：基础属性代表 full-attention 层，`swa_*` 代表 SWA 层。因此 config.py 做了这个 swap 操作，最终形成：

| 属性 | 值 | 含义 |
|------|-----|------|
| `v_head_dim` (= `head_dim`) | 512 | Full-attention 层的 v_head_dim |
| `swa_v_head_dim` | 256 | SWA 层的 v_head_dim |

模型结构为 **交替排列**：SWA 层和 full-attention 层交叉出现（pattern: full, swa, full, swa, ...）。

### 是否需要传额外参数？

**不需要**。SGLang 会自动从模型 `config.json` 中读取 `head_dim`、`global_head_dim` 等字段，在 `ModelConfig` 初始化时（`model_config.py:587-601`）自动推导出 `swa_v_head_dim`，并在 `TritonAttnBackend` 构造时自动判断是否需要分配两份 buffer。

只需正常启动 SGLang 服务即可，例如：

```bash
python -m sglang.launch_server \
    --model google/gemma-4-27b \
    --trust-remote-code
```

SGLang 会在内部自动检测 hybrid SWA 结构并启用双 buffer 机制。

---

## Q: 分析日志 `Llama3.1-8B-Instruct_tp1_quant-simo_w4a4_mxfp.log` 报错原因及修复

### 报错分析

**报错类型**：`httpx.ConnectError: [Errno 101] Network is unreachable`

**发生时机**：模型加载和 CUDA graph 捕获均已成功完成（`Capturing batches: 100%`），进入 MMLU benchmark 评测阶段，在加载 dataset 时崩溃。

**关键日志**：
```
httpcore.ConnectError: [Errno 101] Network is unreachable
```

**调用栈分析**：

```
lm_eval → simple_evaluate → task_manager.load → _factory.build
→ ConfigurableTask.__init__ → download → datasets.load_dataset
→ load_dataset_builder → _create_builder_config → _resolve_data_files
→ HfFileSystem.glob → _ls_tree → list_repo_tree
→ httpx GET → ConnectError
```

`datasets` 库在构建每个 MMLU 子任务（sub-config）时，会调用 `HfFileSystem.glob()` 来解析数据文件路径。这个方法内部会通过 `huggingface_hub` 的 API 列出远程仓库的目录结构（`list_repo_tree`），**即使数据集已经缓存在本地**也需要网络来验证/刷新元数据。

日志中可以看到 33+ 个 MMLU 子任务通过 `"Using the latest cached version of the dataset since cais/mmlu couldn't be found on the Hugging Face Hub"` 成功降级到本地缓存。但在构建下一个子任务（`high_school_geography` 之后的某个子任务）时，`_resolve_data_files` → `HfFileSystem.glob()` 的网络调用彻底失败，无法降级到本地缓存，导致崩溃。

**注意**：这**不是 SIMO 量化的问题**，也与模型加载无关。模型已正常加载完成，CUDA graph 已成功捕获（36/36）。这是运行环境的网络限制导致的 `datasets` 库功能性问题。

### 修复方法

在 lm-eval 命令前添加 `HF_DATASETS_OFFLINE=1` 环境变量，强制 `datasets` 和 `huggingface_hub` 使用离线模式：

```bash
export SIMO_SGLANG_REGISTER=1
export HF_DATASETS_CACHE=/share_data/users/like/huggingface_cache
export HF_DATASETS_OFFLINE=1   # ← 添加这一行

lm-eval --model sglang --model_args '{"pretrained": "/data_gpu/models/share_data/modelzoo/weights/llm/deepseek/DeepSeekV2/DeepSeek-V2-Lite-Chat-16B_A2.4B/safetensor_weights", "quantization": "simo", "json_model_override_args": "{\"quantization_config_file\": \"/data/like/package/simo_conda_sglang/simo/extensions/sglang_simo/example/online_quantization/../simo_quantization_config/online_quantization/quant_config_w4a4_mxfp.json\"}", "tp_size": 1, "dtype": "auto", "mem_fraction_static": 0.5}' --tasks mmlu --batch_size auto
```

`HF_DATASETS_OFFLINE=1` 会告诉 `huggingface_hub` 和 `datasets` 完全跳过网络请求，仅使用本地缓存的 `cais/mmlu` 数据集。由于日志中已经显示大部分 MMLU 子任务的数据都能从缓存加载，加上此变量后剩余的也能正常工作。

如果还需要更严格的离线约束，可以再加 `HF_HUB_OFFLINE=1`，但对 `datasets` 而言 `HF_DATASETS_OFFLINE=1` 已经足够。

---

## Q: 分析 vllm 源码构建失败原因

### 错误摘要

CMake 在 `FetchContent_MakeAvailable` 阶段失败，无法完成 vllm-flash-attn 外部依赖的 git 子模块初始化。

### 关键错误日志

```
fatal: git upload-pack: not our ref 574c1c121a0f3c0b44155b2b1987d89d16159b58
fatal: remote error: upload-pack: not our ref 574c1c121a0f3c0b44155b2b1987d89d16159b58
Errors during submodule fetch:
    csrc/composable_kernel
```

```
CMake Error at cmake/external_projects/vllm_flash_attn.cmake:58 (FetchContent_MakeAvailable):
  Build step for vllm-flash-attn failed: 1
```

### 调用链分析

构建入口 `like-useful/install-vllm.sh:4` 执行 `pip install -e .` 触发 CMake 编译。CMake 配置 `CMakeLists.txt:1254` 包含了 vllm-flash-attn 外部依赖，其定义在 `cmake/external_projects/vllm_flash_attn.cmake:40-47`：

```cmake
FetchContent_Declare(
    vllm-flash-attn
    GIT_REPOSITORY git@gitlabsoft.siorigin.com:xtubk/vllm-project-flash-attention.git
    GIT_TAG cmt-f5bc33cfc02c744d24a2e9d50e6db656de40611c
    GIT_PROGRESS TRUE
    BINARY_DIR ${CMAKE_BINARY_DIR}/vllm-flash-attn
)
```

`cmake/external_projects/vllm_flash_attn.cmake:58` 调用 `FetchContent_MakeAvailable(vllm-flash-attn)`，CMake 执行 git clone + git submodule update --init。

### 根因分析

vllm-flash-attention 仓库本身被成功 fetch（日志 line 376-377），但它含有一个 git 子模块 `csrc/composable_kernel`。子模块引用了一个 git commit `574c1c121a0f3c0b44155b2b1987d89d16159b58`，该 commit **在远程服务器上已不存在**（可能被 force push 覆盖或分支被删除了）。

日志 line 381-382：
```
fatal: git upload-pack: not our ref 574c1c121a0f3c0b44155b2b1987d89d16159b58
fatal: remote error: upload-pack: not our ref 574c1c121a0f3c0b44155b2b1987d89d16159b58
```

这是典型的 **git 子模块引用断裂**：父仓库的 `.gitmodules` 指向的子模块 commit hash 在子模块远程仓库中不再可用。

### 修复方案

有两种方式：

**方案一：使用本地源码绕过网络 clone（推荐）**

`cmake/external_projects/vllm_flash_attn.cmake:28-30` 支持通过环境变量 `VLLM_FLASH_ATTN_SRC_DIR` 指向本地已有副本：

```cmake
if (DEFINED ENV{VLLM_FLASH_ATTN_SRC_DIR})
  set(VLLM_FLASH_ATTN_SRC_DIR $ENV{VLLM_FLASH_ATTN_SRC_DIR})
endif()
```

操作步骤：
1. 手动克隆 vllm-flash-attention 仓库并修复子模块问题：
   ```bash
   VLLM_FA_DIR=/data/like/package/vllm-for-conda-simo/.deps/vllm-flash-attn-local
   git clone git@gitlabsoft.siorigin.com:xtubk/vllm-project-flash-attention.git $VLLM_FA_DIR
   cd $VLLM_FA_DIR
   git checkout cmt-f5bc33cfc02c744d24a2e9d50e6db656de40611c
   # 尝试修复子模块（如果主仓库已更新子模块引用）
   git submodule update --init --recursive
   ```

2. 如果可以自行处理子模块引用，用当前远程最新 commit 替换：
   ```bash
   cd $VLLM_FA_DIR/csrc/composable_kernel
   git fetch origin
   # 找到子模块应指向的新 commit，或使用 HEAD
   git checkout <valid-commit>
   cd $VLLM_FA_DIR
   git add csrc/composable_kernel
   ```

3. 设置环境变量后构建：
   ```bash
   export VLLM_FLASH_ATTN_SRC_DIR=$VLLM_FA_DIR
   pip install -e . --no-build-isolation ...
   ```

**方案二：联系 vllm-flash-attention 仓库维护者更新子模块引用**

子模块 commit `574c1c121a0f3c0b44155b2b1987d89d16159b58` 在 `csrc/composable_kernel` 远程仓库不存在了。需要仓库维护者：
1. 确认 `csrc/composable_kernel` 的正确 commit hash
2. 更新父仓库中的子模块引用（`git submodule update --remote`）
3. 推送到 `cmt-f5bc33cfc02c744d24a2e9d50e6db656de40611c` 分支（或创建新分支/tag）

### 补充说明

这**不是 vLLM 代码本身的问题**，也不是本地环境配置问题。这是一个外部依赖仓库的子模块引用失效问题。CMake 配置和编译环境本身都正常（PyTorch 12.8、nvcc 路径、CUDA arch 9.0 检测均正确）。

---

## Q: sglang 日志时间戳精度从秒提高到 us（微秒）

### 现状分析

当前 `custom_sglang.json` 的配置：

```json
"formatters": {
    "detailed": {
        "format": "[%(asctime)s] %(process)d %(filename)s:%(lineno)d %(funcName)s %(levelname)s %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S"
    }
}
```

日志输出如：`[2026-05-27 17:10:13]`，精度只到**秒**。

### 无法仅靠修改 JSON 达到微秒精度

Python 的 `logging.Formatter` 中 `datefmt` 参数会被传递给 `time.strftime()`，而 `strftime()` **不支持**表示毫秒或微秒的格式符（`%f` 仅在 `datetime.strftime()` 中可用，不在 `time.strftime()` 中）。相关代码在 Python 标准库 `logging/__init__.py` 的 `Formatter.formatTime()` 方法中。

sglang 本身通过 `SGLANG_LOG_MS` 环境变量在 format 中追加 `%(msecs)03d` 实现毫秒精度，参见 `python/sglang/srt/utils/common.py:1323` 的 `configure_logger()`：

```python
maybe_ms = ".%(msecs)03d" if envs.SGLANG_LOG_MS.get() else ""
format = f"[%(asctime)s{maybe_ms}{prefix}] %(message)s"
```

但 `SGLANG_LOG_MS` 只在**不使用** `SGLANG_LOGGING_CONFIG_PATH` 时生效（因为 `configure_logger()` 中若检测到 `SGLANG_LOGGING_CONFIG_PATH` 会直接 `return`，跳过后续所有内置格式逻辑，见 `python/sglang/srt/utils/common.py:1313-1322`）。

而 `%(msecs)03d` 本身就是毫秒，不是微秒。

### 方案一：JSON 修改 → 达到毫秒精度（最简单，改配置就行）

修改 `custom_sglang.json` 的 `format` 字段，在 `%(asctime)s` 后追加 `%(msecs)03d`：

```json
{
    "version": 1,
    "disable_existing_loggers": false,
    "formatters": {
        "detailed": {
            "format": "[%(asctime)s.%(msecs)03d] %(process)d %(filename)s:%(lineno)d %(funcName)s %(levelname)s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "detailed",
            "stream": "ext://sys.stdout"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"]
    }
}
```

输出变为：`[2026-05-27 17:10:13.123]`（毫秒精度）

只改了 `format` 字段，无需改任何代码。`%(msecs)03d` 是 `LogRecord` 的内置属性，dictConfig 可以直接使用。

### 方案二：自定义 Formatter 类 → 达到真微秒精度（需要加一段小代码）

Python `dictConfig` 支持 `()` 键指定 formatter 工厂。可以创建一个微秒级的 Formatter 并引用。

**Step 1**：创建 formatter 模块 `python/sglang/srt/utils/log_formatter_us.py`（放在 sglang 源码中，确保可 import）：

```python
import logging
import time


class USFormatter(logging.Formatter):
    """Formatter with microsecond precision timestamps."""

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = time.strftime(datefmt, ct)
        else:
            s = time.strftime(self.default_time_format, ct)
        us = int((record.created - int(record.created)) * 1_000_000)
        return f"{s}.{us:06d}"
```

**Step 2**：修改 `custom_sglang.json`，用 `()` 指定自定义 formatter：

```json
{
    "version": 1,
    "disable_existing_loggers": false,
    "formatters": {
        "detailed": {
            "()": "sglang.srt.utils.log_formatter_us.USFormatter",
            "format": "[%(asctime)s] %(process)d %(filename)s:%(lineno)d %(funcName)s %(levelname)s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "detailed",
            "stream": "ext://sys.stdout"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"]
    }
}
```

输出变为：`[2026-05-27 17:10:13.123456]`（微秒精度）

`dictConfig` 看到 `()` 键会调用该类的构造函数，传入其余参数（`format`, `datefmt`）。核心原理是覆写 `formatTime()` 方法，从 `record.created`（float，天然包含微秒精度）手动提取微秒部分并拼接到 `strftime` 的结果后。

### 总结对比

| 方案 | 精度 | 需要改代码？ | 需要改 JSON？ |
|------|------|-------------|--------------|
| 方案一 | 毫秒 (ms) | 否 | 是（改 `format` 字段） |
| 方案二 | 微秒 (us) | 是（新增 ~12 行 fomatter） | 是（改 `()` 键引用自定义类） |

---

## Q: `MLATokenToKVPool.get_value_buffer` 中的 `self.kv_buffer[layer_id - self.start_layer][..., : self.kv_lora_rank]` 发生 memory copy 吗？返回的是 view 还是 copy？

### 代码位置

`python/sglang/srt/mem_cache/memory_pool.py:1840-1856`，`MLATokenToKVPool.get_value_buffer()` 方法：

```python
def get_value_buffer(self, layer_id: int):
    if self.layer_transfer_counter is not None:
        self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

    if self.store_dtype != self.dtype:
        return self.kv_buffer[layer_id - self.start_layer][
            ..., : self.kv_lora_rank
        ].view(self.dtype)
    return self.kv_buffer[layer_id - self.start_layer][..., : self.kv_lora_rank]
```

### `kv_buffer` 的 tensor shape

`_create_buffers()`（`memory_pool.py:1754-1777`）中分配：

```python
self.kv_buffer = [
    torch.zeros(
        (self.size + self.page_size, 1, self.kv_cache_dim),
        dtype=self.store_dtype,
        device=self.device,
    )
    for _ in range(self.layer_num)
]
```

其中 `self.kv_cache_dim = kv_lora_rank + qk_rope_head_dim`（`memory_pool.py:1737-1741`）。每个 buffer 的 shape 是 `(size + page_size, 1, kv_lora_rank + qk_rope_head_dim)`。

### 结论：返回的是 **view**，不会发生 memory copy

核心表达式分为两步：

1. **`self.kv_buffer[layer_id - self.start_layer]`** — Python list 索引，取出对应的 `torch.Tensor`（shape `[size+page_size, 1, kv_lora_rank + qk_rope_head_dim]`）

2. **`[..., : self.kv_lora_rank]`** — PyTorch **basic slicing**（冒号切片），取最后一维的前 `kv_lora_rank` 个元素

PyTorch 中 **basic slicing 始终返回 view**（共享同一个 storage），只有 advanced indexing（用 tensor/list 做索引）才会触发 copy。这里用的是最基础的 `:N` 切片，没有 fancy indexing，所以是 view。

**store_dtype != dtype 的分支**（line 1852-1855）：

```python
return self.kv_buffer[layer_id - self.start_layer][
    ..., : self.kv_lora_rank
].view(self.dtype)
```

`.view()` 同样返回 view。由于 `[..., :N]` 对最后一维做切片，产生的 tensor 在最后一维是连续的，所以 `.view()` 可以直接 reshape 而不需要 `.contiguous()` 调用，**也不会触发 copy**。

### 验证逻辑

```python
# 假设 buffer shape = (100, 1, 192), kv_lora_rank = 128
buf = torch.zeros(100, 1, 192)
v = buf[..., :128]
print(v.data_ptr() == buf.data_ptr())  # True — 共享首地址，是 view
print(v.is_contiguous())               # True — 最后一维切片保持连续性
print(v.view(torch.float16).is_contiguous())  # True — .view() 也不会拷贝
```

两条 return 路径都不会触发 memory copy，返回的 tensor 与 `self.kv_buffer[layer_id]` 共享底层 storage。

---

## Q: Code Review — `concat_and_cache_mla_kernel`, `SIMOMLATokenToKVPool.set_mla_kv_buffer`, `set_kv_buffer`, `_temporarily_replace_simo_kv_pool_cls`

### 总览

共发现 **4 个 critical bug**（会导致 SyntaxError/NameError 直接崩溃）和 **3 个 medium bug**（缺少 import 导致运行时 NameError）。

---

### Critical Bug 1: `page_size: int` 语法错误

**文件**: `simo/extensions/sglang_simo/mem_cache/memory_pool.py:256-269`
**函数**: `SIMOMLATokenToKVPool.__init__`

```python
    super().__init__(
      size,
      page_size: int,    # BUG: type annotation 非法出现在函数调用实参位置
      dtype,
```

`page_size: int` 是 Python 类型标注语法，只能用于函数签名、变量标注等，**不能出现在函数调用的参数列表里**。这里意图显然是 `page_size=page_size`（关键字参数传递），属于写错成类型标注的 typo。会导致 `SyntaxError`，代码根本无法加载。

**修复**：改为 `page_size=page_size`。

---

### Critical Bug 2: 缺少逗号导致语法错误

**文件**: `simo/extensions/sglang_simo/mem_cache/memory_pool.py:363-366`
**函数**: `SIMOMLATokenToKVPool.set_mla_kv_buffer`

```python
    concat_and_cache_mla_kernel[grid](
      knope_ptr=cache_k_nope,
      kpe_ptr=cache_k_rope      # BUG: 缺少逗号
      kv_cache_ptr=kv_cache,
```

第 365 行 `kpe_ptr=cache_k_rope` 末尾缺少逗号，下一行 `kv_cache_ptr=kv_cache` 会被解析为独立语句而非函数参数。导致 `SyntaxError`。

**修复**：在 `kpe_ptr=cache_k_rope` 后加逗号。

---

### Critical Bug 3: `concat_dim` 未定义

**文件**: `simo/extensions/sglang_simo/mem_cache/memory_pool.py:356`
**函数**: `SIMOMLATokenToKVPool.set_mla_kv_buffer`

```python
    total_tiles = triton.cdiv(concat_dim, tile_size)
```

`concat_dim` 变量在 `set_mla_kv_buffer` 中从未被定义过。对比 vllm_simo 参照代码 `simo/extensions/vllm_simo/v1/attention/ops/triton_concat_and_cache_mla.py:236`：

```python
    concat_dim = kv_lora_rank + pe_dim          # ← vllm_simo 在调用前定义
    total_tiles = triton.cdiv(concat_dim, tile_size)
```

sglang_simo 版漏掉了这行定义，会导致运行时 `NameError`。

**修复**：在 line 356 前添加 `concat_dim = kv_lora_rank + pe_dim`。

---

### Critical Bug 4: `pe_dim` 使用错误维度索引

**文件**: `simo/extensions/sglang_simo/mem_cache/memory_pool.py:325`
**函数**: `SIMOMLATokenToKVPool.set_mla_kv_buffer`

```python
    kv_lora_rank = cache_k_nope.shape[-1]   # ← 取最后一维，正确
    pe_dim = cache_k_rope.shape[1]           # ← BUG: 应取最后一维
```

对比同一行获取 `kv_lora_rank` 用 `shape[-1]`，获取 `pe_dim` 却用了 `shape[1]`，不一致。

`set_kv_buffer`（line 308）传入的 `cache_k` 形状为 `[num_tokens, num_heads, kv_lora_rank + qk_rope_head_dim]`，拆分后 `cache_k_rope = cache_k[..., self.kv_lora_rank:]` 形状为 `[num_tokens, num_heads, qk_rope_head_dim]`。当 `num_heads > 1` 时 `shape[1]` 取到的是 `num_heads` 而非 `pe_dim`，导致 `pe_dim` 值错误，kernel 的 tile 划分和 load mask 全部出错，写出的 kv cache 数据损坏。

**注意**：对于 DeepSeek-V2 等 MLA 模型，`num_heads=1`（kv 被压缩到 1 个 head），这时 `shape[1]` 恰好等于 `pe_dim`，所以不会触发。但如果 `num_heads > 1`（例如某些 MLA 变体），就会出问题。

**修复**：改为 `pe_dim = cache_k_rope.shape[-1]`。

---

### Medium Bug 5: 缺少 `import os`、`import json`

**文件**: `simo/extensions/sglang_simo/mem_cache/memory_pool.py:270-294`
**函数**: `SIMOMLATokenToKVPool._create_buffers`

```python
  def _create_buffers(self):
    _in_capture = torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing()
    if not _in_capture:
        env_file = os.getenv("debug_env_file", "/dev/ipc.json")     # BUG: os 未 import
        env_data = json.load(open(env_file)) if os.path.exists(...)  # BUG: json 未 import
```

模块顶层 imports（`memory_pool.py:11-29`）中既无 `import os` 也无 `import json`。该方法在 `__init__` → `_create_buffers` 调用链上总是以 non-capture 状态执行，因此 `os.getenv` 和 `json.load` 会被实际执行，触发 `NameError`。

sglang 原始 `memory_pool.py` 模块顶层有 `import os`、`import json`，但 SIMO 版本遗漏了。

**修复**：在模块顶层添加 `import os`、`import json`。

---

### Medium Bug 6: 缺少 `import triton`

**文件**: `simo/extensions/sglang_simo/mem_cache/memory_pool.py:356, 363`
**函数**: `SIMOMLATokenToKVPool.set_mla_kv_buffer`

```python
    total_tiles = triton.cdiv(concat_dim, tile_size)   # BUG: triton 未 import
    ...
    concat_and_cache_mla_kernel[grid](...)              # BUG: triton 未 import
```

`triton` 模块没有在 `memory_pool.py` 中 import。尽管 `concat_and_cache_mla_kernel` 本身已在 line 29 import 进来看似可用，但 `[grid]` 调用方式需要 Triton 的 autotuner/JITFunction 机制，且 `triton.cdiv` 直接引用了 `triton` 命名空间。运行时触发 `NameError`。

**修复**：在模块顶层添加 `import triton`，或在函数内局部 import。

---

### Medium Bug 7: 缺少 `import RadixAttention`

**文件**: `simo/extensions/sglang_simo/mem_cache/memory_pool.py:279, 283`
**函数**: `SIMOMLATokenToKVPool.set_kv_buffer`、`set_mla_kv_buffer`

```python
    def set_kv_buffer(self, layer: RadixAttention, ...):           # BUG: RadixAttention 未 import
    def set_mla_kv_buffer(self, layer: RadixAttention, ...):       # BUG: RadixAttention 未 import
```

函数参数的类型标注使用了 `RadixAttention`，但该类型没有在模块中 import。在 Python 3.12 中，类型标注默认在函数定义时求值，会导致 `NameError`（除非模块有 `from __future__ import annotations`，此时标注转为字符串，不立即求值）。

sglang 原始 `memory_pool.py` 第 219 行有 `from sglang.srt.layers.radix_attention import RadixAttention`，SIMO 版本遗漏。

**修复**：在模块顶层添加 `from sglang.srt.layers.radix_attention import RadixAttention`，或添加 `from __future__ import annotations`。

---

### 对照表

| Bug | 严重程度 | 文件 (相对 simo/) | 行号 | 函数 |
|-----|---------|-------------------|------|------|
| `page_size: int` | Critical (SyntaxError) | `extensions/sglang_simo/mem_cache/memory_pool.py` | 258 | `SIMOMLATokenToKVPool.__init__` |
| 缺逗号 | Critical (SyntaxError) | `extensions/sglang_simo/mem_cache/memory_pool.py` | 365 | `SIMOMLATokenToKVPool.set_mla_kv_buffer` |
| `concat_dim` 未定义 | Critical (NameError) | `extensions/sglang_simo/mem_cache/memory_pool.py` | 356 | `SIMOMLATokenToKVPool.set_mla_kv_buffer` |
| `pe_dim = shape[1]` | Critical (错误值, 数据损坏) | `extensions/sglang_simo/mem_cache/memory_pool.py` | 325 | `SIMOMLATokenToKVPool.set_mla_kv_buffer` |
| 缺 `import os, json` | Medium (NameError) | `extensions/sglang_simo/mem_cache/memory_pool.py` | 273-276 | `SIMOMLATokenToKVPool._create_buffers` |
| 缺 `import triton` | Medium (NameError) | `extensions/sglang_simo/mem_cache/memory_pool.py` | 356, 363 | `SIMOMLATokenToKVPool.set_mla_kv_buffer` |
| 缺 `import RadixAttention` | Medium (NameError) | `extensions/sglang_simo/mem_cache/memory_pool.py` | 301, 312 | `set_kv_buffer`, `set_mla_kv_buffer` |

**附加说明**：`triton_backend.py` 中的 `concat_and_cache_mla_kernel` 本身没有发现 bug。KV cache 的 1D 线性地址计算（`slot_base = slot_idx * stride_cache`）和 nope/rope 分别从两个指针加载的逻辑是正确的。`_temporarily_replace_simo_kv_pool_cls` 的 class 替换/恢复逻辑（try/finally restore）也是正确的，无并发安全问题。

---

## Q: 修复后复查

### 已修复项（7/7 确认）

| # | 原 bug（memory_pool.py） | 状态 | 备注 |
|---|-------------------------|------|------|
| 1 | `page_size: int` (line 258) | 修复 | 改为 positional `page_size` (line 262) |
| 2 | 缺逗号 `kpe_ptr=cache_k_rope` (line 365) | 修复 | line 372 已加逗号 |
| 3 | `concat_dim` 未定义 (line 356) | 修复 | line 362: `concat_dim = kv_lora_rank + pe_dim` |
| 4 | `pe_dim = shape[1]` (line 325) | 修复 | line 329: `cache_k_rope.shape[-1]` |
| 5 | 缺 `import os, json` | 修复 | line 33-35 |
| 6 | 缺 `import triton` | 修复 | line 27 |
| 7 | 缺 `import RadixAttention` | 修复 | 移除了 `RadixAttention` 类型标注 |

---

### 新引入的 Critical Bug 1: `assert` 用 `=` 赋值而非 `==` 比较

**文件**: `simo/extensions/sglang_simo/mem_cache/memory_pool.py:330-331`
**函数**: `SIMOMLATokenToKVPool.set_mla_kv_buffer`

```python
    assert self.kv_lora_rank = kv_lora_rank       # BUG: = → SyntaxError
    assert self.qk_rope_head_dim = pe_dim          # BUG: = → SyntaxError
```

`assert` 语句中必须使用 `==`（相等比较），不能使用 `=`（赋值）。`assert x = y` 在 Python 中是 `SyntaxError`。

意图显然是对 kv_lora_rank 和 pe_dim 做一致性校验。**修复**：改为 `==`。

---

### 新引入的 Critical Bug 2: `kv_cache` 变量在定义前被引用

**文件**: `simo/extensions/sglang_simo/mem_cache/memory_pool.py:364-366`
**函数**: `SIMOMLATokenToKVPool.set_mla_kv_buffer`

```python
    concat_dim = kv_lora_rank + pe_dim                    # line 362
    total_tiles = triton.cdiv(concat_dim, tile_size)      # line 363
    assert kv_cache.dtype == torch.uint8, ...             # line 364  ← BUG! kv_cache 未定义
    grid = (num_tokens, total_tiles)                      # line 365
    kv_cache = self.kv_buffer[layer.layer_id - self.start_layer]  # line 366  ← 定义在这之后
```

`kv_cache` 在 line 364 被引用，但在 line 366 才被赋值定义。会导致运行时 `NameError`。

对比 vllm_simo 版 `triton_concat_and_cache_mla.py:239`，那里 `kv_cache` 是函数参数，所以在 assert 时已存在。sglang_simo 版需要在 assert 之前先定义 `kv_cache`。

**修复**：将 line 366 移到 line 364 之前：

```python
    kv_cache = self.kv_buffer[layer.layer_id - self.start_layer]
    assert kv_cache.dtype == torch.uint8, "Quantized KV cache must be uint8"
    concat_dim = kv_lora_rank + pe_dim
    total_tiles = triton.cdiv(concat_dim, tile_size)
    grid = (num_tokens, total_tiles)
```

---

### 其他检查（无问题）

- `concat_and_cache_mla_kernel`（triton kernel）未修改，正确
- `_temporarily_replace_simo_kv_pool_cls` 未修改，正确
- `set_kv_buffer` 未修改，正确
- `_create_buffers` 中 `os`/`json` 的 `import` 位置（line 33-35）在 `# like_debug {` 注释块内，运行时能正常工作
- `import triton`（line 27）已添加到模块顶层

---

## Q: 第二次修复后复查

### 上次待修复项（2/2 确认）

| # | 上次 bug（memory_pool.py） | 状态 | 备注 |
|---|---------------------------|------|------|
| 1 | `assert self.kv_lora_rank = kv_lora_rank` (line 330-331) | 修复 | 改为 `==` |
| 2 | `kv_cache` 在定义前被引用 | 修复 | line 364 定义，line 365 assert，顺序正确 |

### 完整终审 — 全部 4 个审查目标

**`simo/extensions/sglang_simo/layers/attention/triton_ops/set_kv_buffer.py:306-460` — `concat_and_cache_mla_kernel`**

无 bug。Tile 分配逻辑正确：
- `is_kv_c = (tile_idx * TILE_SIZE) < kv_lora_rank` — 按 tile 起始位置判断归属区域
- nope/rope 分别从 `knope_ptr` / `kpe_ptr` 加载，mask 正确处理 tail tile
- 写入 offset 布局正确：
  ```
  [KV_C packed | KV_C scales | K_PE packed | K_PE scales]
  ```
  与 vllm_simo 版一致，与 `__init__` 中 `kv_cache_dim_in_bytes` 的计算对齐

**`simo/extensions/sglang_simo/mem_cache/memory_pool.py:232-298` — `SIMOMLATokenToKVPool.__init__` + `_create_buffers`**

无 bug。初始化顺序正确：
1. `self.kv_cache_dim_in_bytes` 先计算（依赖 `kv_cache_downcast_kernel`）
2. 调用 `super().__init__()` → `MLATokenToKVPool.__init__()` → `self._create_buffers()`（已被子类 override）
3. 子类 `_create_buffers` 使用步骤 1 已赋值的 `self.kv_cache_dim_in_bytes`

buffer shape 为 `(size + page_size, 1, kv_cache_dim_in_bytes)`，与 1D 线性寻址 kernel 的 `slot_base = slot_idx * stride_cache` 一致。

**`simo/extensions/sglang_simo/mem_cache/memory_pool.py:303-395` — `set_kv_buffer` + `set_mla_kv_buffer`**

无 bug。
- `set_kv_buffer`：basic slicing `cache_k[..., :self.kv_lora_rank]` 拆分 nope/rope，返回 view，无 copy
- `set_mla_kv_buffer`：format params → tile bytes → grid → kernel call，逻辑正确
- `kv_cache_dim_in_bytes`（`__init__` 用 128 批量计算）与 `set_mla_kv_buffer` 用单 tile 计算再合并的每 token bytes 一致

**`simo/extensions/sglang_simo/mem_cache/init_memory_pool_patch.py:34-178` — `_temporarily_replace_simo_kv_pool_cls` + `_patched_init_memory_pool`**

无 bug。class 替换/恢复在 try/finally 中保证原子性，并发安全。SWA 路径通过 `is_hybrid_swa` 正确跳过。

### 结论

**无遗漏，无新 bug。** 所有 4 个目标代码段已通过 review。

---

## Q: Code Review — commit `8c9611c29` (decode_attention support rope part)

### 变更概览

| 文件 | 变更 |
|------|------|
| `simo/extensions/sglang_simo/models/deepseek_v2.py` | 新建 — 替换 `DeepseekV2AttentionMLA`，给 `attn_mqa/attn_mha` 设置 `kv_lora_rank`/`qk_rope_head_dim` |
| `simo/extensions/sglang_simo/layers/attention/triton_ops/decode_attention.py` | `_fwd_grouped_kernel_stage1` 新增 PE 部分量化 K 的加载与 attention；`_decode_grouped_att_m_fwd` 传入 rope 参数 |
| `simo/extensions/sglang_simo/quantization/quantization.py` | `create_weights` 区分 MHA/MLA 分别计算 `packed_head_size`/`scale_head_size` |
| `simo/extensions/sglang_simo/__init__.py` | 新增 import deepseek_v2 |

---

### Critical Bug 1: `super().__init__()` 未传参 — 父类未初始化

**文件**: `simo/extensions/sglang_simo/models/deepseek_v2.py:17-30`
**类**: `SIMODeepseekV2AttentionMLA.__init__`

```python
    def __init__(
        self,
        config,
        hidden_size: int,
        ...
        mla_enable_prefill_cp: bool = False,
    ) -> None:
        super().__init__()          # BUG: 没有传任何参数给父类
        # set kv_lora_rank and qk_rope_head_dim
        self.attn_mqa.kv_lora_rank = kv_lora_rank   # BUG: self.attn_mqa 未定义!
```

父类 `DeepseekV2AttentionMLA.__init__` 的签名 (`sglang_kernel_src/python/sglang/srt/models/deepseek_v2.py:1379-1401`)：

```python
    def __init__(
        self,
        config: PretrainedConfig,   # ← 没有默认值, 必须传!
        hidden_size: int,           # ← 没有默认值, 必须传!
        num_heads: int,
        ...
```

`super().__init__()` 不带任何参数调用 `DeepseekV2AttentionMLA.__init__`，而该函数前两个参数 `config` 和 `hidden_size` 是**必传**的（无默认值），会直接抛出 `TypeError`。

此外，即使修复 `super().__init__()` 传参，`self.attn_mqa` 和 `self.attn_mha` 是由父类 `__init__` 创建的，`super().__init__()` 不传参意味着这两个属性根本不会被创建，后续 setattr 也会报 `AttributeError`。

**修复**：将 SIMO `__init__` 收到的所有参数转发给父类：

```python
super().__init__(
    config=config, hidden_size=hidden_size, num_heads=num_heads,
    qk_nope_head_dim=qk_nope_head_dim, qk_rope_head_dim=qk_rope_head_dim,
    v_head_dim=v_head_dim, q_lora_rank=q_lora_rank, kv_lora_rank=kv_lora_rank,
    rope_theta=rope_theta, rope_scaling=rope_scaling,
    max_position_embeddings=max_position_embeddings, quant_config=quant_config,
    reduce_results=reduce_results, layer_id=layer_id, prefix=prefix,
    alt_stream=alt_stream, skip_rope=skip_rope, is_nextn=is_nextn,
    dsa_enable_prefill_cp=dsa_enable_prefill_cp,
    mla_enable_prefill_cp=mla_enable_prefill_cp,
)
```

---

### Critical Bug 2: PE attention 的 dot 操作使用 `q` 而非 `qpe` — 维度不匹配

**文件**: `simo/extensions/sglang_simo/layers/attention/triton_ops/decode_attention.py:138-155`
**函数**: `_fwd_grouped_kernel_stage1`

`q` 的加载（line 138, 154）：

```python
offs_d = tl.arange(0, BLOCK_DMODEL)               # line 115
offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
q = tl.load(Q + offs_q, ...)                      # q shape: [BLOCK_H, BLOCK_DMODEL]
```

对 MLA 模型：`BLOCK_DMODEL = Lk = kv_lora_rank + qk_rope_head_dim`。`q` 包含 nope **和** PE 两部分的 query，共 `kv_lora_rank + pe_dim` 个元素。

但在所有三个 PE attention 计算路径中，**全部使用了 `q` 而不是 `qpe`**：

| 路径 (line) | 代码 | 问题 |
|------------|------|------|
| Per-group (296) | `tl.dot(q, K_dequant)` | PE key 只有 `pe_dim` 行，q 有 `kv_lora_rank + pe_dim` 行 → 维度不匹配 |
| MX SW dequant (335) | `tl.dot(q, K_dequant)` | 同上 |
| MXFP8 (347) | `tl.dot_scaled(q, ..., k, ...)` | `k` 用 `offs_dpe` 加载，只有 `pe_dim` 行，q 维度不匹配 |

**同时**，`qpe` 虽然在 line 155-156 被加载了：

```python
if BLOCK_DPE > 0:
    qpe = tl.load(Q + off_qpe, mask=(...), other=0.0)
```

但它 `offs_dpe` 的计算（line 141）也有历史 bug：

```python
offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)   # line 141
mask_dpe = offs_dpe < Lk                             # BLOCK_DMODEL == Lk → mask 永远为 False
```

`offs_dpe` 起始于 `BLOCK_DMODEL`（等于 `Lk`），mask `offs_dpe < Lk` 永远为 False，`qpe` 永远是全零 tensor。正确的 offset 应该是 `kv_lora_rank + tl.arange(0, BLOCK_DPE)`。

**修复**（两个问题一起修）：

```python
# 修正 offs_dpe (line 141)
offs_dpe = kv_lora_rank + tl.arange(0, BLOCK_DPE)   # 原: BLOCK_DMODEL + ...

# PE attention 统一改用 qpe（替换所有 3 处 tl.dot(q, ...) 为 tl.dot(qpe, ...)）
qk_pe = tl.dot(qpe, K_dequant)                                        # PG path
qk_pe = tl.dot(qpe, K_dequant)                                        # MX SW path
qk_pe = tl.dot_scaled(qpe, None, "bf16", k, k_scale, "e4m3")        # MXFP8 path
```

**说明**：MHA 模型 `BLOCK_DPE = 0`，PE 分支完全跳过，使用 `q` 或 `qpe` 无影响。但 MLA 模型 `BLOCK_DPE > 0`，此 bug 会导致 Triton JIT 编译失败（维度不匹配）。

---

### NIT 1: `_decode_grouped_att_m_fwd` 传参 — MLA 模式下不应传 `-1`

**文件**: `simo/extensions/sglang_simo/layers/attention/triton_ops/decode_attention.py:558-565`
**函数**: `_decode_grouped_att_m_fwd`

```python
    packed_head_size_rope = getattr(layer, "packed_head_size_rope", -1)
    scale_head_size_rope = getattr(layer, "scale_head_size_rope", -1)
```

非 MLA 模型（MHA）中 `BLOCK_DPE = 0`，kernel 的 PE 分支被完全跳过，`-1` 不会被执行到，所以不造成功能 bug。但 `PACKED_HEAD_SIZE_ROPE = -1` 作为 `tl.constexpr` 传入 kernel 是 dirty 的默认值。

---

### NIT 2: 注释掉的旧代码和 FP6 mask 冗余问题

- `decode_attention.py:352-360` — 旧 K-PE 加载代码被注释而非删除，保留为死代码。
- `decode_attention.py:287` — FP6 mask 中 `(group_offs[None, :] < FP6_NUM_GROUPS)` 永远 True（开发者自己在注释里问了），冗余但无害。

---

### 验证点：`_decode_grouped_att_m_fwd` 参数计算

调用路径：`SIMOKVCacheMethod.create_weights` → `layer.packed_head_size_rope` / `layer.scale_head_size_rope` → `_decode_grouped_att_m_fwd`

- `packed_head_size_rope` 和 `scale_head_size_rope` 在 `simo/extensions/sglang_simo/quantization/quantization.py:1399-1402` 通过 `getattr(layer, "kv_lora_rank")` 区分 MHA/MLA 后分别计算，逻辑正确。
- `PACKED_HEAD_SIZE_PADDED_ROPE`（line 564）对 `> 0` 才做 `next_power_of_2`，否则为 `-1`，防御性正确。
- `rope_offset_in_token = num_kv_heads * (PACKED_HEAD_SIZE + SCALE_HEAD_SIZE)` 计算从 buffer 中跳过 nope 部分的偏移量。buffer 布局为 `[nope_packed | nope_scales | pe_packed | pe_scales]`，其中 nope 部分每个 head 占 `PACKED_HEAD_SIZE + SCALE_HEAD_SIZE` bytes，`num_kv_heads * (...)` 正确跳过了所有 head 的 nope 数据。

### 总结

| # | Bug | 严重程度 | 位置 |
|---|-----|---------|------|
| 1 | `super().__init__()` 未传参 → TypeError | Critical | `deepseek_v2.py:29` |
| 2 | PE dot 用 `q` 而非 `qpe` → 维度不匹配 | Critical | `decode_attention.py:296, 335, 347` |
| - | `qpe` offs_dpe 计算错误 → qpe 全零 | 历史 bug | `decode_attention.py:141` |
| 3 | 非 MLA 传 `-1` | Nit | `decode_attention.py:563-565` |
| 4 | 旧代码未删除 + FP6 mask 冗余 | Nit | `decode_attention.py:287, 352` |



---

## Code Review: Commit 8c9611c29 — MLA KV Cache read/write matching & parameter computation

### Review Scope

1. `_fwd_grouped_kernel_stage1` PE reading correctness
2. Read/write layout consistency with `concat_and_cache_mla_kernel`
3. `_decode_grouped_att_m_fwd` parameter computation
4. `SIMODeepseekV2AttentionMLA` class

---

### 1. Read/Write Layout Matching: VERIFIED CORRECT

**Write layout** (`concat_and_cache_mla_kernel` in `set_kv_buffer.py`):

Per token slot (1D linear buffer), the 4 regions are:

```
[0 .. KV_C_PACKED_BYTES-1]                              : nope packed data
[KV_C_PACKED_BYTES .. KV_C_TOTAL_BYTES-1]               : nope scale data
[KV_C_TOTAL_BYTES .. KV_C_TOTAL_BYTES+K_PE_PACKED_BYTES-1] : pe packed data
[KV_C_TOTAL_BYTES+K_PE_PACKED_BYTES .. END]             : pe scale data
```

Where:
- `KV_C_TOTAL_BYTES = KV_C_PACKED_BYTES + KV_C_SCALE_BYTES`
- `kv_c_num_tiles * packed_tile_bytes = KV_C_PACKED_BYTES`
- `k_pe_num_tiles * packed_tile_bytes = K_PE_PACKED_BYTES`

PE packed write uses `local_tile_idx = tile_idx - kv_c_num_tiles`:
```python
packed_write_offset = KV_C_TOTAL_BYTES + local_tile_idx * PACKED_TILE_BYTES   # set_kv_buffer.py:372
scale_write_offset  = KV_C_TOTAL_BYTES + K_PE_PACKED_BYTES + local_tile_idx * SCALE_TILE_BYTES  # :373
```

**Read layout** (`_fwd_grouped_kernel_stage1` in `decode_attention.py`):

PE start offset per token:
```python
rope_offset_in_token = num_kv_heads * (PACKED_HEAD_SIZE + SCALE_HEAD_SIZE)    # :263
```

For MLA, `num_kv_heads = 1`, so:
- `rope_offset_in_token = PACKED_HEAD_SIZE + SCALE_HEAD_SIZE = KV_C_TOTAL_BYTES` ✓

PE packed data offset (PG/SW paths):
```python
rope_offset_in_token + cur_kv_head * PACKED_HEAD_SIZE_ROPE + packed_offs_d[...]
```
= `KV_C_TOTAL_BYTES + 0 + [0, 1, ...]` ✓

PE scale offset:
```python
rope_offset_in_token + SCALE_PLANE_OFFSET_ROPE + cur_kv_head * SCALE_HEAD_SIZE_ROPE + ...
```
= `KV_C_TOTAL_BYTES + K_PE_PACKED_BYTES + 0 + ...` ✓

**The read layout matches the write layout for PG and SW dequant paths.**

---

### 2. `_decode_grouped_att_m_fwd` Parameter Computation: CORRECT

```python
packed_head_size_rope = getattr(layer, "packed_head_size_rope", -1)    # :550
scale_head_size_rope  = getattr(layer, "scale_head_size_rope", -1)     # :551

PACKED_HEAD_SIZE_PADDED_ROPE = triton.next_power_of_2(packed_head_size_rope) if packed_head_size_rope > 0 else -1   # :564
BLOCK_DMODEL_SCALE_ROPE      = triton.next_power_of_2(scale_head_size_rope)  if scale_head_size_rope > 0 else -1    # :565
```

These values come from `SIMOKVCacheMethod.create_weights` (`quantization.py:1401-1407`), which correctly computes packing sizes for the rope part via meta tensors.

Fallback `-1` is safe: when `packed_head_size_rope == -1` (non-MLA), `BLOCK_DPE` will be 0 (since Lk < 288 won't trigger the pe_dim split), so the MLA PE code block is never entered.

**Parameter computation is correct.**

---

### 3. BUGS FOUND

#### BUG 1 (CRITICAL): `super().__init__()` without arguments

**File:** `simo/extensions/sglang_simo/models/deepseek_v2.py:29`

```python
def __init__(self, config, hidden_size, ..., mla_enable_prefill_cp=False) -> None:
    super().__init__()   # ← BUG: parent __init__() called without any arguments
```

The parent class `deepseek_v2.DeepseekV2AttentionMLA.__init__()` requires ALL the constructor parameters (`config`, `hidden_size`, `num_heads`, `qk_nope_head_dim`, ...) to properly initialize `self.attn_mqa`, `self.attn_mha`, and all the attention projection layers.

Calling `super().__init__()` with no arguments will raise a TypeError at runtime.

Lines 32-35 then attempt to set attributes on `self.attn_mqa` and `self.attn_mha`, which don't exist since the parent was never properly initialized.

**Fix:**
```python
super().__init__(
    config, hidden_size, num_heads, qk_nope_head_dim, qk_rope_head_dim,
    v_head_dim, q_lora_rank, kv_lora_rank, rope_theta, rope_scaling,
    max_position_embeddings, quant_config, reduce_results, layer_id,
    prefix, alt_stream, skip_rope, is_nextn, dsa_enable_prefill_cp,
    mla_enable_prefill_cp,
)
```

---

#### BUG 2 (CRITICAL): PE dot products use `q` instead of `qpe`

**File:** `simo/extensions/sglang_simo/layers/attention/triton_ops/decode_attention.py`

All 5 PE dot product paths use the **nope query vector `q`** instead of the **rope query vector `qpe`**. This is a dimension mismatch that will cause incorrect results or kernel failure.

For DeepSeek-V2 MLA: `q` has shape `[BLOCK_H, kv_lora_rank]` = `[16, 512]`, while the PE key has shape `[BLOCK_N, pe_dim]` = `[32, 64]`. The dot product `q @ K_pe.T` has shape `[16, 512] @ [32, 64]` — dimension mismatch (512 ≠ 64).

`qpe` has shape `[BLOCK_H, BLOCK_DPE]` = `[16, 64]`, so `qpe @ K_pe.T` = `[16, 64] @ [32, 64]` — correct.

**Affected lines (all use `q` instead of `qpe`):**

| Line | Path | Current (wrong) | Correct |
|------|------|-----------------|---------|
| 292 | Per-group | `tl.dot(q, K_dequant)` | `tl.dot(qpe, K_dequant)` |
| 331 | MXFP4 SW | `tl.dot_scaled(q, None, "bf16", ...)` | `tl.dot_scaled(qpe, None, "bf16", ...)` |
| 334 | SW dequant | `tl.dot(q, K_dequant)` | `tl.dot(qpe, K_dequant)` |
| 359 | MXFP8 E4M3 | `tl.dot_scaled(q, None, "bf16", ...)` | `tl.dot_scaled(qpe, None, "bf16", ...)` |
| 361 | MXFP8 E5M2 | `tl.dot_scaled(q, None, "bf16", ...)` | `tl.dot_scaled(qpe, None, "bf16", ...)` |

---

#### BUG 3 (CRITICAL): Transposed MXFP8 PE loading uses wrong offset

**File:** `decode_attention.py:337-338`

```python
# Transposed K loading for MXFP8 PE
offs_buf_k = (
    kv_loc[None, :] * stride_buf_kbs
    + rope_offset_in_token
    + cur_kv_head * PACKED_HEAD_SIZE_ROPE
    + offs_dpe[:, None]          # ← BUG: offs_dpe starts at BLOCK_DMODEL=512
)
```

`offs_dpe` is defined on line 141 as `BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)`, which for DeepSeek-V2 is `[512, 513, ..., 575]`.

The PE packed data for one token starts at `rope_offset_in_token` and has only `PACKED_HEAD_SIZE_ROPE` bytes (e.g., 64 bytes for MXFP8_E4M3, pe_dim=64). Using offsets `[512, ..., 575]` will read far beyond the PE data segment — out of the token's buffer region entirely.

This affects only the transposed MXFP8 path (fast path with `tl.dot_scaled`).

**Fix:** Define a separate offset variable that starts at 0:
```python
offs_d_pe_local = tl.arange(0, BLOCK_DPE)  # or tl.arange(0, PACKED_HEAD_SIZE_PADDED_ROPE)
```
And use it in the transposed PE loading path instead of `offs_dpe`.

Note: the PG and SW dequant paths (lines 266-334) correctly start their offsets at 0 using `tl.arange(0, PACKED_HEAD_SIZE_PADDED_ROPE)`. Only the transposed MXFP8 path has this bug.

---

### 4. ADDITIONAL OBSERVATIONS

#### `qpe` loading offset — actually NOT a historical bug

The earlier analysis suggested `qpe` loading at line 141 was buggy because `offs_dpe < Lk` would always be False. This is incorrect:

```python
offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)  # = [512, 513, ..., 575]
mask_dpe = offs_dpe < Lk                              # Lk = kv_lora_rank + pe_dim = 576
```

For DeepSeek-V2: `[512..575] < 576` is ALL True. So `qpe` loads correctly from the original (non-quantized) Q tensor. The `qpe` vector is correct.

However, note that `Q` here refers to the original **bf16 Q tensor** (not the quantized K buffer). The Q tensor has `qk_head_dim = kv_lora_rank + pe_dim = 576` elements, and the PE part starts at offset `kv_lora_rank = 512`. So `offs_dpe = [512..575]` correctly addresses the PE portion of Q. **This is correct.**

#### `qk_pe` accumulation uses `+=` on `qk`

Line 362: `qk += qk_pe` — once BUG 2 is fixed (using `qpe` for the dot product), this line correctly accumulates the PE attention score onto the nope score. The resulting `qk` has the right shape `[BLOCK_H, BLOCK_N]`.

---
