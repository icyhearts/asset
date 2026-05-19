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
