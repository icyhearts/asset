import torch
import torch.nn.functional as F
import math
import time


class FlashDecodingDemo:
    """Flash-Decoding注意力计算演示"""

    def __init__(self, d_model: int = 64, num_heads: int = 8):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

    def traditional_attention(self, q, k, v):
        """传统连续注意力计算"""
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights

    def flash_decoding_attention(self, q, k, v, block_size=32, tiling_mode='distributed'):
      if tiling_mode == 'distributed':
        return self.flash_decoding_distributed_tiling(q, k, v,
                                  tile_size_kv=block_size)
      else:
        return self.flash_decoding_without_cp(q, k, v, block_size)

    def flash_decoding_without_cp(self, q, k, v, block_size=32):
        """
        分块的FA

        """
        batch_size, num_heads, seq_len_q, _ = q.shape
        seq_len_kv = k.shape[2]
        num_blocks = (seq_len_kv + block_size - 1) // block_size

        # 初始化累积变量
        # 累积的加权和
        numerator = torch.zeros(batch_size, num_heads, seq_len_q, self.head_dim,
                                device=q.device, dtype=q.dtype)
        # 累积的归一化因子
        d_prime = torch.zeros(batch_size, num_heads, seq_len_q, 1,
                                  device=q.device, dtype=q.dtype)

        # 用于数值稳定性的全局最大值（初始设为很小的数）
        global_max = torch.full((batch_size, num_heads, seq_len_q, 1),
                                -float('inf'),
                                device=q.device, dtype=q.dtype)

        # 分块处理
        for block_idx in range(num_blocks):
            start_idx = block_idx * block_size
            end_idx = min(start_idx + block_size, seq_len_kv)

            k_block = k[:, :, start_idx:end_idx, :]
            v_block = v[:, :, start_idx:end_idx, :]

            # 计算当前块的注意力分数
            scores_block = torch.matmul(q, k_block.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # 当前块的最大值
            block_max = scores_block.max(dim=-1, keepdim=True).values

            # 更新全局最大值
            # 我们需要比较每个位置（每个query）在所有块中的最大值
            new_global_max = torch.maximum(global_max, block_max)

            # 调整之前累积的权重（基于新的全局最大值）
            # 当全局最大值更新时，需要重新调整之前累积的权重
            if block_idx > 0:
                # 将之前累积的权重调整到新的尺度
                adjustment_factor = torch.exp(global_max - new_global_max)
                numerator = numerator * adjustment_factor
                d_prime = d_prime * adjustment_factor

            # 更新全局最大值
            global_max = new_global_max

            # 计算当前块的指数权重（减去全局最大值以保持数值稳定）
            exp_scores = torch.exp(scores_block - global_max)
            block_sum_exp = exp_scores.sum(dim=-1, keepdim=True)

            # 累积加权和
            numerator = numerator + torch.matmul(exp_scores, v_block)
            d_prime = d_prime + block_sum_exp

        # 最终归一化
        final_output = numerator / d_prime

        # 为了验证，也计算完整的注意力权重
        full_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        full_attention_weights = F.softmax(full_scores, dim=-1)

        return final_output, full_attention_weights

    def flash_decoding_distributed_tiling(self, q, k, v,
                          tile_size_kv: int = 256,
                          num_streams: int = 5):
        """
        使用数组明确模拟多个计算流（stream）的并行
        每个流有自己独立的累加器数组
        """
        batch_size, num_heads, seq_len_q, head_dim = q.shape
        seq_len_kv = k.shape[2]
        num_tiles = (seq_len_kv + tile_size_kv - 1) // tile_size_kv

        print(f"\n分布式数组实现: {num_streams}个计算流")
        print(f"每个流有自己的O、M、L数组")

        # 创建流数组：每个流有独立的(O, M, L)
        stream_O = []  # 加权和数组
        stream_M = []  # 最大值数组
        stream_L = []  # exp和数组

        for stream_id in range(num_streams):
            # 每个流初始化自己的累加器
            O_stream = torch.zeros_like(q)
            M_stream = torch.full((batch_size, num_heads, seq_len_q, 1),
                                -float('inf'), device=q.device, dtype=q.dtype)
            L_stream = torch.zeros_like(M_stream)

            stream_O.append(O_stream)
            stream_M.append(M_stream)
            stream_L.append(L_stream)

        # 模拟流并行处理tile
        print(f"并行处理{num_tiles}个tile...")

        for tile_idx in range(num_tiles):
            # 确定处理这个tile的流
            stream_id = tile_idx % num_streams

            # 获取当前tile
            start_idx = tile_idx * tile_size_kv
            end_idx = min(start_idx + tile_size_kv, seq_len_kv)

            k_tile = k[:, :, start_idx:end_idx, :]
            v_tile = v[:, :, start_idx:end_idx, :]

            # 当前流处理（只能访问自己的数组）
            O_curr = stream_O[stream_id]
            M_curr = stream_M[stream_id]
            L_curr = stream_L[stream_id]

            # 计算当前tile
            S_tile = torch.matmul(q, k_tile.transpose(-2, -1)) / math.sqrt(head_dim)
            m_tile = S_tile.max(dim=-1, keepdim=True).values

            # 更新当前流的统计量
            new_M = torch.maximum(M_curr, m_tile)

            if not torch.allclose(M_curr, new_M):
                scale = torch.exp(M_curr - new_M)
                O_curr = O_curr * scale
                L_curr = L_curr * scale

            exp_tile = torch.exp(S_tile - new_M)
            l_tile = exp_tile.sum(dim=-1, keepdim=True)

            # 更新当前流的数组
            stream_O[stream_id] = O_curr + torch.matmul(exp_tile, v_tile)
            stream_L[stream_id] = L_curr + l_tile
            stream_M[stream_id] = new_M


        # 归约所有流的结果
        final_output = self.reduce_stream_arrays(stream_O, stream_M, stream_L)

        # 为了验证，也计算完整的注意力权重
        full_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        full_attention_weights = F.softmax(full_scores, dim=-1)

        return final_output, full_attention_weights

    def reduce_stream_arrays(self, stream_O, stream_M, stream_L):
        """归约多个流的数组结果"""
        num_streams = len(stream_O)
        if num_streams == 0:
            return torch.zeros_like(stream_O[0])

        # 使用树形归约算法
        # 第一轮：相邻流两两归约
        current_O = stream_O.copy()
        current_M = stream_M.copy()
        current_L = stream_L.copy()

        remaining = num_streams
        step = 1

        while remaining > 1:
            next_O = []
            next_M = []
            next_L = []

            # 每两个流归约为一个
            for i in range(0, remaining, 2):
                if i + 1 < remaining:
                    # 归约流i和流i+1
                    O1, M1, L1 = current_O[i], current_M[i], current_L[i]
                    O2, M2, L2 = current_O[i+1], current_M[i+1], current_L[i+1]

                    # 合并
                    new_M = torch.maximum(M1, M2)

                    # 调整第一个流
                    if not torch.allclose(M1, new_M):
                        scale1 = torch.exp(M1 - new_M)
                        O1 = O1 * scale1
                        L1 = L1 * scale1

                    # 调整第二个流
                    scale2 = torch.exp(M2 - new_M)
                    O2 = O2 * scale2
                    L2 = L2 * scale2

                    # 合并结果
                    merged_O = O1 + O2
                    merged_L = L1 + L2

                    next_O.append(merged_O)
                    next_M.append(new_M)
                    next_L.append(merged_L)

                else:
                    # 奇数个流时，最后一个流直接进入下一轮
                    next_O.append(current_O[i])
                    next_M.append(current_M[i])
                    next_L.append(current_L[i])

            current_O = next_O
            current_M = next_M
            current_L = next_L
            remaining = len(current_O)
            step += 1

        # 最终归一化
        final_output = current_O[0] / current_L[0]
        print(f"归约完成，最终输出形状: {final_output.shape}")
        return final_output

    def flash_decoding_attention_simple(self, q, k, v, block_size=32):
        """
        简化版本Flash-Decoding实现，包含两个循环。
        需要保存每个块的max值、block_sum_exp值。
        特点：理解直观。
        """
        batch_size, num_heads, seq_len_q, _ = q.shape
        seq_len_kv = k.shape[2]
        num_blocks = (seq_len_kv + block_size - 1) // block_size

        # 存储每个块的中间结果
        block_outputs = []
        block_max_vals = []
        block_sum_exps = []

        # 第一步：计算每个块的局部结果
        for block_idx in range(num_blocks):
            start_idx = block_idx * block_size
            end_idx = min(start_idx + block_size, seq_len_kv)

            k_block = k[:, :, start_idx:end_idx, :]
            v_block = v[:, :, start_idx:end_idx, :]

            # 计算当前块注意力分数
            scores_block = torch.matmul(q, k_block.transpose(-2, -1)) / math.sqrt(self.head_dim)
            block_max = scores_block.max(dim=-1, keepdim=True).values#m_i
            exp_scores = torch.exp(scores_block - block_max)
            block_sum_exp = exp_scores.sum(dim=-1, keepdim=True)#l_i

            # 存储中间结果
            block_outputs.append(torch.matmul(exp_scores, v_block))
            block_max_vals.append(block_max)#m_i
            block_sum_exps.append(block_sum_exp) #l_i

        # 第二步：合并所有块的结果
        # 找到全局最大值
        all_max_vals = torch.stack(block_max_vals, dim=0)  # [num_blocks, ...]
        global_max = all_max_vals.max(dim=0).values  # 在每个query位置取最大值

        # 合并归一化因子
        total_sum_exp = torch.zeros_like(block_sum_exps[0])
        for i in range(num_blocks):
            total_sum_exp += block_sum_exps[i] * torch.exp(block_max_vals[i] - global_max)# l_i_prime

        # 合并输出
        final_output = torch.zeros_like(block_outputs[0])
        for i in range(num_blocks):
            # 将每个块的贡献调整到全局尺度
            weight = torch.exp(block_max_vals[i] - global_max)
            final_output += block_outputs[i] * weight

        # 最终归一化
        final_output = final_output / total_sum_exp

        # 计算完整注意力权重用于验证
        full_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        full_attention_weights = F.softmax(full_scores, dim=-1)

        return final_output, full_attention_weights

    def verify_with_tolerance(self, batch_size=2, seq_len_q=1, seq_len_kv=1024):
        """更严格的验证，包含容差检查"""

        # 生成随机测试数据
        torch.manual_seed(42)  # 固定随机种子以便复现
        q = torch.randn(batch_size, self.num_heads, seq_len_q, self.head_dim)
        k = torch.randn(batch_size, self.num_heads, seq_len_kv, self.head_dim)
        v = torch.randn(batch_size, self.num_heads, seq_len_kv, self.head_dim)

        print("=" * 70)
        print("Flash-Decoding 正确性验证")
        print("=" * 70)

        # 传统方法计算
        traditional_output, traditional_weights = self.traditional_attention(q, k, v)

        # 尝试不同block_size
        block_sizes = [16, 32, 64, 128, 256]

        for block_size in block_sizes:
            print(f"\n使用block_size={block_size}:")

            # 一般方法
            flash_output1, flash_weights1 = self.flash_decoding_attention(
                q, k, v, block_size
            )

            # 简化方法
            flash_output2, flash_weights2 = self.flash_decoding_attention_simple(
                q, k, v, block_size
            )

            # 比较结果
            diff1 = torch.abs(traditional_output - flash_output1).max().item()
            diff2 = torch.abs(traditional_output - flash_output2).max().item()

            # 相对误差
            rel_error1 = diff1 / (torch.abs(traditional_output).max().item() + 1e-10)
            rel_error2 = diff2 / (torch.abs(traditional_output).max().item() + 1e-10)

            # 检查是否在容差范围内
            tolerance = 1e-4
            is_correct1 = diff1 < tolerance
            is_correct2 = diff2 < tolerance

            print(f"  一般方法 - 最大绝对误差: {diff1:.2e}, 相对误差: {rel_error1:.2e}, 正确: {is_correct1}")
            print(f"  简化方法 - 最大绝对误差: {diff2:.2e}, 相对误差: {rel_error2:.2e}, 正确: {is_correct2}")

            # 如果两种方法都正确，还可以比较它们之间的一致性
            if is_correct1 and is_correct2:
                method_diff = torch.abs(flash_output1 - flash_output2).max().item()
                print(f"  两种方法间差异: {method_diff:.2e}")

        return True

    def analyze_numerical_stability(self):
        """数值稳定性分析"""

        print("\n" + "=" * 70)
        print("数值稳定性分析")
        print("=" * 70)

        # 测试不同范围的数值
        test_cases = [
            ("小数值范围", (-1.0, 1.0)),
            ("中等数值范围", (-10.0, 10.0)),
            ("大数值范围", (-50.0, 50.0)),
        ]

        for name, (min_val, max_val) in test_cases:
            print(f"\n{name} [{min_val}, {max_val}]:")

            # 生成特定范围的测试数据
            q = torch.rand(1, self.num_heads, 1, self.head_dim) * (max_val - min_val) + min_val
            k = torch.rand(1, self.num_heads, 1024, self.head_dim) * (max_val - min_val) + min_val
            v = torch.rand(1, self.num_heads, 1024, self.head_dim) * (max_val - min_val) + min_val

            # 传统方法
            traditional_output, _ = self.traditional_attention(q, k, v)

            # Flash-Decoding方法
            flash_output, _ = self.flash_decoding_attention(q, k, v, block_size=64)

            # 计算误差
            diff = torch.abs(traditional_output - flash_output).max().item()

            # 检查是否出现NaN或Inf
            has_nan = torch.isnan(flash_output).any().item()
            has_inf = torch.isinf(flash_output).any().item()

            print(f"  最大绝对误差: {diff:.2e}")
            print(f"  包含NaN: {has_nan}, 包含Inf: {has_inf}")

        return True

# 创建演示实例
demo = FlashDecodingDemo(d_model=512, num_heads=8)

# 1. 验证正确性（使用更严格的验证）
demo.verify_with_tolerance(
    batch_size=2,
    seq_len_q=1,
    seq_len_kv=1024
)

# 2. 数值稳定性分析
demo.analyze_numerical_stability()

# 3. 性能对比演示
print("\n" + "=" * 70)
print("性能对比演示（小batch_size，长序列）")
print("=" * 70)

# 模拟长序列推理场景
seq_lengths = [1024, 4096, 16384, 32768]

for seq_len in seq_lengths:
    print(f"\n序列长度: {seq_len}")

    # 生成测试数据
    q = torch.randn(1, 8, 1, 64)  # batch=1，单token查询
    k = torch.randn(1, 8, seq_len, 64)
    v = torch.randn(1, 8, seq_len, 64)

    # 传统方法时间
    start = time.time()
    traditional_output, _ = demo.traditional_attention(q, k, v)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    traditional_time = (time.time() - start) * 1000

    # Flash-Decoding时间
    start = time.time()
    flash_output, _ = demo.flash_decoding_attention(q, k, v, block_size=256)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    flash_time = (time.time() - start) * 1000

    # 验证一致性
    diff = torch.abs(traditional_output - flash_output).max().item()

    print(f"  传统方法: {traditional_time:.2f}ms")
    print(f"  Flash-Decoding: {flash_time:.2f}ms")
    print(f"  加速比: {traditional_time / flash_time:.2f}x")
    print(f"  输出差异: {diff:.2e}")
    print(f"  结果一致: {diff < 1e-4}")
