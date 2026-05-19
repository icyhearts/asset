#https://github.com/CalvinXKY/InfraTech/blob/main/llm_infer/chunked_prefill_and_flash_decoding.ipynb

import torch
import torch.nn.functional as F
import math

class FinalFlashDecodingTiling:
    """
    最终版Flash-Decoding Tiling实现
    仅存储O和S，使用两步合并算法
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

    def traditional_attention(self, q, k, v):
        """基准：传统连续注意力计算"""
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output

    def compute_stream_output(self, q, k_tile, v_tile):
        """
        计算单个tile的流输出
        返回: (O_i, S_i) 其中 S_i = m_i + log(l_i)
        """
        # 计算当前tile的注意力分数
        S_tile = torch.matmul(q, k_tile.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 计算m_i和l_i
        m_i = S_tile.max(dim=-1, keepdim=True).values
        exp_tile = torch.exp(S_tile - m_i)
        l_i = exp_tile.sum(dim=-1, keepdim=True)

        # 计算加权和 O_i（已经是归一化的）
        O_i = torch.matmul(exp_tile, v_tile) / l_i

        # 计算 S_i = m_i + log(l_i)
        log_l_i = torch.log(l_i + 1e-12)
        S_i = m_i + log_l_i

        return O_i, S_i

    def merge_streams_two_step(self, streams_data):
        """
        两步合并算法：
        1. 迭代计算全局 S_global
        2. 用 S_global 修正每个流的输出贡献
        """
        if not streams_data:
            return None

        # 提取所有流的S_i
        S_list = [S_i for _, S_i in streams_data]

        # 步骤1: 迭代计算全局 S_global (S_lst)
        S_global = S_list[0].clone()

        for i in range(1, len(S_list)):
            S_i = S_list[i]
            S_max = torch.maximum(S_global, S_i)
            S_min = torch.minimum(S_global, S_i)
            # 使用log(1+exp(x))的稳定计算
            log_term = torch.log1p(torch.exp(S_min - S_max))
            S_global = S_max + log_term

        # 步骤2: 修正每个流的输出贡献
        O_global = torch.zeros_like(streams_data[0][0])

        for O_i, S_i in streams_data:
            # 计算该流对全局的贡献权重
            weight = torch.exp(S_i - S_global)
            # 累加加权贡献
            O_global += O_i * weight

        return O_global

    def flash_decoding_with_lse(self, q, k, v,
                            tile_size_kv: int = 256,
                            num_streams: int = 4):
        """
        Flash-Decoding 仅存储O和S
        """
        batch_size, num_heads, seq_len_q, head_dim = q.shape
        seq_len_kv = k.shape[2]
        num_tiles = (seq_len_kv + tile_size_kv - 1) // tile_size_kv

        print(f"使用两步合并算法Flash-Decoding: {num_streams}个流")

        # 初始化流数组
        streams_data = []

        for stream_id in range(num_streams):
            # 每个流存储(O_i, S_i)
            O_stream = torch.zeros_like(q)
            S_stream = torch.full((batch_size, num_heads, seq_len_q, 1),
                                -float('inf'), device=q.device, dtype=q.dtype)
            streams_data.append((O_stream, S_stream))

        # 处理每个tile
        print(f"处理{num_tiles}个tile...")

        for tile_idx in range(num_tiles):
            stream_id = tile_idx % num_streams

            start_idx = tile_idx * tile_size_kv
            end_idx = min(start_idx + tile_size_kv, seq_len_kv)

            k_tile = k[:, :, start_idx:end_idx, :]
            v_tile = v[:, :, start_idx:end_idx, :]

            # 计算当前tile的输出
            O_i, S_i = self.compute_stream_output(q, k_tile, v_tile)

            # 获取当前流的累加器
            O_acc, S_acc = streams_data[stream_id]

            # 合并当前tile结果到流累加器
            if torch.all(S_acc == -float('inf')):
                streams_data[stream_id] = (O_i, S_i)
            else:
                # 使用两步法合并当前tile到流累加器
                # 先计算合并后的S
                S_max = torch.maximum(S_acc, S_i)
                S_min = torch.minimum(S_acc, S_i)
                log_term = torch.log1p(torch.exp(S_min - S_max))
                S_merged = S_max + log_term

                # 修正两个部分的贡献
                weight_acc = torch.exp(S_acc - S_merged)
                weight_i = torch.exp(S_i - S_merged)
                O_merged = O_acc * weight_acc + O_i * weight_i

                streams_data[stream_id] = (O_merged, S_merged)

        print(f"所有tile处理完成，开始归约所有流...")
        # 归约所有流的结果
        O_final = self.merge_streams_two_step(streams_data)

        return O_final

    def verify_correctness(self, seq_len_kv: int = 2048):
        """验证实现的正确性"""

        torch.manual_seed(42)
        batch_size = 2
        seq_len_q = 1

        q = torch.randn(batch_size, self.num_heads, seq_len_q, self.head_dim)
        k = torch.randn(batch_size, self.num_heads, seq_len_kv, self.head_dim)
        v = torch.randn(batch_size, self.num_heads, seq_len_kv, self.head_dim)

        print("=" * 80)
        print("基于lse的Flash-Decoding")
        print("=" * 80)

        # 基准测试：传统方法
        print(f"\n1. 传统注意力计算...")
        baseline = self.traditional_attention(q, k, v)

        # 基于lse的Flash-Decoding
        print(f"\n2. 基于lse的Flash-Decoding...")
        output = self.flash_decoding_with_lse(q, k, v)

        # 验证正确性
        diff = torch.abs(baseline - output).max().item()
        rel_error = diff / torch.abs(baseline).max().item()

        print(f"\n验证结果:")
        print(f"  最大绝对误差: {diff:.2e}")
        print(f"  相对误差: {rel_error:.2e}")
        print(f"  合并算法是否正确: {diff < 1e-4}")

        # 数学正确性验证
        print(f"\n3. 数学正确性验证（小规模测试）...")

        # 创建一个小测试
        torch.manual_seed(123)
        q_test = torch.randn(1, 2, 1, 4)
        k_test = torch.randn(1, 2, 8, 4)
        v_test = torch.randn(1, 2, 8, 4)

        baseline_test = self.traditional_attention(q_test, k_test, v_test)
        output_test = self.flash_decoding_with_lse(q_test, k_test, v_test,
                                               tile_size_kv=4, num_streams=2)

        diff_test = torch.abs(baseline_test - output_test).max().item()
        print(f"  小规模测试最大绝对误差: {diff_test:.2e}")
        print(f"  小规模测试是否正确: {diff_test < 1e-4}")

        return {
            'baseline': baseline,
            'output': output,
            'error': diff,
            'correct': diff < 1e-4
        }


if __name__ == "__main__":
    demo = FinalFlashDecodingTiling(d_model=512, num_heads=8)

    # 验证最终实现的正确性
    print("基于lse的Flash-Decoding验证")
    results = demo.verify_correctness(seq_len_kv=2048)

    if results['correct']:
        print("\n✅ 算法现在可以正确合并结果。")
    else:
        print(f"\n❌ 仍然存在问题，误差: {results['error']:.2e}")

    # 性能测试
    print("\n" + "=" * 80)
    print("性能测试")
    print("=" * 80)

    import time
    torch.manual_seed(42)
    batch_size = 2
    seq_len_q = 1
    seq_len_kv = 8192

    q = torch.randn(batch_size, 8, seq_len_q, 64)
    k = torch.randn(batch_size, 8, seq_len_kv, 64)
    v = torch.randn(batch_size, 8, seq_len_kv, 64)

    # 传统方法
    start = time.time()
    baseline = demo.traditional_attention(q, k, v)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    trad_time = time.time() - start

    # 优化方法
    start = time.time()
    output = demo.flash_decoding_with_lse(q, k, v, tile_size_kv=256)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    opt_time = time.time() - start

    diff = torch.abs(baseline - output).max().item()

    print(f"序列长度: {seq_len_kv}")
    print(f"传统方法时间: {trad_time:.4f}s")
    print(f"优化方法时间: {opt_time:.4f}s")
    print(f"加速比: {trad_time/opt_time:.2f}x")
    print(f"误差: {diff:.2e}")
    print(f"是否一致: {diff < 1e-4}")
