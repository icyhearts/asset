#!/usr/bin/env python3
"""
加载 decode_attention_fwd_grouped 的保存参数，测试不同 num_kv_splits 对输出的影响。

用法:
  python like-useful/load_sgl_decode_attention_fwd_grouped.py [--num_timestamps N] [--device cuda:0]

对于每个时间戳的参数:
  1. 加载原始参数，调用 decode_attention_fwd_grouped 得到原始输出
  2. 至少进行 2 次不同的 num_kv_splits / max_kv_splits 修改
  3. 用修改后的参数再次调用 decode_attention_fwd_grouped
  4. 对比 cosine similarity, L2 norm, bitwise 是否相同
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


def get_timestamps(save_dir):
    """从保存目录中提取所有唯一的时间戳."""
    timestamps = set()
    for fname in os.listdir(save_dir):
        if fname.endswith(".safetensors"):
            prefix = "decode_attention_fwd_grouped."
            suffix = ".safetensors"
            ts = fname[len(prefix): -len(suffix)]
            timestamps.add(ts)
        elif fname.endswith(".json"):
            prefix = "non_tensor_args."
            suffix = ".json"
            ts = fname[len(prefix): -len(suffix)]
            timestamps.add(ts)
    return sorted(timestamps)


def load_params(save_dir, timestamp):
    """加载某个时间戳的张量和非张量参数."""
    safetensors_path = os.path.join(
        save_dir, f"decode_attention_fwd_grouped.{timestamp}.safetensors"
    )
    json_path = os.path.join(save_dir, f"non_tensor_args.{timestamp}.json")

    data = {}
    with safe_open(safetensors_path, framework="pt") as f:
        for key in f.keys():
            data[key] = f.get_tensor(key)

    with open(json_path) as f:
        non_tensor_args = json.load(f)

    return data, non_tensor_args


def run_decode_attention(data, non_tensor_args, num_kv_splits, max_kv_splits):
    """
    使用给定的 num_kv_splits 和 max_kv_splits 运行 decode_attention_fwd_grouped.

    注意: 该函数对 o, attn_logits, attn_lse 是 in-place 写入，所以需要 clone.
    """
    from sglang.srt.layers.attention.triton_ops.decode_attention import (
        decode_attention_fwd_grouped,
    )

    o = data["o"].clone()
    attn_logits = data["attn_logits"].clone()
    attn_lse = data["attn_lse"].clone()

    # Convert sinks: JSON stores None as Python None or null
    sinks_val = non_tensor_args.get("sinks")
    if sinks_val is not None and not isinstance(sinks_val, torch.Tensor):
        sinks_val = None

    decode_attention_fwd_grouped(
        q=data["q"],
        k_buffer=data["k_buffer"],
        v_buffer=data["v_buffer"],
        o=o,
        kv_indptr=data["kv_indptr"],
        kv_indices=data["kv_indices"],
        attn_logits=attn_logits,
        attn_lse=attn_lse,
        num_kv_splits=num_kv_splits,
        max_kv_splits=max_kv_splits,
        sm_scale_withk=non_tensor_args["sm_scale_withk"],
        v_scale=non_tensor_args["v_scale"],
        logit_cap=non_tensor_args["logit_cap"],
        sinks=sinks_val,
        xai_temperature_len=non_tensor_args["xai_temperature_len"],
        has_mla=non_tensor_args["has_mla"],
        use_pdl=non_tensor_args["use_pdl"],
    )
    return o


def compute_bitwise_match(a, b):
    """检查两个 tensor 是否 bitwise 完全相同."""
    return torch.equal(a, b)


def compute_cosine_similarity(a, b):
    """计算两个 tensor 的 cosine similarity."""
    a_flat = a.float().reshape(-1)
    b_flat = b.float().reshape(-1)
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def compute_l2_norm_diff(a, b):
    """计算两个 tensor 的 L2 差异."""
    return torch.norm((a.float() - b.float())).item()


def generate_modifications(original_num_kv_splits, original_max_kv_splits, bs):
    """
    生成至少 2 组不同的 (num_kv_splits, max_kv_splits) 修改方案.

    修改原则:
    - 确保 max_kv_splits >= max(num_kv_splits)
    - num_kv_splits 不能超出原始 attn_logits buffer 的范围
    - 原始 buffer 大小: attn_logits.shape[2] 决定了 max_kv_splits 的上限
    """
    modifications = []

    # 修改1: 设置所有 num_kv_splits 为 1 (最小分割)
    nks1 = torch.full_like(original_num_kv_splits, 1)
    modifications.append(("all_ones", nks1, original_max_kv_splits))

    # 修改2: 设置所有 num_kv_splits 为原始 max_kv_splits (最大分割)
    nks2 = torch.full_like(original_num_kv_splits, original_max_kv_splits)
    modifications.append(("all_max", nks2, original_max_kv_splits))

    # 修改3: 使用原始值的 2 倍并降低 max_kv_splits (如果原始值较小)
    # 计算合适的新 max_kv_splits
    orig_max_val = original_num_kv_splits.max().item()
    if orig_max_val * 2 <= original_max_kv_splits:
        nks3 = (original_num_kv_splits * 2).clamp(max=original_max_kv_splits).int()
        modifications.append(("double_splits", nks3, original_max_kv_splits))

    # 修改4: 将 max_kv_splits 减半（如果原始 num_kv_splits 都 <= 128）
    half_max = original_max_kv_splits // 2
    if orig_max_val <= half_max:
        nks4 = original_num_kv_splits.clone()
        modifications.append(("half_max", nks4, half_max))

    return modifications


def test_timestamp(save_dir, timestamp, device="cuda"):
    """测试单个时间戳的所有修改方案."""
    try:
        data, non_tensor_args = load_params(save_dir, timestamp)
    except Exception as e:
        return {"timestamp": timestamp, "error": f"Failed to load params: {e}"}

    # Move to device if needed
    for key in data:
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].to(device)

    bs = data["q"].shape[0]
    original_num_kv_splits = data["num_kv_splits"]
    original_max_kv_splits = non_tensor_args["max_kv_splits"]
    # 原始 attn_logits buffer 的 max_kv_splits 维度
    buffer_max_kv_splits = data["attn_logits"].shape[2]

    # 运行原始参数
    original_output = run_decode_attention(
        data, non_tensor_args,
        original_num_kv_splits.clone(),
        original_max_kv_splits,
    )

    # 生成修改方案
    modifications = generate_modifications(
        original_num_kv_splits, buffer_max_kv_splits, bs
    )

    results = []
    for mod_name, nks, mks in modifications:
        # 确保 max_kv_splits >= num_kv_splits.max()
        mks = max(mks, int(nks.max().item()))

        try:
            modified_output = run_decode_attention(
                data, non_tensor_args,
                nks,
                mks,
            )
            bitwise = compute_bitwise_match(original_output, modified_output)
            cosine = compute_cosine_similarity(original_output, modified_output)
            l2_diff = compute_l2_norm_diff(original_output, modified_output)
            l2_orig = torch.norm(original_output.float()).item()
            l2_mod = torch.norm(modified_output.float()).item()

            results.append({
                "mod_name": mod_name,
                "num_kv_splits_orig": original_num_kv_splits.tolist(),
                "num_kv_splits_new": nks.tolist(),
                "max_kv_splits_orig": original_max_kv_splits,
                "max_kv_splits_new": mks,
                "bitwise_identical": bitwise,
                "cosine_similarity": cosine,
                "l2_diff": l2_diff,
                "l2_original": l2_orig,
                "l2_modified": l2_mod,
                "l2_relative_diff": l2_diff / l2_orig if l2_orig > 0 else 0.0,
            })
        except Exception as e:
            results.append({
                "mod_name": mod_name,
                "error": str(e),
            })

    return {
        "timestamp": timestamp,
        "bs": bs,
        "original_num_kv_splits_values": original_num_kv_splits.tolist(),
        "original_max_kv_splits": original_max_kv_splits,
        "buffer_max_kv_splits": buffer_max_kv_splits,
        "modifications": results,
    }


def print_summary(all_results):
    """打印汇总结果."""
    print("\n" + "=" * 100)
    print("SUMMARY: decode_attention_fwd_grouped num_kv_splits modification test")
    print("=" * 100)

    total = len(all_results)
    errors = sum(1 for r in all_results if "error" in r)
    success = total - errors
    print(f"Total timestamps tested: {total}, Success: {success}, Errors: {errors}")

    # Aggregate per-modification statistics
    mod_stats = defaultdict(lambda: {
        "bitwise_same": 0,
        "count": 0,
        "cosine_sims": [],
        "l2_diffs": [],
        "l2_relative_diffs": [],
    })

    for r in all_results:
        if "error" in r:
            continue
        for mod in r["modifications"]:
            if "error" in mod:
                continue
            name = mod["mod_name"]
            s = mod_stats[name]
            if mod["bitwise_identical"]:
                s["bitwise_same"] += 1
            s["count"] += 1
            s["cosine_sims"].append(mod["cosine_similarity"])
            s["l2_diffs"].append(mod["l2_diff"])
            s["l2_relative_diffs"].append(mod["l2_relative_diff"])

    print("\n--- Per-modification statistics ---")
    for mod_name in sorted(mod_stats.keys()):
        s = mod_stats[mod_name]
        cosines = np.array(s["cosine_sims"])
        l2_rel = np.array(s["l2_relative_diffs"])
        print(f"\n  Modification: {mod_name}")
        print(f"    Count: {s['count']}")
        print(f"    Bitwise identical: {s['bitwise_same']}/{s['count']}")
        print(f"    Cosine similarity: min={cosines.min():.10f}, max={cosines.max():.10f}, mean={cosines.mean():.10f}")
        print(f"    L2 relative diff: min={l2_rel.min():.2e}, max={l2_rel.max():.2e}, mean={l2_rel.mean():.2e}")

    # Detail for first few timestamps
    print("\n--- Detail: first 3 timestamps ---")
    for r in all_results[:3]:
        if "error" in r:
            print(f"  {r['timestamp']}: ERROR - {r['error']}")
            continue
        print(f"\n  Timestamp: {r['timestamp']}")
        print(f"    bs={r['bs']}, original num_kv_splits={r['original_num_kv_splits_values']}, max_kv_splits={r['original_max_kv_splits']}")
        for mod in r["modifications"]:
            if "error" in mod:
                print(f"    {mod['mod_name']}: ERROR - {mod['error']}")
            else:
                print(f"    {mod['mod_name']}:")
                print(f"      num_kv_splits: {mod['num_kv_splits_orig']} -> {mod['num_kv_splits_new']}")
                print(f"      max_kv_splits: {mod['max_kv_splits_orig']} -> {mod['max_kv_splits_new']}")
                print(f"      bitwise: {mod['bitwise_identical']}")
                print(f"      cosine:  {mod['cosine_similarity']:.10f}")
                print(f"      L2 diff: {mod['l2_diff']:.6e} (relative: {mod['l2_relative_diff']:.6e})")


def append_to_answer_md(all_results, output_md):
    """将测试结论追加到 answer_claude.md."""
    total = len(all_results)
    errors = sum(1 for r in all_results if "error" in r)
    success = total - errors

    # Aggregate per-modification
    mod_stats = defaultdict(lambda: {
        "bitwise_same": 0, "count": 0,
        "cosine_sims": [], "l2_diffs": [], "l2_relative_diffs": [],
    })
    for r in all_results:
        if "error" in r:
            continue
        for mod in r["modifications"]:
            if "error" in mod:
                continue
            name = mod["mod_name"]
            s = mod_stats[name]
            if mod["bitwise_identical"]:
                s["bitwise_same"] += 1
            s["count"] += 1
            s["cosine_sims"].append(mod["cosine_similarity"])
            s["l2_diffs"].append(mod["l2_diff"])
            s["l2_relative_diffs"].append(mod["l2_relative_diff"])

    lines = [
        "",
        "## decode_attention_fwd_grouped num_kv_splits 修改测试",
        "",
        f"测试时间戳数量: {total}, 成功: {success}, 失败: {errors}",
        "",
    ]

    for mod_name in sorted(mod_stats.keys()):
        s = mod_stats[mod_name]
        cosines = np.array(s["cosine_sims"])
        l2_rel = np.array(s["l2_relative_diffs"])
        lines.append(f"### 修改方案: {mod_name}")
        lines.append(f"- 测试次数: {s['count']}")
        lines.append(f"- Bitwise 完全相同: {s['bitwise_same']}/{s['count']}")
        lines.append(f"- Cosine similarity: min={cosines.min():.10f}, max={cosines.max():.10f}, mean={cosines.mean():.10f}")
        lines.append(f"- L2 relative diff: min={l2_rel.min():.2e}, max={l2_rel.max():.2e}, mean={l2_rel.mean():.2e}")
        lines.append("")

    lines.append("### 结论")
    lines.append("num_kv_splits 的修改对 attention output 的影响如下:")
    lines.append("")

    with open(output_md, "a") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Test decode_attention_fwd_grouped with different num_kv_splits"
    )
    parser.add_argument(
        "--num_timestamps", type=int, default=None,
        help="限制测试的时间戳数量 (默认全部)"
    )
    parser.add_argument(
        "--save_dir",
        default="/data/like/temp/sgl_safe_tensor_sgl_decode_attention_fwd_grouped_dir",
        help="保存参数的目录"
    )
    parser.add_argument(
        "--device", default="cuda",
        help="运行设备 (默认: cuda)"
    )
    parser.add_argument(
        "--output_md",
        default="like-useful/answer_claude.md",
        help="追加结论的 markdown 文件"
    )
    parser.add_argument(
        "--skip_md", action="store_true",
        help="跳过追加到 answer_claude.md"
    )
    args = parser.parse_args()

    save_dir = args.save_dir
    if not os.path.isdir(save_dir):
        print(f"ERROR: save_dir not found: {save_dir}")
        sys.exit(1)

    # Get timestamps
    timestamps = get_timestamps(save_dir)
    print(f"Found {len(timestamps)} unique timestamps in {save_dir}")

    if args.num_timestamps is not None:
        timestamps = timestamps[: args.num_timestamps]
        print(f"Testing first {len(timestamps)} timestamps")

    all_results = []
    for i, ts in enumerate(timestamps):
        print(f"\r[{i+1}/{len(timestamps)}] Testing timestamp {ts}...", end="", flush=True)
        result = test_timestamp(save_dir, ts, args.device)
        all_results.append(result)

    print()  # newline after progress

    # Print summary
    print_summary(all_results)

    # Append to answer_claude.md
    if not args.skip_md:
        append_to_answer_md(all_results, args.output_md)
        print(f"\nResults appended to {args.output_md}")


if __name__ == "__main__":
    main()
