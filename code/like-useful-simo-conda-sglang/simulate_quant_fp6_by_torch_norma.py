# f32
x_pos_u32=(torch.tensor([3.5],dtype=torch.float32).view(torch.int32))

biased_exp = (x_pos_u32 >> 23) & 0xFF            # 读 FP32 exponent
adjusted_exp = biased_exp - 126                   # 偏置 127 → 1
mantissa = x_pos_u32 & 0x7FFFFF | 0x800000       # 尾数 + 隐含前导 1
truncated = (mantissa + 0x40000) >> 20            # 加 0.5 ULP 后截断
result = (adjusted_exp << 3) | (truncated & 0x7)  # 组合


#
