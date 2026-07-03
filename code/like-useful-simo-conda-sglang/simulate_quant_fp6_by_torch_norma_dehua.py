F32_MBITS = 23
mbits=3
EXP_BIAS=1
F32_EXP_BIAS=127



magic_adder = (1 << (F32_MBITS - mbits - 1)) - 1
val_to_add = ((EXP_BIAS - F32_EXP_BIAS) << F32_MBITS) + magic_adder
#x_pos_u32=(torch.tensor([3.5],dtype=torch.float32).view(torch.int32))
x_pos_u32=(torch.tensor([5.0],dtype=torch.float32).view(torch.int32))
mant_odd = (x_pos_u32 >> (F32_MBITS - mbits)) & 1

normal_x_u32 = x_pos_u32 + val_to_add
normal_x_u32 += mant_odd
normal_vals = normal_x_u32 >> (F32_MBITS - mbits)
