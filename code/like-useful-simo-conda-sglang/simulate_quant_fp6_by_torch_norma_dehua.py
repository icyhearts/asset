import torch
F32_MBITS = 23
mbits=3
EXP_BIAS=1
F32_EXP_BIAS=127


def print_tensor_in_hex(arr):
  arr_bytes = arr.view(torch.uint8)
  fstr = ""
  for idx in range(3,-1,-1):
    fstr += "," + hex(arr_bytes[idx])
  print( fstr)

magic_adder = (1 << (F32_MBITS - mbits - 1)) - 1
print(f"magic_adder:{hex(magic_adder)}")

val_to_add_step1 = torch.tensor([EXP_BIAS - F32_EXP_BIAS],dtype=torch.int32)
val_to_add_step2 = val_to_add_step1 << F32_MBITS
val_to_add = val_to_add_step2 + magic_adder
print_tensor_in_hex(val_to_add.view(torch.int32))
#x_pos_u32=(torch.tensor([3.5],dtype=torch.float32).view(torch.int32))
x_pos_u32=(torch.tensor([5.0],dtype=torch.float32).view(torch.int32))
mant_odd = (x_pos_u32 >> (F32_MBITS - mbits)) & 1

normal_x_u32 = x_pos_u32 + val_to_add
normal_x_u32 += mant_odd
normal_vals = normal_x_u32 >> (F32_MBITS - mbits)
