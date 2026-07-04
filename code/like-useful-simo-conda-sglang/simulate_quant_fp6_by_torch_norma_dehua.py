import torch
F32_MBITS = 23
mbits=3
EXP_BIAS=1
F32_EXP_BIAS=127


def print_tensor_in_hex(arr):
  arr_bytes = arr.view(torch.uint8)
  fstr = "0X"
  bin_str = "0b"
  for idx in range(3,-1,-1):
    #fstr += "," + hex(arr_bytes[idx])
    #bin_str += "," + bin(arr_bytes[idx])
    fstr += "," + f"{arr_bytes[idx]:02X}"
    bin_str += "," + f"{arr_bytes[idx]:08b}"
  print( fstr)
  print( bin_str)

magic_adder = (1 << (F32_MBITS - mbits - 1)) - 1
print(f"magic_adder:= 1 << 19 -1")
print_tensor_in_hex(torch.tensor([magic_adder],dtype=torch.int32))
print("")

val_to_add_step1 = torch.tensor([EXP_BIAS - F32_EXP_BIAS],dtype=torch.int32)
print(f"val_to_add_step1:{val_to_add_step1} = EXP_BIAS - F32_EXP_BIAS")
print_tensor_in_hex(val_to_add_step1)
print("")

val_to_add_step2 = val_to_add_step1 << F32_MBITS
print(f"val_to_add_step2:{val_to_add_step2}=val_to_add_step1 << {F32_MBITS}")
print_tensor_in_hex(val_to_add_step2)
print("")

val_to_add = val_to_add_step2 + magic_adder
print("val_to_add")
print_tensor_in_hex(val_to_add.view(torch.int32))
print("")

# 1) no ceil, no mant_odd
#x_pos = torch.tensor([3.5],dtype=torch.float32)
x_pos = torch.tensor([3.5],dtype=torch.float32)
# 2) ceil, no mant_odd
x_pos = torch.tensor([0x40690000],dtype=torch.int32).view(torch.float32)

print(f"x_pos:{x_pos}")
x_pos_u32=x_pos.view(torch.int32)
print_tensor_in_hex(x_pos_u32)
print("")
#x_pos_u32=(torch.tensor([5.0],dtype=torch.float32).view(torch.int32))


normal_x_u32 = x_pos_u32 + val_to_add
print(f"normal_x_u32 = x_pos_u32 + val_to_add:{normal_x_u32}")
print_tensor_in_hex(normal_x_u32)
print("")

mant_odd = (x_pos_u32 >> (F32_MBITS - mbits)) & 1
print(f"mant_odd:{mant_odd}")

normal_x_u32 += mant_odd
print(f"normal_x_u32 += mant_odd :{normal_x_u32}")
print_tensor_in_hex(normal_x_u32)
print("")

normal_vals = normal_x_u32 >> (F32_MBITS - mbits)
print(f"normal_vals = normal_x_u32 >> ({F32_MBITS} - {mbits}):{normal_vals}")
print_tensor_in_hex(normal_vals)
