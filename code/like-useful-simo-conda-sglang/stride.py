import torch 

num_kv_heads,head_dim=8,128

PACKED_HEAD_SIZE=head_dim
SCALE_HEAD_SIZE=PACKED_HEAD_SIZE//32

SCALE_PLANE_OFFSET = num_kv_heads * PACKED_HEAD_SIZE

k_buf=torch.randn(3, int(num_kv_heads * head_dim + num_kv_heads * head_dim/32))

stride_buf_kbs = k_buf.stride(0)

offs_kv_loc=torch.tensor([0,1,2])
cur_kv_head = 2

offs_d=torch.arange(128)
offs_d_scale=torch.arange(SCALE_HEAD_SIZE)


offs_buf_k = (
    offs_kv_loc[None, :] * stride_buf_kbs
    + cur_kv_head * PACKED_HEAD_SIZE
    + offs_d[:, None]
            )

dim_slice = torch.concat([torch.arange(0,8), torch.arange(128-8,128)])
print(f"cur_kv_head:{cur_kv_head},dim_slice:{dim_slice}")
print(f"slice tensor elem:\n{offs_buf_k[dim_slice, :]}")
# now scale

offs_buf_k_scale = (
    offs_kv_loc[:, None] * stride_buf_kbs + SCALE_PLANE_OFFSET
    + cur_kv_head * SCALE_HEAD_SIZE
    + offs_d_scale[None, :]
            )

dim_slice = torch.arange(SCALE_HEAD_SIZE)
print(f"slice tensor scale:\n{offs_buf_k_scale[:, dim_slice]}")
