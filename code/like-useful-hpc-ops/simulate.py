import numpy as np
import torch
num_group=8
seqlens = (1+torch.arange(num_group, dtype=torch.int32, device='cuda'))*16
cu_seqlens = torch.cumsum( torch.cat([torch.tensor([0], dtype=torch.int32, device="cuda"), seqlens]), dim=0).to(torch.int32)
seqlens_ptr = seqlens
kTileM = 48
kGroupPerThread = 8
kThreadPerBlock = 32

# out
tiles_ptr = torch.zeros(num_group, dtype=torch.int32, device=seqlens.device)
cu_tiles_ptr =  torch.zeros(num_group+1, dtype=torch.int32, device=seqlens.device)

tiles_thread_block = torch.zeros(kThreadPerBlock, kGroupPerThread, dtype=torch.int32, device=seqlens.device)
block_aggregate = 0
for idx in range(kThreadPerBlock):
    tiles = torch.zeros(kGroupPerThread, dtype=torch.int32, device=seqlens.device)
    for i in range(kGroupPerThread):
        igroup = idx * kGroupPerThread + i
        if igroup < num_group:
            tiles[i] = (seqlens_ptr[igroup] + kTileM - 1) // kTileM
            tiles_ptr[igroup] = tiles[i]
        else:
            tiles[i] = 0
        block_aggregate += tiles[i]
    print(f"idx:{idx},tiles:{tiles}")
    tiles_thread_block[idx] = tiles # This is cpu
# simulate ExclusiveSum
zero=torch.zeros(1,dtype=tiles_thread_block.dtype, device=tiles_thread_block.device)
inclusive=torch.cumsum(tiles_thread_block.reshape(-1), dim=0)
ExclusiveSum_flatten = torch.cat([zero, inclusive])
for idx in range(kThreadPerBlock):
    tiles_thread_block[idx] = ExclusiveSum_flatten[idx*kGroupPerThread: (idx+1)*kGroupPerThread]

#block_aggregate = tiles_thread_block.sum().item()
for idx in range(kThreadPerBlock):
    for i in range( kGroupPerThread):
      igroup = idx * kGroupPerThread + i
      tiles = tiles_thread_block[idx]
      if igroup < num_group:
        cu_tiles_ptr[igroup] = tiles[i]
    if idx == 0:
      cu_tiles_ptr[num_group] = block_aggregate
print(f"cu_tiles_ptr:{cu_tiles_ptr}")
"""
如下cuda 代码 实现了什么功能?
    constexpr int kGroupPerThread = 8
    constexpr int kThreadPerBlock = 32
    int tiles[kGroupPerThread];
    using BlockScan = cub::BlockScan<int, kThreadPerBlock>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    int block_aggregate;
    BlockScan(temp_storage).ExclusiveSum(tiles, tiles, block_aggregate);
"""
