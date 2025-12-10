"""
Test output projection indexing
"""
import torch
import torch.nn.functional as F

# Simulated parameters
HEAD_DIM = 80
TMA_LOAD_ONCE = 64
num_heads = 32
hidden_size = 2560

# Create test data
torch.manual_seed(42)
attn_output = torch.randn(1, num_heads, HEAD_DIM, dtype=torch.float16, device='cuda:0')
weight_o = torch.randn(hidden_size, hidden_size, dtype=torch.float16, device='cuda:0')

# Ground truth: PyTorch
attn_flat = attn_output.reshape(1, num_heads * HEAD_DIM)
output_pytorch = F.linear(attn_flat, weight_o, None)

print("Ground truth (PyTorch):")
print(f"  Output norm: {output_pytorch.norm().item():.4f}")
print(f"  First 8: {output_pytorch[0, :8]}")

# Simulate Pythia kernel indexing
# Assume we process head_id=0 and compute first TMA_LOAD_ONCE outputs
output_pythia = torch.zeros(1, hidden_size, dtype=torch.float16, device='cuda:0')

head_id = 0  # First head
cluster_block_st_id = 0  # First block

# Simulate loading weight block: weight[head_id*HEAD_DIM:head_id*HEAD_DIM+HEAD_DIM, cluster_block_st_id:cluster_block_st_id+TMA_LOAD_ONCE]
# This is a [HEAD_DIM, TMA_LOAD_ONCE] block
weight_block = weight_o[cluster_block_st_id:cluster_block_st_id+TMA_LOAD_ONCE, head_id*HEAD_DIM:(head_id+1)*HEAD_DIM]  # [TMA_LOAD_ONCE, HEAD_DIM]

# Input from this head
input_block = attn_output[0, head_id, :]  # [HEAD_DIM]

# Compute contribution using PyTorch (as reference for kernel logic)
# output[cluster_block_st_id:cluster_block_st_id+TMA_LOAD_ONCE] += input_block @ weight_block.T
contribution = torch.matmul(input_block.float(), weight_block.T.float()).half()

print(f"\nSimulated kernel:")
print(f"  Weight block shape: {weight_block.shape}")
print(f"  Input shape: {input_block.shape}")
print(f"  Contribution shape: {contribution.shape}")
print(f"  Contribution norm: {contribution.norm().item():.4f}")

# Check if Pythia indexing matches
# Pythia: weight[weight_idx_3 * HEAD_DIM + (input_idx_3 + j + d)]
# After TMA load, in shared memory it should be layout [TMA_LOAD_ONCE, HEAD_DIM]
# So accessing row i, col j should be: weight[i * HEAD_DIM + j]

# But in kernel, it uses:
# weight[weight_idx_3 * HEAD_DIM + (input_idx_3 + j + d)]
# where weight_idx_3 is output index, input_idx_3 is input index

print("\n=== Analyzing indexing ===")
print(f"Pythia uses: weight[weight_idx_3 * HEAD_DIM + (input_idx_3 + j + d)]")
print(f"  This treats weight_idx_3 as row, input_idx_3 as column")
print(f"  Expected shape in shmem: [TMA_LOAD_ONCE, HEAD_DIM]")
print(f"Llama uses: weight[(input_idx_3 + j + d) * TMA_LOAD_ONCE + weight_idx_3]")
print(f"  This treats input_idx_3 as row, weight_idx_3 as column")
print(f"  Expected shape in shmem: [HEAD_DIM, TMA_LOAD_ONCE]")

print("\n=== Which is correct? ===")
print("TMA loads box_size={HEAD_DIM, TMA_LOAD_ONCE}")
print("This means HEAD_DIM elements in dim-0, TMA_LOAD_ONCE in dim-1")
print("For weight_o[out_dim, in_dim], we load:")
print("  in_dim: head_id*HEAD_DIM to head_id*HEAD_DIM+HEAD_DIM")
print("  out_dim: cluster_block_st_id to cluster_block_st_id+TMA_LOAD_ONCE")
print("So in shared memory: shmem[in_idx][out_idx]")
print("Llama indexing is CORRECT: (input_idx) * TMA_LOAD_ONCE + (output_idx)")

