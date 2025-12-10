"""
Verify weight_o layout and TMA loading
"""
import torch
from transformers import AutoModelForCausalLM

MODEL_NAME = "EleutherAI/pythia-2.8b"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map='cuda:0')

weight_o = model.gpt_neox.layers[0].attention.dense.weight.data

print(f"weight_o shape: {weight_o.shape}")
print(f"  Interpretation: [out_features, in_features]")
print(f"  = [hidden_dim, hidden_dim]")
print(f"  = [2560, 2560]")

print(f"\nTensor map config:")
print(f"  size = {{HIDDEN_DIM, HIDDEN_DIM}} = {{2560, 2560}}")
print(f"  box_size = {{HEAD_DIM, TMA_LOAD_ONCE}} = {{80, 64}}")
print(f"  stride = {{HIDDEN_DIM * sizeof(half)}} (row-major)")

print(f"\nTMA load at (head_id*80, cluster_block_st_id):")
print(f"  This loads weight_o[cluster_block_st_id:cluster_block_st_id+64, head_id*80:head_id*80+80]")
print(f"  = output rows [cluster_block_st_id:cluster_block_st_id+64]")
print(f"  = input cols [head_id*80:head_id*80+80]")

print(f"\nComputation:")
print(f"  output[out_idx] = sum_in(input[in_idx] * weight[out_idx, in_idx])")
print(f"  We have input from head_id (80 dims)")
print(f"  We compute output for cluster_block (64 dims)")
print(f"  weight block in shmem: [64 output dims, 80 input dims]")

print(f"\nLlama shmem indexing: weight[(in_idx) * TMA_LOAD_ONCE + (out_idx)]")
print(f"  This assumes shmem layout: [80, 64] (input_dim x output_dim)")
print(f"  weight[5, 3] means: input_dim=5, output_dim=3")
print(f"  Linear index: 5 * 64 + 3")

print(f"\nPythia shmem indexing: weight[(out_idx) * HEAD_DIM + (in_idx)]")
print(f"  This assumes shmem layout: [64, 80] (output_dim x input_dim)")
print(f"  weight[3, 5] means: output_dim=3, input_dim=5")
print(f"  Linear index: 3 * 80 + 5")

print(f"\n=== CRITICAL QUESTION ===")
print(f"What is TMA's actual layout in shared memory?")
print(f"TMA box_size={{80, 64}} means:")
print(f"  - 80 elements in dimension-0")
print(f"  - 64 elements in dimension-1")
print(f"For weight[out, in], loading weight[cluster:cluster+64, head:head+80]:")
print(f"  - dim-0 corresponds to 'in' (80 elements)")
print(f"  - dim-1 corresponds to 'out' (64 elements)")
print(f"  - So shmem layout is [in_dim, out_dim] = [80, 64]")
print(f"  - Llama's indexing is CORRECT")
print(f"  - Pythia's indexing is WRONG")

