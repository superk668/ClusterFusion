"""
Debug KV cache write-back functionality
"""
import torch
import clusterfusion

# Simple test
hidden_size = 2560
num_heads = 32
head_dim = 80
rotary_dim = 20

# Create tensors
input_tensor = torch.randn(1, hidden_size, dtype=torch.float16, device='cuda:0')
weight_qkv = torch.randn(3 * num_heads * head_dim, hidden_size, dtype=torch.float16, device='cuda:0')
bias_qkv = torch.randn(3 * num_heads * head_dim, dtype=torch.float16, device='cuda:0')
weight_o = torch.randn(hidden_size, hidden_size, dtype=torch.float16, device='cuda:0')
ln_weight = torch.ones(1, hidden_size, dtype=torch.float16, device='cuda:0')
ln_bias = torch.zeros(1, hidden_size, dtype=torch.float16, device='cuda:0')
cos = torch.randn(1, head_dim, dtype=torch.float32, device='cuda:0')
sin = torch.randn(1, head_dim, dtype=torch.float32, device='cuda:0')

# KV cache for 10 positions (simulate prefill with 5 tokens, then decode 5 more)
max_seq_len = 10
k_cache = torch.randn(max_seq_len, hidden_size, dtype=torch.float16, device='cuda:0')
v_cache = torch.randn(max_seq_len, hidden_size, dtype=torch.float16, device='cuda:0')

print("Testing KV cache write-back...")
print(f"Max seq len: {max_seq_len}")

# First decode step: current_seq_len = 5 (5 prefill tokens)
current_len = 5
print(f"\n=== Decode Step 1: current_len={current_len} ===")
print(f"K cache before (position {current_len}, first 8 values):")
print(k_cache[current_len, :8])

        output1, new_k1, new_v1 = clusterfusion.pythia_decoder_layer(
    input_tensor,
    weight_qkv,
    bias_qkv,
    weight_o,
            torch.zeros((hidden_size,), dtype=torch.float16, device='cuda:0'),
    k_cache,
    v_cache,
    ln_weight,
    ln_bias,
    cos,
    sin,
    current_len
)

print(f"Returned new_k (first 8 values): {new_k1[0, :8]}")
print(f"K cache after (position {current_len}, first 8 values):")
print(k_cache[current_len, :8])

# Check if write-back worked
if torch.allclose(k_cache[current_len, :num_heads * head_dim], new_k1.reshape(-1), atol=1e-3):
    print("✓ KV write-back appears to work!")
else:
    print("✗ KV write-back FAILED!")
    print(f"Expected: {new_k1.reshape(-1)[:8]}")
    print(f"Got: {k_cache[current_len, :8]}")

# Second decode step: current_seq_len = 6
current_len = 6
print(f"\n=== Decode Step 2: current_len={current_len} ===")
print(f"K cache before (position {current_len}, first 8 values):")
print(k_cache[current_len, :8])

        output2, new_k2, new_v2 = clusterfusion.pythia_decoder_layer(
    input_tensor,
    weight_qkv,
    bias_qkv,
    weight_o,
            torch.zeros((hidden_size,), dtype=torch.float16, device='cuda:0'),
    k_cache,
    v_cache,
    ln_weight,
    ln_bias,
    cos,
    sin,
    current_len
)

print(f"Returned new_k (first 8 values): {new_k2[0, :8]}")
print(f"K cache after (position {current_len}, first 8 values):")
print(k_cache[current_len, :8])

if torch.allclose(k_cache[current_len, :num_heads * head_dim], new_k2.reshape(-1), atol=1e-3):
    print("✓ KV write-back appears to work!")
else:
    print("✗ KV write-back FAILED!")

