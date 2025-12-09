import torch
import torch.nn as nn
import torch.nn.functional as F
import flashinfer
import torch.cuda.nvtx as nvtx
import clusterfusion

# Pythia-2.8b model parameters
hidden_size = 2560
num_heads = 32
seqlen = 2048
head_dim = hidden_size // num_heads  # 80
ffn_dim = 10240
rotary_dim = head_dim // 4  # 20 (rotary_pct = 0.25)

torch.manual_seed(42)

# Enable Debug print
debug = 0
print_head = 1
if debug:
    test_run = 10
else:
    test_run = 10000

def initialize_rope_embeddings(rotary_dim):
    """Initialize RoPE embeddings for only rotary_dim dimensions"""
    angles = (torch.rand((1, rotary_dim), dtype=torch.float32) * (2 * torch.pi)).to(0)
    h_cos = torch.cos(angles)
    h_sin = torch.sin(angles)
    return h_cos, h_sin

def apply_neox_style_rotary_pos_emb_partial(q, k, cos, sin, rotary_dim):
    """
    Apply Neox-style RoPE only to the first rotary_dim dimensions.
    For Pythia, rotary_pct=0.25, so only first 20 dims (out of 80) use RoPE.
    """
    cos = cos.unsqueeze(1)  # [1, 1, rotary_dim]
    sin = sin.unsqueeze(1)
    
    # Split q and k into rotary and non-rotary parts
    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]
    
    # Apply RoPE to rotary part
    q_rot_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    
    # Concatenate back
    q_embed = torch.cat([q_rot_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_rot_embed, k_pass], dim=-1)
    
    return q_embed.to(q.dtype), k_embed.to(k.dtype)

def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def pythia_decode(hidden, residual, layernorm_weight, layernorm_bias, eps, kv_cache, qkv_proj, o_proj, head_dim, cos, sin, rotary_dim):
    """
    Pythia decoding reference implementation.
    Note: Pythia uses LayerNorm, not RMSNorm.
    """
    # DEBUG PRINT
    if debug:
        print("----------------------------- python begin -----------------------------")

    # LayerNorm (Pythia uses LayerNorm, not RMSNorm)
    # For simplicity in testing, we can use RMSNorm as approximation
    # or implement proper LayerNorm here
    flashinfer.fused_add_rmsnorm(hidden, residual, layernorm_weight, eps)
    residual = hidden
    
    qkv_new = qkv_proj(hidden).view(3, num_heads, head_dim)
    q = qkv_new[0].view(1, num_heads, head_dim)
    k_new = qkv_new[1].view(1, num_heads, head_dim)
    v_new = qkv_new[2].view(1, num_heads, head_dim)

    # DEBUG PRINT
    if debug: 
        print("normed ref", hidden[..., 0: 80])
        print("before RoPE")
        print(f"q, head_id = {print_head}: first 8, last 8")
        print(f"{q[0, print_head, 0: 8]}")
        print(f"{q[0, print_head, 72: 80]}")
        print(f"k_new, head_id = {print_head}: first 8, last 8")
        print(f"{k_new[0, print_head, 0: 8]}")
        print(f"{k_new[0, print_head, 72: 80]}")

    # Apply RoPE only to first rotary_dim dimensions (Neox-style)
    q, k_new = apply_neox_style_rotary_pos_emb_partial(q, k_new, cos, sin, rotary_dim)

    # DEBUG PRINT
    if debug: 
        print("after RoPE")
        print(f"q, head_id = {print_head}: first 8, last 8")
        print(f"{q[0, print_head, 0: 8]}")
        print(f"{q[0, print_head, 72: 80]}")
        print(f"k_new, head_id = {print_head}: first 8, last 8")
        print(f"{k_new[0, print_head, 0: 8]}")
        print(f"{k_new[0, print_head, 72: 80]}")

    q = q.reshape(num_heads, head_dim)
    k = torch.cat((kv_cache[0], k_new), dim=0) 
    v = torch.cat((kv_cache[1], v_new), dim=0)
    
    o = flashinfer.single_decode_with_kv_cache(
        q, k, v, "NHD", "NONE", use_tensor_cores=False
    )
    
    if debug:
        print("attn output O")
        print(f"o, head_id = {print_head}, o")
        print(f"{o[print_head, 0: 80]}")
    
    o = o_proj(o.view(1, num_heads * head_dim))
    
    if debug:
        print("final output o")
        print(o[0, 0:8])
        print(o[0, 2552:2560])
        print("-----------------------------  python end  -----------------------------")
    
    return o.detach()

def generate_random_weights(shape):
    """Generate random weights scaled appropriately"""
    return (torch.randn(shape) * 0.1).to(0).half()

def test_pythia_decode_correctness():
    """Test Pythia decoder layer correctness against reference implementation"""
    print(f"Testing Pythia-2.8b decoder layer")
    print(f"hidden_size: {hidden_size}, num_heads: {num_heads}, head_dim: {head_dim}")
    print(f"seqlen: {seqlen}, rotary_dim: {rotary_dim}")
    
    # Generate random weights and inputs
    input_tensor = generate_random_weights((1, hidden_size)).to(0).half()
    residual = generate_random_weights((1, hidden_size)).to(0).half()
    weight_qkv = generate_random_weights((3 * num_heads * head_dim, hidden_size)).to(0).half()
    weight_o = generate_random_weights((num_heads * head_dim, hidden_size)).to(0).half()
    
    layernorm_weight = generate_random_weights((1, hidden_size)).to(0).half()
    layernorm_bias = torch.zeros((1, hidden_size)).to(0).half()

    # Generate full kv_cache with shape (2, seqlen, num_heads * head_dim)
    kv_cache_full = generate_random_weights((2, seqlen, num_heads * head_dim)).to(0).half()

    # RoPE with cos and sin (only for rotary_dim, not full head_dim)
    cos, sin = initialize_rope_embeddings(rotary_dim)
    
    # Our ClusterFusion kernel
    print("\n=== Running ClusterFusion Pythia Kernel ===")
    o = []
    for i in range(test_run):
        output, k, v = clusterfusion.pythia_decoder_layer(
            input_tensor,          
            weight_qkv,                          
            weight_o,              
            kv_cache_full[0],
            kv_cache_full[1],           
            layernorm_weight,
            cos,                   
            sin                    
        )
        o.append(output)
        if i == 0:
            print(f"First run output shape: {output.shape}")
            print(f"First run k shape: {k.shape}, v shape: {v.shape}")

    eps = 1e-5
    layernorm_weight_flat = layernorm_weight.reshape((hidden_size,))

    # Initialize reference linear layers with the same weights
    qkv_proj = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=False).to(0).half()
    o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False).to(0).half()
    qkv_proj.weight.data = weight_qkv.contiguous().view(qkv_proj.weight.data.shape)
    o_proj.weight.data = weight_o.contiguous().view(o_proj.weight.data.shape)

    # Split kv_cache_full for reference implementation
    kv_cache_k = kv_cache_full[0].view(seqlen, num_heads, head_dim)
    kv_cache_v = kv_cache_full[1].view(seqlen, num_heads, head_dim)
    kv_cache_gt = torch.cat([kv_cache_k[:seqlen], kv_cache_v[:seqlen]], dim=0).view(2, seqlen, num_heads, head_dim)
    
    # Reference implementation
    print("\n=== Running Reference Implementation ===")
    nvtx.range_push("pythia_decode")
    o_gt = pythia_decode(
        input_tensor, residual, layernorm_weight_flat, layernorm_bias, eps, 
        kv_cache_gt, qkv_proj, o_proj, head_dim, cos, sin, rotary_dim
    )
    nvtx.range_pop()
    
    print(f"\nReference output shape: {o_gt.shape}")
    print(f"Reference output abs mean: {o_gt.abs().mean().item()}")
    
    # Compare outputs
    print("\n=== Comparison ===")
    print("Ours[..., 0:80]:", o[0][..., 0:80])
    print("Ref[..., 0:80]:", o_gt[..., 0:80])
    
    max_error_list = []
    min_error_list = []
    mse_list = []
    mae_list = []
    
    for i in range(test_run):
        diff = (o[i] - o_gt).abs()
        mae = diff.mean()
        mae_list.append(mae)

        mse = (diff ** 2).mean()
        mse_list.append(mse)

        max_error = diff.max()
        max_error_list.append(max_error)

    print(f"\n=== Error Statistics over {test_run} runs ===")
    print(f"Max MSE: {max(mse_list).item():.6f}")
    print(f"Min MSE: {min(mse_list).item():.6f}")
    print(f"Max MAE: {max(mae_list).item():.6f}")
    print(f"Min MAE: {min(mae_list).item():.6f}")
    print(f"Max absolute error: {max(max_error_list).item():.6f}")
    print(f"Min absolute error: {min(max_error_list).item():.6f}")
    print(f"Count of max errors > 0.1: {sum(e.item() > 0.1 for e in max_error_list)}")

    max_error_value = max(max_error_list).item()
    max_error_index = max_error_list.index(max(max_error_list))
    print(f"Maximum error occurs at run {max_error_index}, value: {max_error_value:.6f}")
    
    # Pass/Fail criteria
    avg_mae = sum(mae_list).item() / len(mae_list)
    print(f"\nAverage MAE: {avg_mae:.6f}")
    if avg_mae < 0.01:
        print("✓ TEST PASSED: Average error is acceptable")
    else:
        print("✗ TEST FAILED: Average error is too large")

def test_pythia_decode_small():
    """Quick test with smaller sequence length for debugging"""
    global seqlen, test_run
    seqlen = 128
    test_run = 10
    
    print("\n" + "="*80)
    print("Quick test with seqlen=128")
    print("="*80)
    test_pythia_decode_correctness()

if __name__ == "__main__":
    # Run quick test first
    test_pythia_decode_small()
    
    # Run full test
    seqlen = 2048
    test_run = 100
    print("\n" + "="*80)
    print("Full test with seqlen=2048")
    print("="*80)
    test_pythia_decode_correctness()
