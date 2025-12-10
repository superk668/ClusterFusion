"""
Test single layer with REAL model data (not random)
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import clusterfusion

MODEL_NAME = "EleutherAI/pythia-2.8b"

def apply_neox_style_rotary_pos_emb_partial(q, k, cos, sin, rotary_dim):
    """Apply Neox-style RoPE only to first rotary_dim dimensions"""
    cos = cos[:, :rotary_dim].unsqueeze(1)
    sin = sin[:, :rotary_dim].unsqueeze(1)
    
    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]
    
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    q_rot_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    
    q_embed = torch.cat([q_rot_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_rot_embed, k_pass], dim=-1)
    
    return q_embed.to(q.dtype), k_embed.to(k.dtype)

def pythia_decode_reference(hidden, ln_weight, ln_bias, eps, kv_cache, 
                            weight_qkv, bias_qkv, weight_o, 
                            head_dim, cos, sin, rotary_dim, num_heads=32):
    """Reference implementation matching test_pythia.py"""
    # LayerNorm
    mean = hidden.mean(dim=-1, keepdim=True)
    var = hidden.var(dim=-1, keepdim=True, unbiased=False)
    hidden_normed = (hidden - mean) / torch.sqrt(var + eps) * ln_weight + ln_bias
    
    # QKV projection
    qkv = torch.matmul(hidden_normed, weight_qkv.t()) + bias_qkv
    qkv = qkv.view(num_heads, 3, head_dim)
    q = qkv[:, 0, :].unsqueeze(0)
    k_new = qkv[:, 1, :].unsqueeze(0)
    v_new = qkv[:, 2, :].unsqueeze(0)
    
    # Apply RoPE
    q, k_new = apply_neox_style_rotary_pos_emb_partial(q, k_new, cos, sin, rotary_dim)
    
    q = q.reshape(num_heads, head_dim)
    k = torch.cat((kv_cache[0], k_new), dim=0)
    v = torch.cat((kv_cache[1], v_new), dim=0)
    
    # Attention
    q = q.unsqueeze(0).unsqueeze(2)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)
    
    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
    attn_weights = F.softmax(scores, dim=-1)
    o = torch.matmul(attn_weights, v)
    o = o.squeeze(0).squeeze(1)
    
    # Output projection (NO BIAS like test_pythia.py)
    o = torch.matmul(o.view(1, num_heads * head_dim), weight_o.t())
    
    return o.detach()

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map='cuda:0')
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Get real input
prompt = "The meaning of life is"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda:0')

# Prefill
with torch.no_grad():
    outputs = model(input_ids, use_cache=True)
    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]

next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
print(f"Testing with token: {next_token.item()}")

# Get first layer weights
layer = model.gpt_neox.layers[0]
ln_weight = layer.input_layernorm.weight.data.unsqueeze(0).half()
ln_bias = layer.input_layernorm.bias.data.unsqueeze(0).half()
weight_qkv = layer.attention.query_key_value.weight.data.half()
bias_qkv = layer.attention.query_key_value.bias.data.half()
weight_o = layer.attention.dense.weight.data.half()

# Get REAL cache from prefill
k = past_key_values[0][0].squeeze(0).transpose(0, 1).contiguous()  # [seq, heads, dim]
v = past_key_values[0][1].squeeze(0).transpose(0, 1).contiguous()
k_cache = k.reshape(k.shape[0], -1)  # [seq, heads*dim]
v_cache = v.reshape(v.shape[0], -1)

print(f"\nCache info:")
print(f"  k_cache shape: {k_cache.shape}")
print(f"  k_cache norm: {k_cache.norm().item():.4f}")
print(f"  First position norm: {k_cache[0].norm().item():.4f}")

# Preallocate for kernel
max_seq_len = 100
k_cache_full = torch.zeros((max_seq_len, 2560), dtype=torch.float16, device='cuda:0')
v_cache_full = torch.zeros((max_seq_len, 2560), dtype=torch.float16, device='cuda:0')
k_cache_full[:k_cache.shape[0]] = k_cache
v_cache_full[:v_cache.shape[0]] = v_cache

# Get REAL RoPE for position 5
position = 5
inv_freq = 1.0 / (10000 ** (torch.arange(0, 20, 2, dtype=torch.float32, device='cuda:0') / 20))
position_tensor = torch.tensor([position], dtype=torch.float32, device='cuda:0')
freqs = torch.outer(position_tensor, inv_freq)
emb = torch.cat([freqs, freqs], dim=-1)
cos = emb.cos()
sin = emb.sin()

print(f"\nRoPE info:")
print(f"  cos shape: {cos.shape}")
print(f"  cos first 8: {cos[0, :8]}")

# Get input
hidden_states = model.gpt_neox.embed_in(next_token).half().squeeze(1)
print(f"\nInput norm: {hidden_states.norm().item():.4f}")

# Reference implementation
eps = 1e-5
ln_weight_flat = ln_weight.reshape((2560,))
ln_bias_flat = ln_bias.reshape((2560,))

# Prepare cache for reference: [2, seqlen, num_heads, head_dim]
kv_cache_k = k_cache.view(k_cache.shape[0], 32, 80)
kv_cache_v = v_cache.view(v_cache.shape[0], 32, 80)
kv_cache_ref = torch.stack([kv_cache_k, kv_cache_v], dim=0)

with torch.no_grad():
    o_ref = pythia_decode_reference(
        hidden_states.clone(), ln_weight_flat, ln_bias_flat, eps,
        kv_cache_ref, weight_qkv, bias_qkv, weight_o,
        80, cos, sin, 20
    )

print(f"\n✓ Reference output:")
print(f"  norm: {o_ref.norm().item():.4f}")
print(f"  first 8: {o_ref[0, :8]}")

# Kernel
with torch.no_grad():
    o_kernel, _, _ = clusterfusion.pythia_decoder_layer(
        hidden_states.clone(),
        weight_qkv,
        bias_qkv,
        weight_o,
        layer.attention.dense.bias.data.half(),
        k_cache_full,
        v_cache_full,
        ln_weight,
        ln_bias,
        cos,
        sin,
        k_cache.shape[0]
    )

print(f"\n✓ Kernel output:")
print(f"  norm: {o_kernel.norm().item():.4f}")
print(f"  first 8: {o_kernel[0, :8]}")

# Compare
diff = (o_ref - o_kernel).abs()
print(f"\nDifference:")
print(f"  Max error: {diff.max().item():.4f}")
print(f"  Mean error: {diff.mean().item():.4f}")
print(f"  Relative error: {(diff.mean() / o_ref.abs().mean()).item():.4f}")

if diff.mean().item() < 0.01:
    print(f"\n✓ TEST PASSED with real data!")
else:
    print(f"\n✗ TEST FAILED with real data!")
    print(f"  This explains why end-to-end generation fails!")

