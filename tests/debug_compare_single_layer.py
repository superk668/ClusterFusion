"""
Compare kernel vs PyTorch for a single layer with real model weights
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import clusterfusion

MODEL_NAME = "EleutherAI/pythia-2.8b"

def apply_rotary_pos_emb(q, k, cos, sin, rotary_dim):
    """Apply Neox-style RoPE"""
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    
    return q_embed.to(q.dtype), k_embed.to(k.dtype)

def pytorch_attn_only(hidden_states, ln_weight, ln_bias, qkv_weight, qkv_bias, o_weight, 
                       k_cache, v_cache, cos, sin, num_heads=32, head_dim=80, rotary_dim=20):
    """PyTorch attention only (for comparison)"""
    hidden_size = hidden_states.shape[-1]
    
    # LayerNorm
    attn_input = F.layer_norm(hidden_states, (hidden_size,), ln_weight.squeeze(0), ln_bias.squeeze(0), eps=1e-5)
    
    # QKV projection
    qkv = F.linear(attn_input, qkv_weight, qkv_bias)
    qkv = qkv.view(1, num_heads, 3, head_dim)
    q = qkv[:, :, 0, :]
    k_new = qkv[:, :, 1, :]
    v_new = qkv[:, :, 2, :]
    
    # RoPE
    cos_expanded = cos.unsqueeze(1)
    sin_expanded = sin.unsqueeze(1)
    q, k_new = apply_rotary_pos_emb(q, k_new, cos_expanded, sin_expanded, rotary_dim)
    
    # Concat cache
    seq_len = k_cache.shape[0]
    k_cache_shaped = k_cache.view(seq_len, num_heads, head_dim)
    v_cache_shaped = v_cache.view(seq_len, num_heads, head_dim)
    k = torch.cat([k_cache_shaped, k_new.squeeze(0).unsqueeze(0)], dim=0)
    v = torch.cat([v_cache_shaped, v_new.squeeze(0).unsqueeze(0)], dim=0)
    
    # Attention
    q_attn = q.unsqueeze(2)
    k_attn = k.transpose(0, 1).unsqueeze(0)
    v_attn = v.transpose(0, 1).unsqueeze(0)
    
    attn_scores = torch.matmul(q_attn.float(), k_attn.float().transpose(-2, -1)) / (head_dim ** 0.5)
    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_weights, v_attn.float()).half()
    attn_output = attn_output.squeeze(2)
    
    # Output projection (NO BIAS in kernel)
    attn_output = attn_output.reshape(1, num_heads * head_dim)
    attn_output_no_bias = F.linear(attn_output, o_weight, None)  # No bias
    
    return attn_output_no_bias, k_new.squeeze(0), v_new.squeeze(0)

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
weights = {
    'ln_weight': layer.input_layernorm.weight.data.unsqueeze(0).half(),
    'ln_bias': layer.input_layernorm.bias.data.unsqueeze(0).half(),
    'qkv_weight': layer.attention.query_key_value.weight.data.half(),
    'qkv_bias': layer.attention.query_key_value.bias.data.half(),
    'o_weight': layer.attention.dense.weight.data.half(),
    'o_bias': layer.attention.dense.bias.data.half(),
}

# Get cache
k = past_key_values[0][0].squeeze(0).transpose(0, 1).contiguous()
v = past_key_values[0][1].squeeze(0).transpose(0, 1).contiguous()
k_cache = k.reshape(k.shape[0], -1)
v_cache = v.reshape(v.shape[0], -1)

# Preallocate
max_seq_len = 100
k_cache_full = torch.zeros((max_seq_len, 2560), dtype=torch.float16, device='cuda:0')
v_cache_full = torch.zeros((max_seq_len, 2560), dtype=torch.float16, device='cuda:0')
k_cache_full[:k_cache.shape[0]] = k_cache
v_cache_full[:v_cache.shape[0]] = v_cache

# Get RoPE for position 5
position = 5
inv_freq = 1.0 / (10000 ** (torch.arange(0, 20, 2, dtype=torch.float32, device='cuda:0') / 20))
position_tensor = torch.tensor([position], dtype=torch.float32, device='cuda:0')
freqs = torch.outer(position_tensor, inv_freq)
emb = torch.cat([freqs, freqs], dim=-1)
cos = emb.cos()
sin = emb.sin()

# Get input
hidden_states = model.gpt_neox.embed_in(next_token).half().squeeze(1)
print(f"Input norm: {hidden_states.norm().item():.4f}")

# PyTorch
with torch.no_grad():
    attn_pytorch, _, _ = pytorch_attn_only(
        hidden_states.clone(),
        weights['ln_weight'],
        weights['ln_bias'],
        weights['qkv_weight'],
        weights['qkv_bias'],
        weights['o_weight'],
        k_cache,
        v_cache,
        cos,
        sin
    )
    
print(f"PyTorch attn output norm (no bias): {attn_pytorch.norm().item():.4f}")
print(f"PyTorch first 8 values: {attn_pytorch[0, :8]}")

# Kernel
with torch.no_grad():
    attn_kernel, _, _ = clusterfusion.pythia_decoder_layer(
        hidden_states.clone(),
        weights['qkv_weight'],
        weights['qkv_bias'],
        weights['o_weight'],
        weights['o_bias'],
        k_cache_full,
        v_cache_full,
        weights['ln_weight'],
        weights['ln_bias'],
        cos,
        sin,
        k_cache.shape[0]
    )

print(f"Kernel attn output norm: {attn_kernel.norm().item():.4f}")
print(f"Kernel first 8 values: {attn_kernel[0, :8]}")

# Compare
diff = (attn_pytorch - attn_kernel).abs()
print(f"\nDifference:")
print(f"  Max error: {diff.max().item():.4f}")
print(f"  Mean error: {diff.mean().item():.4f}")
print(f"  Relative error: {(diff.mean() / attn_pytorch.abs().mean()).item():.4f}")

