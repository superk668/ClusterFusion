"""
Debug attention output before output projection
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import clusterfusion

MODEL_NAME = "EleutherAI/pythia-2.8b"

def apply_rotary_pos_emb(q, k, cos, sin, rotary_dim):
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

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map='cuda:0')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

prompt = "The meaning of life is"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda:0')

with torch.no_grad():
    outputs = model(input_ids, use_cache=True)
    past_key_values = outputs.past_key_values

next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

# Get data
layer = model.gpt_neox.layers[0]
hidden_states = model.gpt_neox.embed_in(next_token).half().squeeze(1)

# LayerNorm + QKV
ln_weight = layer.input_layernorm.weight.data
ln_bias = layer.input_layernorm.bias.data
mean = hidden_states.mean(dim=-1, keepdim=True)
var = hidden_states.var(dim=-1, keepdim=True, unbiased=False)
hidden_normed = (hidden_states - mean) / torch.sqrt(var + 1e-5) * ln_weight + ln_bias

weight_qkv = layer.attention.query_key_value.weight.data.half()
bias_qkv = layer.attention.query_key_value.bias.data.half()
qkv = torch.matmul(hidden_normed, weight_qkv.t()) + bias_qkv
qkv = qkv.view(32, 3, 80)
q = qkv[:, 0, :].unsqueeze(0)
k_new = qkv[:, 1, :].unsqueeze(0)
v_new = qkv[:, 2, :].unsqueeze(0)

# RoPE
position = 5
inv_freq = 1.0 / (10000 ** (torch.arange(0, 20, 2, dtype=torch.float32, device='cuda:0') / 20))
position_tensor = torch.tensor([position], dtype=torch.float32, device='cuda:0')
freqs = torch.outer(position_tensor, inv_freq)
emb = torch.cat([freqs, freqs], dim=-1)
cos = emb.cos().unsqueeze(1)
sin = emb.sin().unsqueeze(1)

q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin, 20)

# Cache
k = past_key_values[0][0].squeeze(0).transpose(0, 1).contiguous()
v = past_key_values[0][1].squeeze(0).transpose(0, 1).contiguous()
k_cache = k.reshape(k.shape[0], -1)
v_cache = v.reshape(v.shape[0], -1)

k_cache_shaped = k_cache.view(5, 32, 80)
v_cache_shaped = v_cache.view(5, 32, 80)
k_full = torch.cat([k_cache_shaped, k_new.squeeze(0).unsqueeze(0)], dim=0)
v_full = torch.cat([v_cache_shaped, v_new.squeeze(0).unsqueeze(0)], dim=0)

# Attention
q_attn = q.unsqueeze(2)  # [1, 32, 1, 80]
k_attn = k_full.transpose(0, 1).unsqueeze(0)  # [1, 32, 6, 80]
v_attn = v_full.transpose(0, 1).unsqueeze(0)

attn_scores = torch.matmul(q_attn.float(), k_attn.float().transpose(-2, -1)) / (80 ** 0.5)
attn_weights = F.softmax(attn_scores, dim=-1)
attn_output = torch.matmul(attn_weights, v_attn.float()).half()
attn_output = attn_output.squeeze(2)  # [1, 32, 80]

print(f"\n=== Reference attention output (before output proj) ===")
print(f"attn_output norm: {attn_output.norm().item():.4f}")
print(f"attn_output range: [{attn_output.min().item():.4f}, {attn_output.max().item():.4f}]")
print(f"attn_output[0, 0, :8]: {attn_output[0, 0, :8]}")

# Now check what kernel produces
# We need to extract attention output BEFORE output projection
# But kernel fuses everything... Let me check the actual final output

max_seq_len = 100
k_cache_full = torch.zeros((max_seq_len, 2560), dtype=torch.float16, device='cuda:0')
v_cache_full = torch.zeros((max_seq_len, 2560), dtype=torch.float16, device='cuda:0')
k_cache_full[:k_cache.shape[0]] = k_cache
v_cache_full[:v_cache.shape[0]] = v_cache

ln_weight_kernel = layer.input_layernorm.weight.data.unsqueeze(0).half()
ln_bias_kernel = layer.input_layernorm.bias.data.unsqueeze(0).half()
weight_o = layer.attention.dense.weight.data.half()

cos_kernel = emb.cos()
sin_kernel = emb.sin()

with torch.no_grad():
    output_kernel, _, _ = clusterfusion.pythia_decoder_layer(
        hidden_states.clone(),
        weight_qkv,
        bias_qkv,
        weight_o,
        layer.attention.dense.bias.data.half(),
        k_cache_full,
        v_cache_full,
        ln_weight_kernel,
        ln_bias_kernel,
        cos_kernel,
        sin_kernel,
        k_cache.shape[0]
    )

# Reference output projection
attn_flat = attn_output.reshape(1, 32 * 80)
output_ref = F.linear(attn_flat, weight_o, None)

print(f"\n=== After output projection ===")
print(f"Reference norm: {output_ref.norm().item():.4f}")
print(f"Kernel norm: {output_kernel.norm().item():.4f}")
print(f"Ratio: {output_kernel.norm().item() / output_ref.norm().item():.2f}x")

diff = (output_ref - output_kernel).abs()
print(f"\nDifference:")
print(f"  Max: {diff.max().item():.4f}")
print(f"  Mean: {diff.mean().item():.4f}")

