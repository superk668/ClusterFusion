"""
Debug kernel vs reference step by step
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import clusterfusion

MODEL_NAME = "EleutherAI/pythia-2.8b"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map='cuda:0')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

prompt = "The meaning of life is"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda:0')

# Prefill
with torch.no_grad():
    outputs = model(input_ids, use_cache=True)
    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]

next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
print(f"Token: {next_token.item()}")

# Get layer 0
layer = model.gpt_neox.layers[0]
hidden_states = model.gpt_neox.embed_in(next_token).half().squeeze(1)

print(f"\n=== Input ===")
print(f"hidden_states norm: {hidden_states.norm().item():.4f}")
print(f"hidden_states range: [{hidden_states.min().item():.4f}, {hidden_states.max().item():.4f}]")

# Step 1: LayerNorm
ln_weight = layer.input_layernorm.weight.data
ln_bias = layer.input_layernorm.bias.data
mean = hidden_states.mean(dim=-1, keepdim=True)
var = hidden_states.var(dim=-1, keepdim=True, unbiased=False)
hidden_normed = (hidden_states - mean) / torch.sqrt(var + 1e-5) * ln_weight + ln_bias

print(f"\n=== After LayerNorm ===")
print(f"normed norm: {hidden_normed.norm().item():.4f}")
print(f"normed range: [{hidden_normed.min().item():.4f}, {hidden_normed.max().item():.4f}]")

# Step 2: QKV projection
weight_qkv = layer.attention.query_key_value.weight.data.half()
bias_qkv = layer.attention.query_key_value.bias.data.half()
qkv = torch.matmul(hidden_normed, weight_qkv.t()) + bias_qkv
qkv = qkv.view(32, 3, 80)

print(f"\n=== After QKV projection ===")
print(f"qkv norm: {qkv.norm().item():.4f}")
print(f"Q norm: {qkv[:, 0, :].norm().item():.4f}")
print(f"K norm: {qkv[:, 1, :].norm().item():.4f}")
print(f"V norm: {qkv[:, 2, :].norm().item():.4f}")

# Get V for final check (V doesn't use RoPE)
v_new = qkv[:, 2, :].unsqueeze(0)  # [1, 32, 80]

# Get cache
k = past_key_values[0][0].squeeze(0).transpose(0, 1).contiguous()
v = past_key_values[0][1].squeeze(0).transpose(0, 1).contiguous()
k_cache = k.reshape(k.shape[0], -1)
v_cache = v.reshape(v.shape[0], -1)

print(f"\n=== Cache ===")
print(f"K cache norm: {k_cache.norm().item():.2f}")
print(f"K cache range: [{k_cache.min().item():.2f}, {k_cache.max().item():.2f}]")
print(f"V cache norm: {v_cache.norm().item():.2f}")

# Prepare for kernel test
max_seq_len = 100
k_cache_full = torch.zeros((max_seq_len, 2560), dtype=torch.float16, device='cuda:0')
v_cache_full = torch.zeros((max_seq_len, 2560), dtype=torch.float16, device='cuda:0')
k_cache_full[:k_cache.shape[0]] = k_cache
v_cache_full[:v_cache.shape[0]] = v_cache

# RoPE
position = 5
inv_freq = 1.0 / (10000 ** (torch.arange(0, 20, 2, dtype=torch.float32, device='cuda:0') / 20))
position_tensor = torch.tensor([position], dtype=torch.float32, device='cuda:0')
freqs = torch.outer(position_tensor, inv_freq)
emb = torch.cat([freqs, freqs], dim=-1)
cos = emb.cos()
sin = emb.sin()

# Run kernel and get new K, V
ln_weight_kernel = layer.input_layernorm.weight.data.unsqueeze(0).half()
ln_bias_kernel = layer.input_layernorm.bias.data.unsqueeze(0).half()
weight_o = layer.attention.dense.weight.data.half()

with torch.no_grad():
    _, new_k_kernel, new_v_kernel = clusterfusion.pythia_decoder_layer(
        hidden_states.clone(),
        weight_qkv,
        bias_qkv,
        weight_o,
        layer.attention.dense.bias.data.half(),
        k_cache_full,
        v_cache_full,
        ln_weight_kernel,
        ln_bias_kernel,
        cos,
        sin,
        k_cache.shape[0]
    )

print(f"\n=== Kernel output K,V ===")
print(f"new_k_kernel norm: {new_k_kernel.norm().item():.4f}")
print(f"new_v_kernel norm: {new_v_kernel.norm().item():.4f}")
print(f"new_v_kernel first head norm: {new_v_kernel[0, :80].norm().item():.4f}")

print(f"\n=== Reference V (no RoPE) ===")
print(f"v_new norm: {v_new.norm().item():.4f}")
print(f"v_new first head norm: {v_new[0, 0, :].norm().item():.4f}")

# Compare V (should be identical since no RoPE)
diff_v = (v_new.squeeze(0).reshape(-1) - new_v_kernel.reshape(-1)).abs()
print(f"\n=== V comparison (no RoPE, should match) ===")
print(f"Max diff: {diff_v.max().item():.6f}")
print(f"Mean diff: {diff_v.mean().item():.6f}")

if diff_v.mean().item() < 0.01:
    print("✓ V matches! Problem is likely in Q/K RoPE or attention")
else:
    print("✗ V doesn't match! Problem is in QKV projection itself")

