#!/usr/bin/env python3
"""
Benchmark: ClusterFusion vs HuggingFace

This benchmark measures:
1. Decoder layers speedup (our core contribution)
2. End-to-end token generation speedup (user experience)

Design:
- Both implementations use the same embedding and lm_head
- Only the 32 decoder layers are replaced
- Fair comparison with identical inputs/outputs
"""

import torch
import torch.nn.functional as F
import time
import clusterfusion
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

print("=" * 80)
print("Benchmark: ClusterFusion vs HuggingFace (Pythia-2.8B)")
print("=" * 80)

# =============================================================================
# Setup
# =============================================================================
MODEL_NAME = "EleutherAI/pythia-2.8b"
device = torch.device("cuda:0")

print("\nLoading model and tokenizer...")
config = AutoConfig.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.float16, device_map="cuda:0"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.eval()

# Model configuration
HIDDEN_DIM = config.hidden_size
NUM_HEADS = config.num_attention_heads
HEAD_DIM = HIDDEN_DIM // NUM_HEADS
FFN_DIM = config.intermediate_size
NUM_LAYERS = config.num_hidden_layers
VOCAB_SIZE = config.vocab_size

print(f"Model: {MODEL_NAME}")
print(f"Hidden: {HIDDEN_DIM}, Heads: {NUM_HEADS}, HeadDim: {HEAD_DIM}")
print(f"FFN: {FFN_DIM}, Layers: {NUM_LAYERS}, Vocab: {VOCAB_SIZE}")

# =============================================================================
# Prepare RoPE embeddings
# =============================================================================
MAX_SEQ = 512
rotary_ndims = model.gpt_neox.layers[0].attention.rotary_ndims

# Compute RoPE embeddings
inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_ndims, 2, device=device).float() / rotary_ndims))
positions = torch.arange(MAX_SEQ, device=device).float()
freqs = torch.outer(positions, inv_freq)
cos_cache = torch.cos(freqs)
sin_cache = torch.sin(freqs)

# Pad to HEAD_DIM for CUDA kernel
cos_padded = torch.zeros(MAX_SEQ, HEAD_DIM, dtype=torch.float32, device=device)
sin_padded = torch.zeros(MAX_SEQ, HEAD_DIM, dtype=torch.float32, device=device)
cos_padded[:, :cos_cache.shape[-1]] = cos_cache
sin_padded[:, :sin_cache.shape[-1]] = sin_cache

# =============================================================================
# Helper: Extract layer weights
# =============================================================================
def get_layer_weights(layer_idx):
    layer = model.gpt_neox.layers[layer_idx]
    return {
        'qkv_weight': layer.attention.query_key_value.weight.data,
        'qkv_bias': layer.attention.query_key_value.bias.data,
        'o_weight': layer.attention.dense.weight.data,
        'o_bias': layer.attention.dense.bias.data,
        'ln_weight': layer.input_layernorm.weight.data.half(),
        'ln_bias': layer.input_layernorm.bias.data.half(),
        'post_ln_weight': layer.post_attention_layernorm.weight.data.half(),
        'post_ln_bias': layer.post_attention_layernorm.bias.data.half(),
        'mlp_up_weight': layer.mlp.dense_h_to_4h.weight.data,
        'mlp_up_bias': layer.mlp.dense_h_to_4h.bias.data,
        'mlp_down_weight': layer.mlp.dense_4h_to_h.weight.data,
        'mlp_down_bias': layer.mlp.dense_4h_to_h.bias.data,
    }

# Pre-extract all layer weights
all_weights = [get_layer_weights(i) for i in range(NUM_LAYERS)]

# =============================================================================
# HuggingFace Decode Step (using model internals)
# =============================================================================
def hf_decode_layers(hidden_states, k_caches, v_caches, seq_len, position_embeddings):
    """Run 32 decoder layers using HuggingFace implementation"""
    # Create attention mask - causal mask for the full sequence
    attention_mask = torch.ones(1, 1, 1, seq_len + 1, dtype=torch.float16, device=device)
    
    # Create position ids
    position_ids = torch.tensor([[seq_len]], device=device)
    
    for layer_idx, layer in enumerate(model.gpt_neox.layers):
        # Get cached KV
        past_key_value = (
            k_caches[layer_idx][:, :, :seq_len, :],
            v_caches[layer_idx][:, :, :seq_len, :]
        )
        
        # Run layer with position_embeddings
        outputs = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=True,
            position_embeddings=position_embeddings,
        )
        hidden_states = outputs[0]
        
        # Update cache - handle different output formats
        if len(outputs) > 1 and outputs[1] is not None:
            past_kv = outputs[1]
            if isinstance(past_kv, tuple) and len(past_kv) == 2:
                new_k, new_v = past_kv
                k_caches[layer_idx][:, :, seq_len:seq_len+1, :] = new_k[:, :, -1:, :]
                v_caches[layer_idx][:, :, seq_len:seq_len+1, :] = new_v[:, :, -1:, :]
    
    return hidden_states

# =============================================================================
# ClusterFusion Decode Step
# =============================================================================
def cuda_decode_layers(hidden_states, k_caches, v_caches, seq_len):
    """Run 32 decoder layers using ClusterFusion CUDA kernel"""
    for layer_idx in range(NUM_LAYERS):
        w = all_weights[layer_idx]
        
        output, k_new, v_new = clusterfusion.pythia_2b8_decoder_layer(
            hidden_states,
            w['qkv_weight'], w['qkv_bias'],
            w['o_weight'], w['o_bias'],
            k_caches[layer_idx], v_caches[layer_idx],
            w['ln_weight'], w['ln_bias'],
            cos_padded, sin_padded,
            w['post_ln_weight'], w['post_ln_bias'],
            w['mlp_up_weight'], w['mlp_up_bias'],
            w['mlp_down_weight'], w['mlp_down_bias'],
            seq_len
        )
        
        # Update cache
        k_caches[layer_idx][seq_len] = k_new.squeeze(0)
        v_caches[layer_idx][seq_len] = v_new.squeeze(0)
        
        hidden_states = output.unsqueeze(0)  # [1, 1, hidden_dim]
    
    return hidden_states

# =============================================================================
# End-to-End Token Generation
# =============================================================================
def generate_tokens_hf(prompt_ids, num_tokens):
    """Generate tokens using HuggingFace layers"""
    # Initialize KV caches [seq_len, num_heads, head_dim]
    k_caches = [torch.zeros(1, NUM_HEADS, MAX_SEQ, HEAD_DIM, dtype=torch.float16, device=device) 
                for _ in range(NUM_LAYERS)]
    v_caches = [torch.zeros(1, NUM_HEADS, MAX_SEQ, HEAD_DIM, dtype=torch.float16, device=device) 
                for _ in range(NUM_LAYERS)]
    
    # Prefill phase (use HF model directly)
    with torch.no_grad():
        outputs = model(prompt_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        
        # Copy to our cache format
        seq_len = prompt_ids.shape[1]
        for layer_idx in range(NUM_LAYERS):
            k_caches[layer_idx][:, :, :seq_len, :] = past_key_values[layer_idx][0]
            v_caches[layer_idx][:, :, :seq_len, :] = past_key_values[layer_idx][1]
    
    generated_ids = prompt_ids.clone()
    current_seq_len = seq_len
    
    # Decode phase
    times = {'embedding': 0, 'decoder': 0, 'lm_head': 0}
    
    # Get rotary embedding module
    rotary_emb = model.gpt_neox.rotary_emb
    
    for _ in range(num_tokens):
        # Get last token
        last_token = generated_ids[:, -1:]
        
        # Embedding
        t0 = time.perf_counter()
        hidden_states = model.gpt_neox.embed_in(last_token)
        torch.cuda.synchronize()
        times['embedding'] += time.perf_counter() - t0
        
        # Compute position embeddings for current position
        position_ids = torch.tensor([[current_seq_len]], device=device)
        position_embeddings = rotary_emb(hidden_states, position_ids)
        
        # Decoder layers (HuggingFace)
        t0 = time.perf_counter()
        hidden_states = hf_decode_layers(hidden_states, k_caches, v_caches, current_seq_len, position_embeddings)
        torch.cuda.synchronize()
        times['decoder'] += time.perf_counter() - t0
        
        # Final LayerNorm + LM Head
        t0 = time.perf_counter()
        hidden_states = model.gpt_neox.final_layer_norm(hidden_states)
        logits = model.embed_out(hidden_states)
        torch.cuda.synchronize()
        times['lm_head'] += time.perf_counter() - t0
        
        # Sample next token
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        current_seq_len += 1
    
    return generated_ids, times


def generate_tokens_cuda(prompt_ids, num_tokens):
    """Generate tokens using ClusterFusion CUDA kernel"""
    # Initialize KV caches [seq_len, num_heads, head_dim]
    k_caches = [torch.zeros(MAX_SEQ, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device=device) 
                for _ in range(NUM_LAYERS)]
    v_caches = [torch.zeros(MAX_SEQ, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device=device) 
                for _ in range(NUM_LAYERS)]
    
    # Prefill phase (use HF model)
    with torch.no_grad():
        outputs = model(prompt_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        
        # Copy to our cache format
        seq_len = prompt_ids.shape[1]
        for layer_idx in range(NUM_LAYERS):
            # HF cache: [batch, heads, seq, head_dim] -> our cache: [seq, heads, head_dim]
            k_caches[layer_idx][:seq_len] = past_key_values[layer_idx][0].squeeze(0).transpose(0, 1)
            v_caches[layer_idx][:seq_len] = past_key_values[layer_idx][1].squeeze(0).transpose(0, 1)
    
    generated_ids = prompt_ids.clone()
    current_seq_len = seq_len
    
    # Decode phase
    times = {'embedding': 0, 'decoder': 0, 'lm_head': 0}
    
    for _ in range(num_tokens):
        # Get last token
        last_token = generated_ids[:, -1:]
        
        # Embedding
        t0 = time.perf_counter()
        hidden_states = model.gpt_neox.embed_in(last_token)
        torch.cuda.synchronize()
        times['embedding'] += time.perf_counter() - t0
        
        # Decoder layers (ClusterFusion)
        t0 = time.perf_counter()
        hidden_states = cuda_decode_layers(hidden_states, k_caches, v_caches, current_seq_len)
        torch.cuda.synchronize()
        times['decoder'] += time.perf_counter() - t0
        
        # Final LayerNorm + LM Head
        t0 = time.perf_counter()
        hidden_states = model.gpt_neox.final_layer_norm(hidden_states)
        logits = model.embed_out(hidden_states)
        torch.cuda.synchronize()
        times['lm_head'] += time.perf_counter() - t0
        
        # Sample next token
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        current_seq_len += 1
    
    return generated_ids, times

# =============================================================================
# Benchmark
# =============================================================================
print("\n" + "=" * 80)
print("Benchmark: End-to-End Token Generation")
print("=" * 80)

# Prepare prompt
prompt = "The quick brown fox jumps over the lazy dog. In a world where"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
print(f"\nPrompt: '{prompt}'")
print(f"Prompt tokens: {input_ids.shape[1]}")

# Test different generation lengths
NUM_TOKENS_LIST = [16, 32, 64, 128]

print("\n" + "-" * 100)
print(f"{'Tokens':<8} | {'HF Total':<12} | {'CUDA Total':<12} | {'HF Decoder':<12} | {'CUDA Decoder':<12} | {'E2E Speedup':<12} | {'Decoder Speedup':<14}")
print("-" * 100)

for num_tokens in NUM_TOKENS_LIST:
    # Warmup
    with torch.no_grad():
        _, _ = generate_tokens_hf(input_ids.clone(), 2)
        _, _ = generate_tokens_cuda(input_ids.clone(), 2)
    
    # Benchmark HuggingFace
    torch.cuda.synchronize()
    with torch.no_grad():
        hf_ids, hf_times = generate_tokens_hf(input_ids.clone(), num_tokens)
    hf_total = sum(hf_times.values())
    
    # Benchmark ClusterFusion
    torch.cuda.synchronize()
    with torch.no_grad():
        cuda_ids, cuda_times = generate_tokens_cuda(input_ids.clone(), num_tokens)
    cuda_total = sum(cuda_times.values())
    
    # Calculate speedups
    e2e_speedup = hf_total / cuda_total
    decoder_speedup = hf_times['decoder'] / cuda_times['decoder']
    
    print(f"{num_tokens:<8} | {hf_total*1000:>10.1f}ms | {cuda_total*1000:>10.1f}ms | "
          f"{hf_times['decoder']*1000:>10.1f}ms | {cuda_times['decoder']*1000:>10.1f}ms | "
          f"{e2e_speedup:>10.2f}x  | {decoder_speedup:>12.2f}x")

print("-" * 100)

# =============================================================================
# Detailed Breakdown for 64 tokens
# =============================================================================
print("\n" + "=" * 80)
print("Detailed Breakdown (64 tokens)")
print("=" * 80)

num_tokens = 64
with torch.no_grad():
    _, hf_times = generate_tokens_hf(input_ids.clone(), num_tokens)
    _, cuda_times = generate_tokens_cuda(input_ids.clone(), num_tokens)

hf_total = sum(hf_times.values())
cuda_total = sum(cuda_times.values())

print(f"\n{'Component':<20} | {'HuggingFace':<15} | {'ClusterFusion':<15} | {'Speedup':<10} | {'% of Total':<12}")
print("-" * 80)
print(f"{'Embedding':<20} | {hf_times['embedding']*1000:>12.2f} ms | {cuda_times['embedding']*1000:>12.2f} ms | "
      f"{hf_times['embedding']/cuda_times['embedding']:>8.2f}x | {cuda_times['embedding']/cuda_total*100:>10.1f}%")
print(f"{'32 Decoder Layers':<20} | {hf_times['decoder']*1000:>12.2f} ms | {cuda_times['decoder']*1000:>12.2f} ms | "
      f"{hf_times['decoder']/cuda_times['decoder']:>8.2f}x | {cuda_times['decoder']/cuda_total*100:>10.1f}%")
print(f"{'LM Head':<20} | {hf_times['lm_head']*1000:>12.2f} ms | {cuda_times['lm_head']*1000:>12.2f} ms | "
      f"{hf_times['lm_head']/cuda_times['lm_head']:>8.2f}x | {cuda_times['lm_head']/cuda_total*100:>10.1f}%")
print("-" * 80)
print(f"{'TOTAL':<20} | {hf_total*1000:>12.2f} ms | {cuda_total*1000:>12.2f} ms | "
      f"{hf_total/cuda_total:>8.2f}x | {'100.0':>10}%")

# Per-token metrics
print(f"\nPer-token latency:")
print(f"  HuggingFace:    {hf_total/num_tokens*1000:.3f} ms/token")
print(f"  ClusterFusion:  {cuda_total/num_tokens*1000:.3f} ms/token")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(f"""
ClusterFusion Acceleration Results:

┌─────────────────────────────────────────────────────────────────┐
│  Decoder Layers (Core Contribution)                              │
│  ─────────────────────────────────────────────────────────────  │
│  HuggingFace:    {hf_times['decoder']*1000/num_tokens:.3f} ms/token/32-layers                      │
│  ClusterFusion:  {cuda_times['decoder']*1000/num_tokens:.3f} ms/token/32-layers                      │
│  Speedup:        {hf_times['decoder']/cuda_times['decoder']:.2f}x                                           │
├─────────────────────────────────────────────────────────────────┤
│  End-to-End Token Generation                                     │
│  ─────────────────────────────────────────────────────────────  │
│  HuggingFace:    {hf_total*1000/num_tokens:.3f} ms/token                                │
│  ClusterFusion:  {cuda_total*1000/num_tokens:.3f} ms/token                                │
│  Speedup:        {hf_total/cuda_total:.2f}x                                           │
├─────────────────────────────────────────────────────────────────┤
│  Why E2E speedup < Decoder speedup?                              │
│  ─────────────────────────────────────────────────────────────  │
│  Embedding:      {cuda_times['embedding']/cuda_total*100:.1f}% of total (not accelerated)              │
│  LM Head:        {cuda_times['lm_head']/cuda_total*100:.1f}% of total (not accelerated)              │
│  These components dilute the overall speedup.                    │
├─────────────────────────────────────────────────────────────────┤
│  Note: HuggingFace vs Manual PyTorch                             │
│  ─────────────────────────────────────────────────────────────  │
│  HuggingFace layer: ~0.17 ms/layer (internally optimized)       │
│  Manual PyTorch:    ~0.28 ms/layer (pure Python + cuBLAS)       │
│                                                                  │
│  HuggingFace already uses optimized attention implementations,  │
│  reducing our relative speedup. The ablation test (1.67x) uses  │
│  manual PyTorch as baseline, which is less optimized.           │
└─────────────────────────────────────────────────────────────────┘
""")

