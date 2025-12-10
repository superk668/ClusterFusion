"""
Step 2: Replace attention computation with ClusterFusion kernel
Keep everything else as PyTorch to isolate kernel correctness
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import clusterfusion
import time

MODEL_NAME = "EleutherAI/pythia-2.8b"

def compute_rope_embeddings(position, rotary_dim, head_dim, base=10000, device='cuda:0'):
    """Compute RoPE embeddings matching HuggingFace"""
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim))
    position_tensor = torch.tensor([position], dtype=torch.float32, device=device)
    freqs = torch.outer(position_tensor, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin

def generate_with_kernel(model, tokenizer, prompt, num_new_tokens=20):
    """Generate using ClusterFusion kernel for attention"""
    print(f"\\n{'='*80}")
    print(f"ClusterFusion Kernel + PyTorch MLP")
    print(f"Prompt: '{prompt}', Generating {num_new_tokens} tokens")
    print(f"{'='*80}\\n")
    
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_length = input_ids.shape[1]
    
    # Prefill with HuggingFace
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
    
    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    generated_ids = [next_token.item()]
    print(f"First token: {next_token.item()} ('{tokenizer.decode([next_token.item()])}')")
    
    # Extract weights
    num_layers = len(model.gpt_neox.layers)
    num_heads = 32
    head_dim = 80
    hidden_size = 2560
    rotary_dim = 20
    
    all_weights = []
    for layer_idx in range(num_layers):
        layer = model.gpt_neox.layers[layer_idx]
        weights = {
            'ln_weight': layer.input_layernorm.weight.data.unsqueeze(0).half(),
            'ln_bias': layer.input_layernorm.bias.data.unsqueeze(0).half(),
            'qkv_weight': layer.attention.query_key_value.weight.data.half(),
            'qkv_bias': layer.attention.query_key_value.bias.data.half(),
            'o_weight': layer.attention.dense.weight.data.half(),
            'o_bias': layer.attention.dense.bias.data.half(),
            'post_ln_weight': layer.post_attention_layernorm.weight.data.unsqueeze(0).half(),
            'post_ln_bias': layer.post_attention_layernorm.bias.data.unsqueeze(0).half(),
            'mlp_up_weight': layer.mlp.dense_h_to_4h.weight.data.half(),
            'mlp_up_bias': layer.mlp.dense_h_to_4h.bias.data.half(),
            'mlp_down_weight': layer.mlp.dense_4h_to_h.weight.data.half(),
            'mlp_down_bias': layer.mlp.dense_4h_to_h.bias.data.half(),
        }
        all_weights.append(weights)
    
    # Convert KV cache - preallocate for max sequence length
    max_seq_len = prompt_length + num_new_tokens  # Prefill + decode tokens
    kv_caches = []
    for layer_idx in range(num_layers):
        k = past_key_values[layer_idx][0].squeeze(0).transpose(0, 1).contiguous()
        v = past_key_values[layer_idx][1].squeeze(0).transpose(0, 1).contiguous()
        k = k.reshape(k.shape[0], -1)
        v = v.reshape(v.shape[0], -1)
        
        # Preallocate cache with max length
        k_cache_full = torch.zeros((max_seq_len, hidden_size), dtype=torch.float16, device=device)
        v_cache_full = torch.zeros((max_seq_len, hidden_size), dtype=torch.float16, device=device)
        k_cache_full[:k.shape[0]] = k
        v_cache_full[:v.shape[0]] = v
        
        kv_caches.append((k_cache_full, v_cache_full, k.shape[0]))  # (k_cache, v_cache, current_len)
    
    # Decoding
    for step in range(num_new_tokens - 1):
        current_position = prompt_length + step
        
        # Debug: print first two steps info
        if step <= 1:
            print(f"\n[DEBUG] Decode step {step}:")
            print(f"  current_position: {current_position}")
            print(f"  next_token to embed: {next_token.item()}")
            print(f"  Layer 0 current_len: {kv_caches[0][2]}")
        
        # Embedding
        hidden_states = model.gpt_neox.embed_in(next_token).half().squeeze(1)
        
        # RoPE embeddings
        cos, sin = compute_rope_embeddings(current_position, rotary_dim, head_dim, base=10000, device=device)
        
        # Through all layers
        for layer_idx in range(num_layers):
            weights = all_weights[layer_idx]
            k_cache_full, v_cache_full, current_len = kv_caches[layer_idx]
            
            residual = hidden_states.clone()
            
            # ========== Attention with ClusterFusion Kernel ==========
            # Pass full cache buffer and current length
            # Kernel will read cache[0:current_len] and write to cache[current_len]
            attn_output, new_k, new_v = clusterfusion.pythia_decoder_layer(
                hidden_states,
                weights['qkv_weight'],
                weights['qkv_bias'],
                weights['o_weight'],
                weights['o_bias'],  # Kernel adds bias using atomicAdd
                k_cache_full,  # Full cache buffer
                v_cache_full,  # Full cache buffer
                weights['ln_weight'],
                weights['ln_bias'],
                cos,
                sin,
                current_len  # Current sequence length
            )
            
            # Bias is now added in kernel, no need to add here
            
            # ========== MLP with PyTorch (parallel) ==========
            mlp_input = F.layer_norm(residual, (hidden_size,), weights['post_ln_weight'].squeeze(0), weights['post_ln_bias'].squeeze(0), eps=1e-5)
            mlp_hidden = F.linear(mlp_input, weights['mlp_up_weight'], weights['mlp_up_bias'])
            mlp_hidden = F.gelu(mlp_hidden)
            mlp_output = F.linear(mlp_hidden, weights['mlp_down_weight'], weights['mlp_down_bias'])
            
            # ========== Parallel Residual ==========
            hidden_states = residual + attn_output + mlp_output
            
            # Debug: track all layers in first step
            if step == 0:
                hs_norm = hidden_states.norm().item()
                attn_norm = attn_output.norm().item()
                if layer_idx < 15 or torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
                    print(f"  Layer {layer_idx}: hidden_norm={hs_norm:.2f}, attn_norm={attn_norm:.2f}, has_nan={torch.isnan(hidden_states).any()}, has_inf={torch.isinf(hidden_states).any()}")
                if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
                    print(f"    !!! Problem in layer {layer_idx} !!!")
                    break
            
            # Update current length (kernel already wrote to cache[current_len])
            kv_caches[layer_idx] = (k_cache_full, v_cache_full, current_len + 1)
            
            # Debug: check if cache was written for first layer, first two steps
            if step <= 1 and layer_idx == 0:
                print(f"  Layer 0 after kernel (step {step}):")
                print(f"    attn_output norm: {attn_output.norm().item():.4f}")
                print(f"    new_k norm: {new_k.norm().item():.4f}")
                print(f"    cache[{current_len}] norm: {k_cache_full[current_len].norm().item():.4f}")
                if step == 1:
                    # Check if previous cache is still there
                    print(f"    cache[{current_len-1}] norm (previous): {k_cache_full[current_len-1].norm().item():.4f}")
        
        # Final LayerNorm
        hidden_states = F.layer_norm(
            hidden_states, (hidden_size,),
            model.gpt_neox.final_layer_norm.weight.data,
            model.gpt_neox.final_layer_norm.bias.data,
            eps=1e-5
        )
        
        # Logits
        logits = model.embed_out(hidden_states)
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated_ids.append(next_token.item())
        
        # Debug: print logits for first two steps
        if step <= 1:
            print(f"  Final hidden_states norm: {hidden_states.norm().item():.4f}")
            print(f"  logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
            print(f"  Generated token: {next_token.item()}")
        
        if (step + 1) % 5 == 0:
            print(f"  Step {step+1}: token={next_token.item()}")
    
    full_ids = input_ids[0].tolist() + generated_ids
    generated_text = tokenizer.decode(full_ids, skip_special_tokens=True)
    
    print(f"\\n{'='*80}")
    print(f"Generated text:\\n{generated_text}")
    print(f"{'='*80}\\n")
    
    return generated_text, generated_ids

if __name__ == "__main__":
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map='cuda:0')
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    prompt = "The meaning of life is"
    num_tokens = 20
    
    # ClusterFusion + PyTorch MLP
    start = time.time()
    text_kernel, ids_kernel = generate_with_kernel(model, tokenizer, prompt, num_tokens)
    time_kernel = time.time() - start
    
    # HuggingFace reference
    print(f"{'='*80}")
    print(f"HuggingFace Reference")
    print(f"{'='*80}\\n")
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda:0')
    with torch.no_grad():
        start = time.time()
        output_ids_hf = model.generate(input_ids, max_new_tokens=num_tokens, do_sample=False, use_cache=True)
        time_hf = time.time() - start
    
    text_hf = tokenizer.decode(output_ids_hf[0], skip_special_tokens=True)
    ids_hf = output_ids_hf[0].tolist()
    
    print(f"Generated text:\\n{text_hf}")
    
    # Compare
    print(f"\\n{'='*80}")
    print(f"Comparison")
    print(f"{'='*80}")
    print(f"ClusterFusion+PyTorch time: {time_kernel:.3f}s")
    print(f"HuggingFace time: {time_hf:.3f}s")
    print(f"Speedup: {time_hf/time_kernel:.2f}x")
    print(f"\\nText match: {text_kernel == text_hf}")
    
    ids_kernel_full = input_ids[0].tolist() + ids_kernel
    print(f"Token IDs match: {ids_kernel_full == ids_hf}")
    
    if ids_kernel_full != ids_hf:
        print(f"\\nKernel IDs: {ids_kernel_full}")
        print(f"HuggingFace IDs: {ids_hf}")
        
        for i, (k, h) in enumerate(zip(ids_kernel_full, ids_hf)):
            if k != h:
                print(f"\\nFirst mismatch at position {i}:")
                print(f"  Kernel: {k} ('{tokenizer.decode([k])}')")
                print(f"  HuggingFace: {h} ('{tokenizer.decode([h])}')")
                break

