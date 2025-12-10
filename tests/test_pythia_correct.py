"""
Correct PyTorch implementation matching HuggingFace Pythia exactly
Based on actual HuggingFace source code analysis
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

MODEL_NAME = "EleutherAI/pythia-2.8b"

def rotate_half(x):
    """Rotates half the hidden dims of the input (HuggingFace style)"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, rotary_dim):
    """Apply Neox-style RoPE (HuggingFace implementation)"""
    # Split into rotary and pass-through parts
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    
    # Apply rotation to rotary part only
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    
    # Concatenate back
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    
    return q_embed, k_embed

def compute_rope_embeddings(position, rotary_dim, head_dim, base=10000, device='cuda:0'):
    """
    Compute RoPE embeddings following HuggingFace GPTNeoXRotaryEmbedding
    emb = torch.cat((freqs, freqs), dim=-1)
    """
    # Compute inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim))
    
    # Compute frequencies for this position
    position_tensor = torch.tensor([position], dtype=torch.float32, device=device)
    freqs = torch.outer(position_tensor, inv_freq)  # [1, rotary_dim//2]
    
    # Neox-style: duplicate frequencies
    emb = torch.cat([freqs, freqs], dim=-1)  # [1, rotary_dim]
    
    cos = emb.cos()  # [1, rotary_dim]
    sin = emb.sin()  # [1, rotary_dim]
    
    return cos, sin

def pythia_layer_forward(
    hidden_states,
    k_cache, v_cache,
    # Weights
    ln_weight, ln_bias,
    qkv_weight, qkv_bias,
    o_weight, o_bias,
    post_ln_weight, post_ln_bias,
    mlp_up_weight, mlp_up_bias,
    mlp_down_weight, mlp_down_bias,
    # RoPE
    cos, sin,
    # Config
    num_heads=32,
    head_dim=80,
    rotary_dim=20,
):
    """
    PyTorch implementation matching HuggingFace Pythia layer exactly
    """
    hidden_size = hidden_states.shape[-1]
    residual = hidden_states
    
    # ========== Attention Branch ==========
    # 1. Input LayerNorm
    attn_input = F.layer_norm(hidden_states, (hidden_size,), ln_weight.squeeze(0), ln_bias.squeeze(0), eps=1e-5)
    
    # 2. QKV projection (interleaved)
    qkv = F.linear(attn_input, qkv_weight, qkv_bias)  # [1, 7680]
    qkv = qkv.view(1, num_heads, 3, head_dim)  # [1, 32, 3, 80]
    q = qkv[:, :, 0, :]  # [1, 32, 80]
    k_new = qkv[:, :, 1, :]
    v_new = qkv[:, :, 2, :]
    
    # 3. Apply RoPE (only to rotary_dim)
    # cos, sin are [1, rotary_dim], need to broadcast to [1, num_heads, rotary_dim]
    cos_expanded = cos.unsqueeze(1)  # [1, 1, rotary_dim]
    sin_expanded = sin.unsqueeze(1)
    q, k_new = apply_rotary_pos_emb(q, k_new, cos_expanded, sin_expanded, rotary_dim)
    
    # 4. Concatenate with KV cache
    seq_len = k_cache.shape[0]
    k_cache_shaped = k_cache.view(seq_len, num_heads, head_dim)
    v_cache_shaped = v_cache.view(seq_len, num_heads, head_dim)
    
    k = torch.cat([k_cache_shaped, k_new.squeeze(0).unsqueeze(0)], dim=0)  # [seq_len+1, 32, 80]
    v = torch.cat([v_cache_shaped, v_new.squeeze(0).unsqueeze(0)], dim=0)
    
    # 5. Attention
    q_attn = q.unsqueeze(2)  # [1, 32, 1, 80]
    k_attn = k.transpose(0, 1).unsqueeze(0)  # [1, 32, seq_len+1, 80]
    v_attn = v.transpose(0, 1).unsqueeze(0)
    
    # Compute attention in float32 for numerical stability
    attn_scores = torch.matmul(q_attn.float(), k_attn.float().transpose(-2, -1)) / (head_dim ** 0.5)
    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_weights, v_attn.float()).half()
    attn_output = attn_output.squeeze(2)  # [1, 32, 80]
    
    # 6. Output projection
    attn_output = attn_output.reshape(1, num_heads * head_dim)
    attn_output = F.linear(attn_output, o_weight, o_bias)
    
    # ========== MLP Branch (parallel with attention!) ==========
    # LayerNorm on ORIGINAL input (not attention output!)
    mlp_input = F.layer_norm(residual, (hidden_size,), post_ln_weight.squeeze(0), post_ln_bias.squeeze(0), eps=1e-5)
    
    # MLP
    mlp_hidden = F.linear(mlp_input, mlp_up_weight, mlp_up_bias)
    mlp_hidden = F.gelu(mlp_hidden)
    mlp_output = F.linear(mlp_hidden, mlp_down_weight, mlp_down_bias)
    
    # ========== Parallel Residual ==========
    # Pythia: x = x + attn(ln1(x)) + mlp(ln2(x))
    output = residual + attn_output + mlp_output
    
    return output, k_new.squeeze(0), v_new.squeeze(0)

def generate_pytorch(model, tokenizer, prompt, num_new_tokens=20):
    """Generate using pure PyTorch implementation"""
    print(f"\\n{'='*80}")
    print(f"PyTorch Implementation")
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
    
    # Convert KV cache
    kv_caches = []
    for layer_idx in range(num_layers):
        k = past_key_values[layer_idx][0].squeeze(0).transpose(0, 1).contiguous()
        v = past_key_values[layer_idx][1].squeeze(0).transpose(0, 1).contiguous()
        k = k.reshape(k.shape[0], -1)
        v = v.reshape(v.shape[0], -1)
        kv_caches.append((k, v))
    
    # Decoding
    for step in range(num_new_tokens - 1):
        current_position = prompt_length + step
        
        # Embedding
        hidden_states = model.gpt_neox.embed_in(next_token).half().squeeze(1)
        
        # RoPE embeddings for current position
        cos, sin = compute_rope_embeddings(current_position, rotary_dim, head_dim, base=10000, device=device)
        
        # Through all layers
        for layer_idx in range(num_layers):
            weights = all_weights[layer_idx]
            k_cache, v_cache = kv_caches[layer_idx]
            
            # Debug: track norms
            if step == 0 and layer_idx < 15:
                print(f"  PyTorch Layer {layer_idx} input norm: {hidden_states.norm().item():.2f}")
            
            hidden_states, new_k, new_v = pythia_layer_forward(
                hidden_states, k_cache, v_cache,
                weights['ln_weight'], weights['ln_bias'],
                weights['qkv_weight'], weights['qkv_bias'],
                weights['o_weight'], weights['o_bias'],
                weights['post_ln_weight'], weights['post_ln_bias'],
                weights['mlp_up_weight'], weights['mlp_up_bias'],
                weights['mlp_down_weight'], weights['mlp_down_bias'],
                cos, sin,
                num_heads, head_dim, rotary_dim
            )
            
            # Update cache
            kv_caches[layer_idx] = (
                torch.cat([k_cache, new_k.reshape(1, -1)], dim=0),
                torch.cat([v_cache, new_v.reshape(1, -1)], dim=0)
            )
        
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
    
    # PyTorch implementation
    start = time.time()
    text_pytorch, ids_pytorch = generate_pytorch(model, tokenizer, prompt, num_tokens)
    time_pytorch = time.time() - start
    
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
    print(f"PyTorch time: {time_pytorch:.3f}s")
    print(f"HuggingFace time: {time_hf:.3f}s")
    print(f"\\nText match: {text_pytorch == text_hf}")
    
    ids_pytorch_full = input_ids[0].tolist() + ids_pytorch
    print(f"Token IDs match: {ids_pytorch_full == ids_hf}")
    
    if ids_pytorch_full != ids_hf:
        print(f"\\nPyTorch IDs: {ids_pytorch_full}")
        print(f"HuggingFace IDs: {ids_hf}")
        
        # Find first mismatch
        for i, (p, h) in enumerate(zip(ids_pytorch_full, ids_hf)):
            if p != h:
                print(f"\\nFirst mismatch at position {i}:")
                print(f"  PyTorch: {p} ('{tokenizer.decode([p])}')")
                print(f"  HuggingFace: {h} ('{tokenizer.decode([h])}')")
                break

