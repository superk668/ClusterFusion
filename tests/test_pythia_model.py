"""
Test ClusterFusion Pythia kernel with real Pythia-2.8b model.
This script loads the actual model weights and tests the kernel integration.
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoXForCausalLM
import clusterfusion
import time

# Model configuration
MODEL_NAME = "EleutherAI/pythia-2.8b"
hidden_size = 2560
num_heads = 32
head_dim = 80
num_layers = 32
rotary_dim = 20  # 25% of head_dim

def load_pythia_model(device="cuda:0"):
    """Load Pythia-2.8b model and tokenizer"""
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=device
    )
    model.eval()
    print("Model loaded successfully!")
    return model, tokenizer

def extract_layer_weights(model, layer_idx):
    """Extract weights from a specific transformer layer"""
    layer = model.gpt_neox.layers[layer_idx]
    
    # Attention weights
    # Pythia uses separate Q, K, V projections
    weight_q = layer.attention.query_key_value.weight.data
    # weight_q shape: [3*hidden_size, hidden_size] = [7680, 2560]
    # Split into Q, K, V
    weight_qkv = weight_q.view(3, hidden_size, hidden_size).transpose(0, 1).contiguous()
    # Reshape to [hidden_size, 3*hidden_size]
    weight_qkv = weight_qkv.view(hidden_size, 3 * hidden_size).T
    
    weight_o = layer.attention.dense.weight.data
    
    # LayerNorm weights (input layernorm)
    layernorm_weight = layer.input_layernorm.weight.data.unsqueeze(0)
    
    return {
        'weight_qkv': weight_qkv.half(),
        'weight_o': weight_o.half(),
        'layernorm_weight': layernorm_weight.half()
    }

def initialize_rope_embeddings(rotary_dim, base=10000, seq_len=2048, device="cuda:0"):
    """
    Initialize RoPE embeddings matching Pythia's configuration.
    Pythia uses rotary_emb_base=10000 and rotary_pct=0.25
    """
    # Generate position indices
    position = torch.arange(seq_len, dtype=torch.float32, device=device)
    
    # Generate frequency
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim))
    
    # Compute angles
    freqs = torch.einsum('i,j->ij', position, inv_freq)
    
    # Get cos and sin for current position (last position for decoding)
    cos = freqs[-1:].cos()
    sin = freqs[-1:].sin()
    
    # Duplicate to match the Neox-style RoPE format
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    
    return cos, sin

def test_single_layer_inference(model, layer_idx=0, seq_len=128):
    """Test a single layer with ClusterFusion kernel"""
    print(f"\n{'='*80}")
    print(f"Testing Layer {layer_idx} with sequence length {seq_len}")
    print(f"{'='*80}")
    
    device = next(model.parameters()).device
    
    # Extract layer weights
    weights = extract_layer_weights(model, layer_idx)
    
    # Generate random input (simulating hidden states from previous layer)
    input_tensor = torch.randn(1, hidden_size, dtype=torch.float16, device=device) * 0.1
    
    # Initialize KV cache
    kv_cache = torch.randn(2, seq_len, hidden_size, dtype=torch.float16, device=device) * 0.1
    
    # Initialize RoPE embeddings
    cos, sin = initialize_rope_embeddings(rotary_dim, device=device)
    
    # Run ClusterFusion kernel
    print("\nRunning ClusterFusion Pythia kernel...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    output, k, v = clusterfusion.pythia_decoder_layer(
        input_tensor,
        weights['weight_qkv'],
        weights['weight_o'],
        kv_cache[0],
        kv_cache[1],
        weights['layernorm_weight'],
        cos,
        sin
    )
    
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    
    print(f"Kernel execution time: {elapsed_time*1000:.3f} ms")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.6f}")
    print(f"Output std: {output.std().item():.6f}")
    print(f"Output range: [{output.min().item():.6f}, {output.max().item():.6f}]")
    
    return output, k, v

def benchmark_kernel(model, layer_idx=0, seq_len=2048, num_runs=100, warmup=10):
    """Benchmark the kernel performance"""
    print(f"\n{'='*80}")
    print(f"Benchmarking Layer {layer_idx} with sequence length {seq_len}")
    print(f"Warmup runs: {warmup}, Benchmark runs: {num_runs}")
    print(f"{'='*80}")
    
    device = next(model.parameters()).device
    
    # Extract layer weights
    weights = extract_layer_weights(model, layer_idx)
    
    # Generate input
    input_tensor = torch.randn(1, hidden_size, dtype=torch.float16, device=device) * 0.1
    kv_cache = torch.randn(2, seq_len, hidden_size, dtype=torch.float16, device=device) * 0.1
    cos, sin = initialize_rope_embeddings(rotary_dim, device=device)
    
    # Warmup
    print("Warming up...")
    for _ in range(warmup):
        output, k, v = clusterfusion.pythia_decoder_layer(
            input_tensor, weights['weight_qkv'], weights['weight_o'],
            kv_cache[0], kv_cache[1], weights['layernorm_weight'], cos, sin
        )
    torch.cuda.synchronize()
    
    # Benchmark
    print(f"Running {num_runs} iterations...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_runs):
        output, k, v = clusterfusion.pythia_decoder_layer(
            input_tensor, weights['weight_qkv'], weights['weight_o'],
            kv_cache[0], kv_cache[1], weights['layernorm_weight'], cos, sin
        )
    
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    
    avg_time = elapsed_time / num_runs * 1000  # ms
    throughput = num_runs / elapsed_time
    
    print(f"\n{'='*80}")
    print(f"Benchmark Results:")
    print(f"  Average time per iteration: {avg_time:.3f} ms")
    print(f"  Throughput: {throughput:.2f} iterations/sec")
    print(f"  Total time: {elapsed_time:.3f} sec")
    print(f"{'='*80}")

def compare_with_huggingface(model, tokenizer, prompt="Hello, world!", max_length=50):
    """
    Compare generation results between ClusterFusion and HuggingFace implementation.
    Note: This is a placeholder for future full model integration.
    """
    print(f"\n{'='*80}")
    print("Comparing with HuggingFace Implementation")
    print(f"{'='*80}")
    print("Note: Full model integration not yet implemented.")
    print("This test only validates single layer inference.")
    
    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    print(f"Prompt: {prompt}")
    print(f"Input tokens: {input_ids}")
    
    # HuggingFace generation
    print("\nRunning HuggingFace generation...")
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=max_length, do_sample=False)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Generated text: {output_text}")
    
    # TODO: Implement full model generation with ClusterFusion kernels
    print("\nFull ClusterFusion model integration: TODO")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test ClusterFusion Pythia kernel with real model")
    parser.add_argument("--model-path", type=str, default=MODEL_NAME, help="Model path or HuggingFace model name")
    parser.add_argument("--layer", type=int, default=0, help="Layer index to test")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of benchmark runs")
    parser.add_argument("--test-generation", action="store_true", help="Test generation (HuggingFace comparison)")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_pythia_model()
    
    # Test single layer
    test_single_layer_inference(model, layer_idx=args.layer, seq_len=args.seq_len)
    
    # Benchmark if requested
    if args.benchmark:
        benchmark_kernel(model, layer_idx=args.layer, seq_len=args.seq_len, num_runs=args.num_runs)
    
    # Test generation if requested
    if args.test_generation:
        compare_with_huggingface(model, tokenizer)
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)
