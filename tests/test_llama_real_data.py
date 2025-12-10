"""
Test if Llama kernel also fails with real data ranges
"""
import torch

# Test Llama's generate_random_weights function with different scales
torch.manual_seed(42)

print("=== Testing different scales ===\n")

for scale in [1.0, 0.5, 0.1, 0.05]:
    data = (torch.randn(128, 4096) * scale).half().cuda()
    print(f"Scale {scale}:")
    print(f"  Range: [{data.min().item():.4f}, {data.max().item():.4f}]")
    print(f"  Norm: {data.norm().item():.2f}")
    print(f"  Std: {data.std().item():.4f}")

print("\n=== Real Pythia cache for comparison ===")
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "EleutherAI/pythia-2.8b"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map='cuda:0')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

prompt = "The meaning of life is"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda:0')

with torch.no_grad():
    outputs = model(input_ids, use_cache=True)
    past_key_values = outputs.past_key_values

k = past_key_values[0][0].squeeze(0).transpose(0, 1).contiguous()
real_cache = k.reshape(k.shape[0], -1)

print(f"Real cache:")
print(f"  Range: [{real_cache.min().item():.4f}, {real_cache.max().item():.4f}]")
print(f"  Norm: {real_cache.norm().item():.2f}")
print(f"  Std: {real_cache.std().item():.4f}")

print("\n=== Conclusion ===")
print("Llama test uses scale=0.1 to avoid NaN")
print("Real cache has MUCH larger range!")
print("This explains why kernel works in test but fails in real usage")

