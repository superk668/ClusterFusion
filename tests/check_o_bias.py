"""
Check if Pythia model has output projection bias
"""
from transformers import AutoModelForCausalLM

MODEL_NAME = "EleutherAI/pythia-2.8b"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype='float16', device_map='cuda:0')

layer = model.gpt_neox.layers[0]
print(f"\nLayer 0 attention.dense (output projection):")
print(f"  weight shape: {layer.attention.dense.weight.shape}")
print(f"  has bias: {layer.attention.dense.bias is not None}")

if layer.attention.dense.bias is not None:
    print(f"  bias shape: {layer.attention.dense.bias.shape}")
    print(f"  bias norm: {layer.attention.dense.bias.norm().item():.4f}")
    print(f"  bias first 8 values: {layer.attention.dense.bias[:8]}")
    print(f"  bias abs mean: {layer.attention.dense.bias.abs().mean().item():.4f}")

