"""
Compare random data (test_pythia.py) vs real data statistics
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "EleutherAI/pythia-2.8b"

# Random data (like test_pythia.py)
torch.manual_seed(42)
random_input = (torch.randn((1, 2560)) * 0.1).half().cuda()
random_cache = (torch.randn((128, 2560)) * 0.1).half().cuda()
random_cos = torch.cos((torch.rand((1, 20)) * (2 * torch.pi)).cuda())
random_sin = torch.sin((torch.rand((1, 20)) * (2 * torch.pi)).cuda())

print("=== RANDOM DATA (test_pythia.py style) ===")
print(f"Input norm: {random_input.norm().item():.4f}")
print(f"Input range: [{random_input.min().item():.4f}, {random_input.max().item():.4f}]")
print(f"Cache norm: {random_cache.norm().item():.4f}")
print(f"Cache range: [{random_cache.min().item():.4f}, {random_cache.max().item():.4f}]")
print(f"Cos range: [{random_cos.min().item():.4f}, {random_cos.max().item():.4f}]")
print(f"Sin range: [{random_sin.min().item():.4f}, {random_sin.max().item():.4f}]")

# Real data
print("\n=== REAL DATA (from model) ===")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map='cuda:0')
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

prompt = "The meaning of life is"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda:0')

with torch.no_grad():
    outputs = model(input_ids, use_cache=True)
    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]

next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
real_input = model.gpt_neox.embed_in(next_token).half().squeeze(1)

k = past_key_values[0][0].squeeze(0).transpose(0, 1).contiguous()
real_cache = k.reshape(k.shape[0], -1)

position = 5
inv_freq = 1.0 / (10000 ** (torch.arange(0, 20, 2, dtype=torch.float32, device='cuda:0') / 20))
position_tensor = torch.tensor([position], dtype=torch.float32, device='cuda:0')
freqs = torch.outer(position_tensor, inv_freq)
emb = torch.cat([freqs, freqs], dim=-1)
real_cos = emb.cos()
real_sin = emb.sin()

print(f"Input norm: {real_input.norm().item():.4f}")
print(f"Input range: [{real_input.min().item():.4f}, {real_input.max().item():.4f}]")
print(f"Cache norm: {real_cache.norm().item():.4f}")
print(f"Cache range: [{real_cache.min().item():.4f}, {real_cache.max().item():.4f}]")
print(f"Cos range: [{real_cos.min().item():.4f}, {real_cos.max().item():.4f}]")
print(f"Sin range: [{real_sin.min().item():.4f}, {real_sin.max().item():.4f}]")

print("\n=== COMPARISON ===")
print(f"Input norm ratio (real/random): {real_input.norm().item() / random_input.norm().item():.2f}x")
print(f"Cache norm ratio (real/random): {real_cache.norm().item() / random_cache.norm().item():.2f}x")

