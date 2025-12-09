# ClusterFusion Pythia-2.8b 测试指南

本指南说明如何编译和测试为Pythia-2.8b模型移植的ClusterFusion kernel。

## 环境要求

- NVIDIA RTX 5090 GPU (SM 12.0)
- CUDA Toolkit 12.x
- Python 3.8+
- PyTorch 2.0+ with CUDA support
- FlashInfer (用于参考实现)
- Transformers (用于加载真实模型)

## 安装依赖

```bash
# 安装PyTorch (根据你的CUDA版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install transformers flashinfer packaging
```

## 编译

### Linux/Mac

```bash
chmod +x scripts/pythia_build_test.sh
./scripts/pythia_build_test.sh
```

### Windows

```powershell
.\scripts\pythia_build_test.ps1
```

### 手动编译

```bash
python setup.py build_ext --inplace
```

## 测试

### 1. 基础正确性测试

这个测试使用随机数据验证kernel的数值正确性，与参考实现对比。

```bash
python tests/test_pythia.py
```

**预期输出：**
- 多次运行的误差统计
- MAE (Mean Absolute Error) 应该 < 0.01
- 测试通过提示

### 2. 真实模型集成测试

这个测试使用真实的Pythia-2.8b模型权重。

```bash
# 首次运行会自动下载模型（约5GB）
python tests/test_pythia_model.py --layer 0 --seq-len 128
```

**参数说明：**
- `--layer N`: 测试第N层 (0-31)
- `--seq-len N`: KV cache序列长度
- `--benchmark`: 运行性能benchmark
- `--num-runs N`: benchmark迭代次数
- `--test-generation`: 测试文本生成（TODO）

**示例：**

```bash
# 测试第0层，序列长度128
python tests/test_pythia_model.py --layer 0 --seq-len 128

# 运行benchmark，序列长度2048，100次迭代
python tests/test_pythia_model.py --layer 0 --seq-len 2048 --benchmark --num-runs 100

# 测试多个层
for i in {0..5}; do
    python tests/test_pythia_model.py --layer $i --seq-len 128
done
```

### 3. 使用自动化脚本

**完整测试（包括模型）：**

```bash
# Linux/Mac
./scripts/pythia_build_test.sh --full

# Windows
.\scripts\pythia_build_test.ps1 -full
```

**带benchmark：**

```bash
# Linux/Mac
./scripts/pythia_build_test.sh --full --benchmark

# Windows
.\scripts\pythia_build_test.ps1 -full -benchmark
```

## 预期结果

### 基础测试 (test_pythia.py)

```
=== Running ClusterFusion Pythia Kernel ===
First run output shape: torch.Size([1, 2560])
...
=== Error Statistics over 100 runs ===
Max MAE: 0.00234
Min MAE: 0.00189
...
Average MAE: 0.00211
✓ TEST PASSED: Average error is acceptable
```

### 模型测试 (test_pythia_model.py)

```
Testing Layer 0 with sequence length 128
========================================
Running ClusterFusion Pythia kernel...
Kernel execution time: 2.345 ms
Output shape: torch.Size([1, 2560])
Output mean: 0.001234
...
```

### Benchmark结果

```
Benchmark Results:
  Average time per iteration: 2.34 ms
  Throughput: 427.35 iterations/sec
  Total time: 0.234 sec
```

## 调试

### 启用调试输出

在`test_pythia.py`中设置：
```python
debug = 1  # 启用详细输出
```

这会打印：
- 归一化后的隐藏状态
- RoPE前后的Q/K值
- 注意力输出
- 最终输出

### 常见问题

#### 1. 编译错误：`cuda_runtime.h not found`

**解决方案：**
```bash
# 确保CUDA环境变量正确设置
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

#### 2. 运行时错误：`CUDA error: invalid configuration argument`

**可能原因：**
- 配置常量计算错误（如DEC_TILE, NUM_THREAD_PER_ROW_2）
- 共享内存超限

**解决方案：**
- 检查`config.h`中的宏定义
- 使用较小的序列长度测试

#### 3. 数值误差过大

**可能原因：**
- RoPE实现不匹配（rotary_pct=0.25）
- LayerNorm vs RMSNorm差异

**解决方案：**
- 确认RoPE只应用于前20维
- 检查cos/sin的初始化

#### 4. 性能低于预期

**优化建议：**
- 调整`TMA_LOAD_ONCE`参数（64, 128, 256）
- 检查warp利用率（HEAD_DIM=80可能导致warp空闲）
- 使用nsight compute profiling

## 性能分析

### 使用Nsight Compute

```bash
ncu --set full -o profile python tests/test_pythia_model.py --layer 0 --seq-len 2048
```

### 关键指标

- **SM Occupancy**: 应该 > 50%
- **Memory Throughput**: 检查全局内存带宽利用率
- **Warp Execution Efficiency**: HEAD_DIM=80可能导致较低的warp效率

## 已知限制

1. **只支持5090 (SM 12.0)**：H100版本未实现
2. **不支持sglang batch processing**：移除了batch相关代码
3. **LayerNorm简化为RMSNorm**：可能有轻微精度损失
4. **Warp利用率**：HEAD_DIM=80导致NUM_THREAD_PER_ROW=1，部分线程空闲

## 性能优化TODO

- [ ] 优化warp级并行以提高HEAD_DIM=80的利用率
- [ ] 实现真正的LayerNorm而非RMSNorm
- [ ] 添加H100支持
- [ ] 优化TMA加载参数
- [ ] 支持更大的batch size

## 联系与反馈

如有问题或改进建议，请在GitHub issues中提出。

## 参考

- 原始ClusterFusion论文和实现（llama-2-7b）
- Pythia模型：https://github.com/EleutherAI/pythia
- GPT-NeoX架构文档
