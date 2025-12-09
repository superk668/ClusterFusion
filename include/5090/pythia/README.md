# Pythia-2.8b ClusterFusion 移植说明

## 模型参数对比

| 参数 | Llama-2-7B | Pythia-2.8B |
|------|------------|-------------|
| hidden_size | 4096 | 2560 |
| num_attention_heads | 32 | 32 |
| head_dim | 128 | 80 |
| intermediate_size (FFN) | 12288 | 10240 |
| num_layers | 32 | 32 |
| max_position_embeddings | 4096 | 2048 |

## 关键差异

### 1. HEAD_DIM: 128 → 80
这是最关键的变化，影响：
- **Warp级并行计算**：80不能被32整除，需要重新设计线程分配
- **内存对齐**：需要确保uint4向量化加载对齐（80 bytes = 40 half = 5个uint4）
- **寄存器分配**：qk数组和reg_reduce数组大小需要调整

### 2. RoPE配置
- **Llama**: 全部128维都应用RoPE
- **Pythia**: rotary_pct=0.25，只有前20维（ROTARY_DIM）应用Neox-style RoPE
- **实现**: 在kernel中只对tid < ROTARY_DIM的线程应用RoPE变换

### 3. LayerNorm vs RMSNorm
- **Llama**: 使用RMSNorm
- **Pythia**: 使用LayerNorm（但当前实现保留了RMSNorm以简化移植）
- **TODO**: 考虑实现完整的LayerNorm以提高精度

## Warp配置说明

```
NUM_WARPS = 4
WARP_SIZE = 32
BLOCK_SIZE = 128 threads

NUM_ROW_PER_WARP = HEAD_DIM / NUM_WARPS = 80 / 4 = 20
NUM_THREAD_PER_ROW = WARP_SIZE / NUM_ROW_PER_WARP = 32 / 20 = 1 (floor)
```

**注意**: 这意味着每个warp的32个线程中，只有20个线程有效工作于20行，另外12个线程可能空闲或需要特殊处理。

## 关键配置宏

### 基础参数
```cpp
#define HEAD_DIM 80
#define HEAD_NUM 32
#define HIDDEN_DIM 2560
#define FFN_DIM 10240
#define ROTARY_DIM 20  // 25% of HEAD_DIM
```

### 计算参数
```cpp
#define NUM_ROW_PER_WARP 20
#define NUM_THREAD_PER_ROW 1
#define NUM_THREAD_PER_ROW_2 10  // 80/8 for decoding
#define DIM_BLOCK_REDUCE 25      // 2*128/10
#define DEC_TILE 2               // 8/(32/10)
```

## 文件结构

```
include/5090/pythia/
├── config.h                    # 模型参数配置
├── kernel.cuh                  # 主kernel实现（基于llama修改）
└── pythia_kernel_dispatch.cu   # PyTorch绑定接口
```

## 使用注意事项

1. **不支持sglang**: 移除了所有sglang相关的batch processing代码
2. **仅支持5090架构**: H100版本未实现
3. **精度问题**: LayerNorm简化为RMSNorm可能导致轻微精度损失
4. **调试建议**: 
   - 先用小的sequence length测试（如128）
   - 验证RoPE只应用于前20维
   - 检查attention mask是否正确处理

## 待优化项

1. [ ] 实现真正的LayerNorm而非RMSNorm
2. [ ] 优化NUM_THREAD_PER_ROW=1的warp利用率
3. [ ] 添加H100支持
4. [ ] 性能profiling和调优
5. [ ] 添加单元测试验证正确性

## 编译命令（示例）

```bash
# 需要根据实际的setup.py或CMakeLists.txt调整
nvcc -arch=sm_120 -std=c++17 \
  -I include \
  pythia_kernel_dispatch.cu \
  -o pythia_kernel.so
```

## 参考
- 原始ClusterFusion论文和代码（llama-2-7b）
- Pythia模型配置：`config.json`
- GPT-NeoX架构文档（Pythia基于此架构）
