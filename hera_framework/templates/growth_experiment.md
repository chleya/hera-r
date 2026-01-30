# 生长实验模板

## 快速开始

### 1. 创建实验目录
在 `experiments/` 目录下创建新实验目录。

### 2. 创建配置文件 (config.yaml)
```yaml
# 生长实验配置
experiment:
  name: "growth_experiment"
  random_seed: 42

data:
  dataset: "mnist"  # 可选: mnist, cifar10
  batch_size: 32

network:
  input_size: 784    # MNIST: 784, CIFAR-10: 3072
  output_size: 10
  initial_hidden_size: 64

training:
  total_epochs: 20
  learning_rate: 0.001
  growth_check_interval: 5  # 每5个epoch检查生长

growth:
  strategy: "alternating"  # width_only, depth_only, alternating
  threshold: 0.5           # 准确率提升小于0.5%时生长
  max_growth_steps: 3
```

### 3. 创建运行脚本 (run.py)
```python
#!/usr/bin/env python3
import sys
import os

# 添加生长平台路径
platform_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "growth_platform"
)
sys.path.append(platform_path)

from run_experiment import main

if __name__ == "__main__":
    config_file = os.path.join(os.path.dirname(__file__), "config.yaml")
    sys.argv = ["run_experiment.py", "--config", config_file]
    
    success = main()
    exit(0 if success else 1)
```

### 4. 运行实验
```bash
python run.py
```

## 实验设计

### 生长策略
- **width_only**: 只增加神经元数量
- **depth_only**: 只增加网络层数  
- **alternating**: 交替宽度和深度生长

### 关键参数
- **threshold**: 生长触发阈值 (0.1-2.0)
- **check_interval**: 生长检查间隔 (3-10 epochs)
- **max_steps**: 最大生长次数 (3-10)

## 结果分析

实验结果保存在 `results/` 目录中，包括：
- 训练历史数据
- 最终模型参数
- 表示质量指标
- 实验日志

## 示例

查看 `experiments/growth_example/` 目录中的完整示例。