# 生长实验示例

这是一个使用生长实验平台的完整示例。

## 实验目的

演示如何使用H.E.R.A.-R框架管理生长实验，包括：
1. 实验配置管理
2. 自动化实验运行
3. 结果分析和记录

## 文件结构

```
growth_example/
├── config.yaml      # 实验配置
├── run.py          # 运行脚本
├── README.md       # 说明文档
└── results/        # 实验结果（运行后生成）
```

## 运行方法

### 1. 确保依赖已安装
```bash
# 进入生长平台目录安装依赖
cd ../../growth_platform
pip install -r requirements.txt
```

### 2. 运行实验
```bash
# 返回示例目录
cd ../experiments/growth_example

# 运行实验（调试模式，快速）
python run.py
```

### 3. 查看结果
```bash
# 查看结果目录
ls results/

# 查看日志
cat results/experiment.log
```

## 实验配置说明

### 关键参数
- **生长策略**: alternating (交替宽度和深度生长)
- **生长阈值**: 0.5% (准确率提升小于0.5%时生长)
- **训练轮数**: 15 epochs
- **生长检查**: 每3个epoch检查一次

### 预期输出
1. 下载MNIST数据集
2. 训练初始网络
3. 根据性能触发生长
4. 记录表示质量变化
5. 保存实验结果

## 结果分析

实验完成后，可以：
1. 查看 `results/` 目录中的输出文件
2. 分析表示分离度的变化
3. 比较不同生长阶段的效果
4. 使用H.E.R.A.-R模板撰写实验报告

## 扩展使用

基于此示例，可以：
1. 修改 `config.yaml` 测试不同参数
2. 创建新的实验目录进行比较研究
3. 集成到H.E.R.A.-R的研究工作流中
4. 使用 `growth_experiment.md` 模板设计新实验