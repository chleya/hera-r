# H.E.R.A.-R 依赖安装指南

## 问题描述

在运行H.E.R.A.-R时，可能会遇到以下错误：
```
ImportError: cannot import name 'BertForPreTraining' from 'transformers'
```

这是由于`transformers`库版本不兼容导致的。

## 解决方案

### 方法1：使用requirements.txt（推荐）

```bash
# 安装所有依赖（使用项目提供的requirements.txt）
pip install -r requirements.txt
```

### 方法2：手动安装特定版本

```bash
# 安装兼容版本的transformers
pip install transformers==4.36.0

# 安装其他核心依赖
pip install torch
pip install transformer-lens
pip install sae-lens
pip install pyyaml
pip install rich  # 用于彩色输出
```

### 方法3：创建虚拟环境（最佳实践）

```bash
# 创建虚拟环境
python -m venv hera_env

# 激活虚拟环境
# Windows:
hera_env\Scripts\activate
# Linux/Mac:
source hera_env/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 验证安装

运行以下命令验证安装是否成功：

```bash
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import transformer_lens; print('TransformerLens: OK')"
python -c "import sae_lens; print('SAE Lens: OK')"
```

## 已知兼容版本

| 包 | 推荐版本 | 测试状态 |
|----|----------|----------|
| transformers | 4.36.0 | ✅ 兼容 |
| torch | >=2.0.0 | ✅ 兼容 |
| transformer-lens | 最新版 | ✅ 兼容 |
| sae-lens | 最新版 | ✅ 兼容 |
| python | 3.9+ | ✅ 兼容 |

## 故障排除

### 问题1：CUDA不可用
```bash
# 如果使用CPU
# 修改 configs/default.yaml 中的 device: "cpu"
```

### 问题2：内存不足
```bash
# 使用较小的模型
# 修改 configs/default.yaml 中的 model.name
# 可选: "gpt2-small", "gpt2", "gpt2-medium"
```

### 问题3：其他导入错误
```bash
# 更新所有包
pip install --upgrade transformers transformer-lens sae-lens
```

## 快速测试

安装完成后，运行快速测试：

```bash
# 测试项目结构
python test_project_structure.py

# 运行最小示例
python minimal_working_example.py
```

## 支持

如果仍有问题，请：
1. 检查Python版本（需要3.9+）
2. 确保使用虚拟环境
3. 查看详细的错误信息
4. 在项目issues中报告问题