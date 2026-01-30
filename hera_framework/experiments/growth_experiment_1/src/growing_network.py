"""
可生长神经网络基类
实现网络生长的基础功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

class GrowthType(Enum):
    """生长类型枚举"""
    WIDTH = "width"      # 宽度生长
    DEPTH = "depth"      # 深度生长
    MIXED = "mixed"      # 混合生长

@dataclass
class GrowthRecord:
    """生长记录数据结构"""
    growth_step: int
    growth_type: GrowthType
    timestamp: float
    network_state: Dict[str, Any]
    performance_before: Dict[str, float]
    performance_after: Dict[str, float]
    representation_metrics: Dict[str, float]

class GrowingNeuralNetwork(nn.Module):
    """
    可生长神经网络基类
    
    支持功能:
    1. 宽度生长: 增加隐藏层神经元数量
    2. 深度生长: 增加网络层数
    3. 混合生长: 宽度和深度交替生长
    4. 表示质量跟踪: 记录生长过程中的表示变化
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        initial_hidden_dims: List[int] = [128],
        activation: str = "relu",
        growth_strategy: GrowthType = GrowthType.WIDTH,
        growth_threshold: float = 0.1,
        max_growth_steps: int = 10
    ):
        """
        初始化可生长神经网络
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            initial_hidden_dims: 初始隐藏层维度列表
            activation: 激活函数类型
            growth_strategy: 生长策略
            growth_threshold: 生长阈值(性能提升小于此值则触发生长)
            max_growth_steps: 最大生长步数
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = initial_hidden_dims.copy()
        self.activation_type = activation
        self.growth_strategy = growth_strategy
        self.growth_threshold = growth_threshold
        self.max_growth_steps = max_growth_steps
        
        # 生长记录
        self.growth_history: List[GrowthRecord] = []
        self.current_growth_step = 0
        
        # 创建初始网络
        self.layers = self._create_layers()
        
        # 性能跟踪
        self.performance_history = []
        
    def _create_layers(self) -> nn.ModuleList:
        """创建网络层"""
        layers = nn.ModuleList()
        
        # 输入层
        if len(self.hidden_dims) > 0:
            layers.append(nn.Linear(self.input_dim, self.hidden_dims[0]))
        else:
            # 如果没有隐藏层，直接连接到输出层
            layers.append(nn.Linear(self.input_dim, self.output_dim))
            return layers
        
        # 隐藏层
        for i in range(len(self.hidden_dims) - 1):
            layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
        
        # 输出层
        layers.append(nn.Linear(self.hidden_dims[-1], self.output_dim))
        
        return layers
    
    def _get_activation(self, x: torch.Tensor) -> torch.Tensor:
        """应用激活函数"""
        if self.activation_type == "relu":
            return F.relu(x)
        elif self.activation_type == "sigmoid":
            return torch.sigmoid(x)
        elif self.activation_type == "tanh":
            return torch.tanh(x)
        else:
            return F.relu(x)  # 默认使用ReLU
    
    def forward(self, x: torch.Tensor, return_activations: bool = False) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            return_activations: 是否返回激活向量
            
        Returns:
            输出张量，如果return_activations为True则同时返回激活向量列表
        """
        activations = []
        current = x
        
        # 处理所有层（除了最后一层）
        for i, layer in enumerate(self.layers[:-1]):
            current = layer(current)
            current = self._get_activation(current)
            if return_activations:
                activations.append(current.detach().cpu())
        
        # 最后一层（输出层，无激活函数）
        output = self.layers[-1](current)
        
        if return_activations:
            return output, activations
        return output
    
    def get_network_info(self) -> Dict[str, Any]:
        """获取网络信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dims": self.hidden_dims.copy(),
            "num_layers": len(self.layers),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "growth_step": self.current_growth_step,
            "growth_history_length": len(self.growth_history)
        }
    
    def grow_width(self, layer_idx: int, new_neurons: int, initialization: str = "random") -> None:
        """
        宽度生长：增加特定层的神经元数量
        
        Args:
            layer_idx: 层索引（0-based）
            new_neurons: 新增神经元数量
            initialization: 新权重初始化方式（"random"或"zero"）
        """
        if layer_idx < 0 or layer_idx >= len(self.layers) - 1:
            raise ValueError(f"层索引 {layer_idx} 超出范围")
        
        old_layer = self.layers[layer_idx]
        old_in_features = old_layer.in_features
        old_out_features = old_layer.out_features
        new_out_features = old_out_features + new_neurons
        
        # 创建新层
        new_layer = nn.Linear(old_in_features, new_out_features)
        
        # 复制旧权重
        with torch.no_grad():
            new_layer.weight[:old_out_features, :] = old_layer.weight
            new_layer.bias[:old_out_features] = old_layer.bias
            
            # 初始化新增部分
            if initialization == "random":
                # 随机初始化
                nn.init.xavier_uniform_(new_layer.weight[old_out_features:, :])
                nn.init.zeros_(new_layer.bias[old_out_features:])
            else:
                # 零初始化
                nn.init.zeros_(new_layer.weight[old_out_features:, :])
                nn.init.zeros_(new_layer.bias[old_out_features:])
        
        # 替换层
        self.layers[layer_idx] = new_layer
        
        # 如果这不是最后一层，还需要更新下一层的输入维度
        if layer_idx < len(self.layers) - 2:
            next_layer = self.layers[layer_idx + 1]
            new_next_layer = nn.Linear(new_out_features, next_layer.out_features)
            
            with torch.no_grad():
                # 复制旧权重到新权重的相应位置
                new_next_layer.weight[:, :old_out_features] = next_layer.weight
                new_next_layer.bias = next_layer.bias
                
                # 初始化新增部分为零
                nn.init.zeros_(new_next_layer.weight[:, old_out_features:])
            
            self.layers[layer_idx + 1] = new_next_layer
        
        # 更新隐藏层维度记录
        if layer_idx < len(self.hidden_dims):
            self.hidden_dims[layer_idx] = new_out_features
    
    def grow_depth(self, new_layer_dim: int, position: int = -1) -> None:
        """
        深度生长：在网络中插入新层
        
        Args:
            new_layer_dim: 新层的维度
            position: 插入位置（-1表示在最后一层前插入）
        """
        if position < 0:
            position = len(self.layers) - 1  # 在输出层前插入
        
        if position < 1 or position > len(self.layers) - 1:
            raise ValueError(f"插入位置 {position} 无效")
        
        # 获取插入位置前后层的维度
        prev_layer = self.layers[position - 1]
        next_layer = self.layers[position]
        
        # 创建新层
        new_layer = nn.Linear(prev_layer.out_features, new_layer_dim)
        nn.init.xavier_uniform_(new_layer.weight)
        nn.init.zeros_(new_layer.bias)
        
        # 创建新的下一层（调整输入维度）
        new_next_layer = nn.Linear(new_layer_dim, next_layer.out_features)
        
        with torch.no_grad():
            # 随机初始化新下一层
            nn.init.xavier_uniform_(new_next_layer.weight)
            nn.init.zeros_(new_next_layer.bias)
        
        # 插入新层
        self.layers.insert(position, new_layer)
        
        # 替换下一层
        self.layers[position + 1] = new_next_layer
        
        # 更新隐藏层维度记录
        insert_hidden_idx = position - 1
        self.hidden_dims.insert(insert_hidden_idx, new_layer_dim)
    
    def should_grow(self, performance_improvement: float) -> bool:
        """
        判断是否应该生长
        
        Args:
            performance_improvement: 性能提升幅度
            
        Returns:
            是否应该执行生长
        """
        if self.current_growth_step >= self.max_growth_steps:
            return False
        
        # 简单策略：性能提升小于阈值时生长
        return performance_improvement < self.growth_threshold
    
    def execute_growth(
        self,
        growth_type: Optional[GrowthType] = None,
        growth_params: Optional[Dict[str, Any]] = None
    ) -> GrowthRecord:
        """
        执行生长操作
        
        Args:
            growth_type: 生长类型，如果为None则使用默认策略
            growth_params: 生长参数
            
        Returns:
            生长记录
        """
        if growth_type is None:
            growth_type = self.growth_strategy
        
        if growth_params is None:
            growth_params = {}
        
        # 记录生长前的状态
        record = GrowthRecord(
            growth_step=self.current_growth_step,
            growth_type=growth_type,
            timestamp=time.time(),
            network_state=self.get_network_info(),
            performance_before={},
            performance_after={},
            representation_metrics={}
        )
        
        # 执行生长
        if growth_type == GrowthType.WIDTH:
            # 宽度生长：选择最需要生长的层
            layer_idx = growth_params.get("layer_idx", 0)
            new_neurons = growth_params.get("new_neurons", 32)
            initialization = growth_params.get("initialization", "random")
            self.grow_width(layer_idx, new_neurons, initialization)
            
        elif growth_type == GrowthType.DEPTH:
            # 深度生长
            new_layer_dim = growth_params.get("new_layer_dim", 128)
            position = growth_params.get("position", -1)
            self.grow_depth(new_layer_dim, position)
            
        elif growth_type == GrowthType.MIXED:
            # 混合生长：交替进行宽度和深度生长
            if self.current_growth_step % 2 == 0:
                # 偶数步：宽度生长
                self.grow_width(
                    growth_params.get("width_layer_idx", 0),
                    growth_params.get("width_new_neurons", 32),
                    growth_params.get("width_initialization", "random")
                )
            else:
                # 奇数步：深度生长
                self.grow_depth(
                    growth_params.get("depth_new_layer_dim", 128),
                    growth_params.get("depth_position", -1)
                )
        
        # 更新生长步数
        self.current_growth_step += 1
        
        # 保存记录
        self.growth_history.append(record)
        
        return record
    
    def get_activations(self, data_loader, device="cpu") -> Tuple[np.ndarray, np.ndarray]:
        """
        获取网络在所有隐藏层的激活向量
        
        Args:
            data_loader: 数据加载器
            device: 计算设备
            
        Returns:
            (activations, labels): 激活向量和对应标签
        """
        self.eval()
        all_activations = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(device), target.to(device)
                
                # 获取激活向量
                _, activations = self.forward(data, return_activations=True)
                
                # 将每层的激活向量展平并拼接
                batch_activations = []
                for layer_act in activations:
                    # 对每个样本取平均（减少维度）
                    layer_act_mean = layer_act.mean(dim=1) if len(layer_act.shape) > 2 else layer_act
                    batch_activations.append(layer_act_mean.numpy())
                
                # 按样本维度拼接各层特征
                if batch_activations:
                    combined = np.concatenate(batch_activations, axis=1)
                    all_activations.append(combined)
                    all_labels.append(target.cpu().numpy())
        
        if not all_activations:
            return np.array([]), np.array([])
        
        return np.vstack(all_activations), np.hstack(all_labels)
    
    def save_growth_history(self, filepath: str) -> None:
        """保存生长历史"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.growth_history, f)
    
    def load_growth_history(self, filepath: str) -> None:
        """加载生长历史"""
        import pickle
        with open(filepath, 'rb') as f:
            self.growth_history = pickle.load(f)

# 测试代码
if __name__ == "__main__":
    import time
    
    # 创建可生长网络
    print("创建可生长神经网络...")
    network = GrowingNeuralNetwork(
        input_dim=784,
        output_dim=10,
        initial_hidden_dims=[64],
        growth_strategy=GrowthType.WIDTH,
        max_growth_steps=5
    )
    
    print("初始网络信息:")
    info = network.get_network_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 测试宽度生长
    print("\n执行宽度生长...")
    record = network.execute_growth(
        growth_type=GrowthType.WIDTH,
        growth_params={"layer_idx": 0, "new_neurons": 32}
    )
    
    print("生长后网络信息:")
    info = network.get_network_info()
    print(f"  隐藏层维度: {info['hidden_dims']}")
    print(f"  总参数数: {info['total_params']:,}")
    
    # 测试深度生长
    print("\n执行深度生长...")
    record = network.execute_growth(
        growth_type=GrowthType.DEPTH,
        growth_params={"new_layer_dim": 128, "position": -1}
    )
    
    print("深度生长后网络信息:")
    info = network.get_network_info()
    print(f"  隐藏层维度: {info['hidden_dims']}")
    print(f"  总参数数: {info['total_params']:,}")
    print(f"  生长步数: {info['growth_step']}")
    
    print("\n测试完成！")