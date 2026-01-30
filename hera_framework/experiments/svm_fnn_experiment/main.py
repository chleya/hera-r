#!/usr/bin/env python3
"""
SVM分析FNN隐藏层激活向量的主实验脚本
实验设计: 验证SVM分析神经网络激活向量的可行性
"""

import os
import sys
import yaml
import logging
import numpy as np
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import load_and_preprocess_data
from models.fnn_model import create_fnn_model, train_fnn_model
from analysis.activation_extractor import extract_activations
from analysis.svm_analyzer import analyze_with_svm
from visualization.plotter import visualize_results
from utils.logger import setup_logger
from utils.config_loader import load_config

def main():
    """主实验流程"""
    # 1. 加载配置
    config = load_config()
    logger = setup_logger(config)
    
    logger.info("=" * 60)
    logger.info("开始SVM分析FNN激活向量实验")
    logger.info(f"实验名称: {config['experiment']['name']}")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    try:
        # 2. 数据准备
        logger.info("步骤1: 数据准备")
        train_loader, test_loader, simplified_loader = load_and_preprocess_data(config)
        logger.info(f"数据加载完成: {len(train_loader.dataset)}训练样本")
        
        # 3. 训练FNN模型
        logger.info("步骤2: 训练FNN模型")
        fnn_models = {}
        
        # 训练简单FNN
        logger.info("训练简单FNN模型...")
        simple_fnn = create_fnn_model(config, model_type="simple_fnn")
        simple_fnn, simple_history = train_fnn_model(
            simple_fnn, train_loader, test_loader, config
        )
        fnn_models["simple_fnn"] = simple_fnn
        logger.info(f"简单FNN训练完成，测试准确率: {simple_history['test_accuracy'][-1]:.2%}")
        
        # 训练中等FNN
        logger.info("训练中等FNN模型...")
        medium_fnn = create_fnn_model(config, model_type="medium_fnn")
        medium_fnn, medium_history = train_fnn_model(
            medium_fnn, train_loader, test_loader, config
        )
        fnn_models["medium_fnn"] = medium_fnn
        logger.info(f"中等FNN训练完成，测试准确率: {medium_history['test_accuracy'][-1]:.2%}")
        
        # 4. 提取激活向量
        logger.info("步骤3: 提取激活向量")
        activations_data = {}
        
        for model_name, model in fnn_models.items():
            logger.info(f"提取{model_name}的激活向量...")
            
            # 提取完整MNIST数据的激活向量
            activations_full, labels_full = extract_activations(
                model, train_loader, config, layer_type="hidden"
            )
            activations_data[f"{model_name}_full"] = {
                "activations": activations_full,
                "labels": labels_full
            }
            logger.info(f"  完整数据: {activations_full.shape}激活向量")
            
            # 提取简化MNIST数据的激活向量
            activations_simple, labels_simple = extract_activations(
                model, simplified_loader, config, layer_type="hidden"
            )
            activations_data[f"{model_name}_simple"] = {
                "activations": activations_simple,
                "labels": labels_simple
            }
            logger.info(f"  简化数据: {activations_simple.shape}激活向量")
        
        # 5. SVM分析
        logger.info("步骤4: SVM分析")
        svm_results = {}
        
        experiment_groups = config["experiment_groups"]
        for group_name, group_config in experiment_groups.items():
            if group_config.get("baseline", False):
                # 随机基线
                logger.info(f"运行{group_name}: 随机基线")
                continue
            
            logger.info(f"运行{group_name}: {group_config['name']}")
            
            # 选择对应的激活向量数据
            if "simple_fnn" in group_name:
                model_key = "simple_fnn"
            else:
                model_key = "medium_fnn"
                
            if "simplified" in group_config.get("dataset", ""):
                data_key = f"{model_key}_simple"
            else:
                data_key = f"{model_key}_full"
            
            activations = activations_data[data_key]["activations"]
            labels = activations_data[data_key]["labels"]
            
            # 随机采样指定数量的样本
            n_samples = min(group_config.get("samples", 1000), len(activations))
            indices = np.random.choice(len(activations), n_samples, replace=False)
            activations_sampled = activations[indices]
            labels_sampled = labels[indices]
            
            # SVM分析
            result = analyze_with_svm(
                activations_sampled, 
                labels_sampled, 
                config, 
                kernel=group_config["svm_kernel"]
            )
            
            svm_results[group_name] = result
            logger.info(f"  SVM准确率: {result['accuracy']:.2%}")
        
        # 6. 可视化
        logger.info("步骤5: 结果可视化")
        visualize_results(activations_data, svm_results, config)
        
        # 7. 保存结果
        logger.info("步骤6: 保存结果")
        save_results(fnn_models, activations_data, svm_results, config)
        
        # 8. 实验总结
        logger.info("=" * 60)
        logger.info("实验完成!")
        logger.info("主要结果:")
        
        for group_name, result in svm_results.items():
            if "accuracy" in result:
                logger.info(f"  {group_name}: SVM准确率 = {result['accuracy']:.2%}")
        
        logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"实验执行失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def save_results(fnn_models, activations_data, svm_results, config):
    """保存实验结果"""
    import torch
    import pickle
    
    output_dir = config["output"]["results_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存模型
    models_dir = config["output"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    
    for model_name, model in fnn_models.items():
        model_path = os.path.join(models_dir, f"{model_name}.pth")
        torch.save(model.state_dict(), model_path)
    
    # 保存激活向量数据
    data_path = os.path.join(output_dir, "activations_data.pkl")
    with open(data_path, 'wb') as f:
        pickle.dump(activations_data, f)
    
    # 保存SVM结果
    results_path = os.path.join(output_dir, "svm_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(svm_results, f)
    
    # 保存配置
    config_path = os.path.join(output_dir, "experiment_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    logging.getLogger(__name__).info(f"结果已保存到: {output_dir}")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)