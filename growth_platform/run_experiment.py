#!/usr/bin/env python3
"""
生长实验平台 - 主运行脚本
"""

import os
import sys
import yaml
import logging
import argparse
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.growing_net import SimpleGrowingNet
from core.trainer import GrowthTrainer
from utils.metrics import calculate_representation_quality
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def setup_logging(config):
    """设置日志"""
    log_level = getattr(logging, config['logging']['level'])
    
    # 创建日志目录
    save_dir = config['output']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    # 配置日志
    log_file = os.path.join(save_dir, config['logging']['log_file'])
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler() if config['logging']['console_output'] else logging.NullHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_data(config):
    """加载数据"""
    logger = logging.getLogger(__name__)
    
    # 数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) if config['data']['normalize'] else transforms.Lambda(lambda x: x)
    ])
    
    # 下载MNIST数据集
    logger.info("加载MNIST数据集...")
    train_dataset = datasets.MNIST(
        './data', train=True, download=config['data']['download'], transform=transform
    )
    test_dataset = datasets.MNIST(
        './data', train=False, download=config['data']['download'], transform=transform
    )
    
    # 转换为扁平向量
    def flatten_transform(batch):
        images, labels = batch
        images = images.view(images.size(0), -1)  # 展平为向量
        return images, labels
    
    # 创建数据加载器
    batch_size = config['data']['batch_size']
    
    # 为了快速测试，可以使用子集
    if config.get('debug', {}).get('use_subset', False):
        logger.info("使用数据子集进行调试...")
        subset_size = config['debug']['subset_size']
        train_dataset = Subset(train_dataset, range(subset_size))
        test_dataset = Subset(test_dataset, range(min(subset_size, len(test_dataset))))
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=flatten_transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=flatten_transform
    )
    
    logger.info(f"训练集: {len(train_dataset)}样本")
    logger.info(f"测试集: {len(test_dataset)}样本")
    
    return train_loader, test_loader

def run_experiment(config, experiment_name):
    """运行单个实验"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"开始实验: {experiment_name}")
    logger.info(f"配置: {config['growth']['strategy']}")
    
    # 设置随机种子
    torch.manual_seed(config['experiment']['random_seed'])
    np.random.seed(config['experiment']['random_seed'])
    
    # 创建模型
    model = SimpleGrowingNet(
        input_size=config['network']['input_size'],
        hidden_size=config['network']['initial_hidden_size'],
        output_size=config['network']['output_size']
    )
    
    # 创建训练器
    trainer = GrowthTrainer(
        model,
        device=config['experiment']['device'],
        learning_rate=config['training']['learning_rate'],
        growth_threshold=config['growth']['threshold'],
        max_growth_steps=config['growth']['max_growth_steps']
    )
    
    # 加载数据
    train_loader, test_loader = load_data(config)
    
    # 训练（带生长）
    history = trainer.train_with_growth(
        train_loader,
        test_loader,
        total_epochs=config['training']['total_epochs'],
        growth_check_interval=config['training']['growth_check_interval']
    )
    
    # 最终评估
    final_metrics = trainer.evaluate(test_loader)
    
    # 获取最终表示质量
    if config['evaluation']['save_activations']:
        activations, labels = trainer.get_activations_dataset(
            test_loader,
            max_samples=config['evaluation']['max_activation_samples']
        )
        representation_report = calculate_representation_quality(
            activations, labels, growth_step=len(model.growth_history)
        )
    else:
        representation_report = None
    
    # 保存结果
    if config['output']['save_history']:
        save_results(config, experiment_name, history, final_metrics, representation_report, model)
    
    return {
        'history': history,
        'final_metrics': final_metrics,
        'representation_report': representation_report,
        'growth_history': model.growth_history,
        'growth_decisions': trainer.growth_decisions
    }

def save_results(config, experiment_name, history, final_metrics, representation_report, model):
    """保存实验结果"""
    import pickle
    import json
    
    save_dir = config['output']['save_dir']
    exp_dir = os.path.join(save_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # 保存历史数据
    history_file = os.path.join(exp_dir, 'history.pkl')
    with open(history_file, 'wb') as f:
        pickle.dump(history, f)
    
    # 保存模型
    if config['output']['save_model']:
        model_file = os.path.join(exp_dir, 'model.pth')
        torch.save(model.state_dict(), model_file)
    
    # 保存最终指标
    metrics_file = os.path.join(exp_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump({
            'final_accuracy': final_metrics['accuracy'],
            'final_loss': final_metrics['loss'],
            'growth_steps': len(model.growth_history),
            'total_params': model.get_info()['total_params'],
            'representation_quality': representation_report
        }, f, indent=2)
    
    # 保存配置
    config_file = os.path.join(exp_dir, 'config.yaml')
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    logging.getLogger(__name__).info(f"结果已保存到: {exp_dir}")

def create_summary_report(results_dict, config):
    """创建实验总结报告"""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*60)
    logger.info("实验总结报告")
    logger.info("="*60)
    
    for exp_name, results in results_dict.items():
        logger.info(f"\n实验组: {exp_name}")
        logger.info(f"  最终测试准确率: {results['final_metrics']['accuracy']:.2f}%")
        logger.info(f"  生长次数: {len(results['growth_history'])}")
        logger.info(f"  生长决策: {len(results['growth_decisions'])}次")
        
        if results['representation_report']:
            sep = results['representation_report']['separability']
            logger.info(f"  表示分离度: {sep['separability_score']:.4f}")
            logger.info(f"  SVM准确率: {sep['svm_accuracy']:.4f}")
    
    # 比较不同策略
    logger.info("\n" + "-"*60)
    logger.info("策略比较:")
    
    best_acc = -1
    best_strategy = None
    
    for exp_name, results in results_dict.items():
        acc = results['final_metrics']['accuracy']
        logger.info(f"  {exp_name}: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            best_strategy = exp_name
    
    logger.info(f"\n最佳策略: {best_strategy} ({best_acc:.2f}%)")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='运行生长实验')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--debug', action='store_true', help='调试模式（使用数据子集）')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 调试模式
    if args.debug:
        config['debug'] = {'use_subset': True, 'subset_size': 1000}
        config['training']['total_epochs'] = 10
        config['growth']['max_growth_steps'] = 2
    
    # 设置日志
    logger = setup_logging(config)
    
    logger.info("="*60)
    logger.info("生长实验平台启动")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"实验名称: {config['experiment']['name']}")
    logger.info("="*60)
    
    try:
        # 运行所有实验组
        results_dict = {}
        
        for group in config['experiment_groups']:
            # 为每个实验组创建配置副本
            group_config = config.copy()
            group_config['growth']['strategy'] = group['growth_strategy']
            
            # 运行实验
            results = run_experiment(group_config, group['name'])
            results_dict[group['name']] = results
        
        # 创建总结报告
        create_summary_report(results_dict, config)
        
        logger.info("\n" + "="*60)
        logger.info("所有实验完成！")
        logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"实验执行失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)