"""
日志配置工具
"""

import os
import logging
import sys
from datetime import datetime

def setup_logger(config=None):
    """
    设置日志配置
    
    Args:
        config: 配置字典，包含logging部分
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    if config is None or 'logging' not in config:
        # 使用默认配置
        log_config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'experiment.log',
            'console': True
        }
    else:
        log_config = config['logging']
    
    # 创建日志目录
    if config and 'output' in config and 'logs_dir' in config['output']:
        logs_dir = config['output']['logs_dir']
    else:
        logs_dir = './logs'
    
    os.makedirs(logs_dir, exist_ok=True)
    
    # 生成日志文件名
    if 'file' in log_config:
        log_file = os.path.join(logs_dir, log_config['file'])
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(logs_dir, f'experiment_{timestamp}.log')
    
    # 配置日志级别
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    log_level = level_map.get(log_config.get('level', 'INFO'), logging.INFO)
    
    # 配置日志格式
    log_format = log_config.get('format', 
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 清除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(log_format)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # 创建控制台处理器
    if log_config.get('console', True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(log_format)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # 创建实验主日志记录器
    experiment_logger = logging.getLogger('experiment')
    experiment_logger.setLevel(log_level)
    
    # 记录日志配置信息
    experiment_logger.info(f"日志系统初始化完成")
    experiment_logger.info(f"日志级别: {log_config.get('level', 'INFO')}")
    experiment_logger.info(f"日志文件: {log_file}")
    experiment_logger.info(f"控制台输出: {'启用' if log_config.get('console', True) else '禁用'}")
    
    return experiment_logger

def get_module_logger(module_name):
    """
    获取模块特定的日志记录器
    
    Args:
        module_name: 模块名称
        
    Returns:
        logging.Logger: 模块日志记录器
    """
    return logging.getLogger(module_name)

def log_experiment_start(experiment_name, config):
    """
    记录实验开始信息
    
    Args:
        experiment_name: 实验名称
        config: 实验配置
    """
    logger = logging.getLogger('experiment')
    
    logger.info("=" * 60)
    logger.info(f"实验开始: {experiment_name}")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"随机种子: {config.get('experiment', {}).get('random_seed', '未设置')}")
    logger.info("=" * 60)

def log_experiment_end(experiment_name, success=True, error_message=None):
    """
    记录实验结束信息
    
    Args:
        experiment_name: 实验名称
        success: 是否成功
        error_message: 错误信息（如果失败）
    """
    logger = logging.getLogger('experiment')
    
    logger.info("=" * 60)
    if success:
        logger.info(f"实验完成: {experiment_name}")
    else:
        logger.error(f"实验失败: {experiment_name}")
        if error_message:
            logger.error(f"错误信息: {error_message}")
    
    logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

def log_step_start(step_name, step_description=""):
    """
    记录步骤开始信息
    
    Args:
        step_name: 步骤名称
        step_description: 步骤描述
    """
    logger = logging.getLogger('experiment')
    
    logger.info("-" * 40)
    logger.info(f"开始步骤: {step_name}")
    if step_description:
        logger.info(f"步骤描述: {step_description}")
    logger.info("-" * 40)

def log_step_end(step_name, success=True, details=None):
    """
    记录步骤结束信息
    
    Args:
        step_name: 步骤名称
        success: 是否成功
        details: 详细信息
    """
    logger = logging.getLogger('experiment')
    
    if success:
        logger.info(f"步骤完成: {step_name}")
    else:
        logger.error(f"步骤失败: {step_name}")
    
    if details:
        if isinstance(details, dict):
            for key, value in details.items():
                logger.info(f"  {key}: {value}")
        else:
            logger.info(f"  详情: {details}")
    
    logger.info("-" * 40)

def log_metric(metric_name, value, group=None):
    """
    记录指标值
    
    Args:
        metric_name: 指标名称
        value: 指标值
        group: 分组信息（如实验组名称）
    """
    logger = logging.getLogger('experiment')
    
    if group:
        logger.info(f"[{group}] {metric_name}: {value}")
    else:
        logger.info(f"{metric_name}: {value}")

def log_warning(warning_message, module=None):
    """
    记录警告信息
    
    Args:
        warning_message: 警告信息
        module: 模块名称
    """
    if module:
        logger = logging.getLogger(module)
    else:
        logger = logging.getLogger('experiment')
    
    logger.warning(warning_message)

def log_error(error_message, module=None, exception=None):
    """
    记录错误信息
    
    Args:
        error_message: 错误信息
        module: 模块名称
        exception: 异常对象
    """
    if module:
        logger = logging.getLogger(module)
    else:
        logger = logging.getLogger('experiment')
    
    logger.error(error_message)
    if exception:
        logger.error(f"异常类型: {type(exception).__name__}")
        logger.error(f"异常详情: {str(exception)}")

def setup_test_logger():
    """
    设置测试用的日志配置
    """
    test_config = {
        'logging': {
            'level': 'DEBUG',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'test.log',
            'console': True
        },
        'output': {
            'logs_dir': './test_logs'
        }
    }
    
    return setup_logger(test_config)

if __name__ == "__main__":
    # 测试日志系统
    logger = setup_test_logger()
    
    # 测试各种日志级别
    logger.debug("这是一条调试信息")
    logger.info("这是一条信息")
    logger.warning("这是一条警告")
    logger.error("这是一条错误")
    
    # 测试步骤记录
    log_step_start("测试步骤", "测试步骤记录功能")
    log_metric("测试指标", 0.95, "测试组")
    log_step_end("测试步骤", success=True, details={"结果": "成功", "分数": 0.95})
    
    print("日志测试完成，请查看test_logs/test.log文件")