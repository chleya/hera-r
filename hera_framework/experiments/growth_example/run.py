#!/usr/bin/env python3
"""
生长实验示例 - 运行脚本
"""

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
    print("=== 运行生长实验示例 ===")
    
    config_file = os.path.join(os.path.dirname(__file__), "config.yaml")
    
    # 设置命令行参数
    sys.argv = ["run_experiment.py", "--config", config_file, "--debug"]
    
    print(f"配置文件: {config_file}")
    print("使用调试模式 (小数据集，快速运行)")
    
    try:
        success = main()
        if success:
            print("\n✅ 实验完成！")
            print("结果保存在 results/ 目录中")
        else:
            print("\n❌ 实验失败")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ 运行错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)