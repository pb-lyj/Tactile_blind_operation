"""
配置管理工具
用于保存和加载配置文件
"""

import os
import json


def save_config_to_json(config: dict, save_path: str, overwrite: bool = False):
    """
    将配置字典保存为 JSON 文件
    
    Args:
        config: 配置字典
        save_path: 保存路径
        overwrite: 是否覆盖已存在的文件
    """
    if os.path.exists(save_path) and not overwrite:
        raise FileExistsError(f"{save_path} already exists. Use overwrite=True to overwrite.")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Config saved to {save_path}")


def load_config_from_json(config_path: str) -> dict:
    """
    从 JSON 文件加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        dict: 配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    print(f"Config loaded from {config_path}")
    return config
