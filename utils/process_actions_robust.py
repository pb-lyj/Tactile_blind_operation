#!/usr/bin/env python3
"""
将end_states.txt文件转换为动作变化量
处理各种数据格式和异常情况
"""

import numpy as np
import os
from pathlib import Path

def process_end_states_to_actions(file_path):
    """
    将_end_states.txt文件转换为动作变化量
    """
    try:
        # 读取数据
        data = np.loadtxt(file_path)
        
        # 检查数据维度
        if data.ndim == 1:
            # 只有一行数据，无法计算变化量
            print(f"文件 {file_path} 只有一行数据，跳过处理")
            return None
        
        # 检查是否至少有两行数据
        if data.shape[0] < 2:
            print(f"文件 {file_path} 数据行数不足，无法计算变化量，跳过处理")
            return None
            
        # 提取时间戳和位姿数据
        timestamps = data[:, 0]
        poses = data[:, 1:7]  # x, y, z, rx, ry, rz
        
        # 计算位姿变化量
        pose_deltas = np.diff(poses, axis=0)
        action_timestamps = timestamps[1:]  # 与动作对应的时间戳
        
        # 组合时间戳和动作
        actions = np.column_stack([action_timestamps, pose_deltas])
        
        return actions
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None

def save_actions(actions, base_path):
    """
    保存动作数据为txt和npy格式
    """
    if actions is None:
        return
        
    try:
        # 保存为txt格式
        txt_path = base_path / "_action.txt"
        np.savetxt(txt_path, actions, fmt='%.6f')
        
        # 保存为npy格式
        npy_path = base_path / "_action.npy"
        np.save(npy_path, actions)
        
        print(f"保存动作数据: {txt_path}")
    except Exception as e:
        print(f"保存动作数据时出错: {e}")

def main():
    # 数据目录
    data_dir = Path("/home/lyj/Program_python/Tactile_blind_operation/datasets/data25.7_aligned")
    
    if not data_dir.exists():
        print(f"数据目录不存在: {data_dir}")
        return
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    # 遍历所有子目录
    for category_dir in data_dir.iterdir():
        if category_dir.is_dir():
            print(f"\n处理类别目录: {category_dir.name}")
            
            for exp_dir in category_dir.iterdir():
                if exp_dir.is_dir():
                    end_states_file = exp_dir / "_end_states.txt"
                    
                    if end_states_file.exists():
                        print(f"处理: {exp_dir.name}")
                        
                        # 处理文件
                        actions = process_end_states_to_actions(end_states_file)
                        
                        if actions is not None:
                            save_actions(actions, exp_dir)
                            processed_count += 1
                        else:
                            skipped_count += 1
                    else:
                        print(f"文件不存在: {end_states_file}")
                        error_count += 1
    
    print(f"\n处理完成!")
    print(f"成功处理: {processed_count} 个文件")
    print(f"跳过: {skipped_count} 个文件")
    print(f"错误: {error_count} 个文件")

if __name__ == "__main__":
    main()
