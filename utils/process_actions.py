#!/usr/bin/env python3
"""
处理末端位姿数据，计算位姿变化量
将_end_states.txt转换为_action.txt和_action.npy
"""

import numpy as np
import os
import glob
from pathlib import Path


def process_end_states_to_actions(file_path):
    """
    处理单个_end_states.txt文件，计算位姿变化量
    
    Args:
        file_path: _end_states.txt文件路径
        
    Returns:
        numpy.ndarray: 动作数据 [timestamp, dx, dy, dz, drx, dry, drz]
    """
    # 读取数据
    data = np.loadtxt(file_path)
    
    # 检查数据维度
    if data.ndim == 1:
        # 只有一行数据，无法计算变化量
        print(f"Warning: 文件 {file_path} 只有一行数据，跳过处理")
        return None
    
    if len(data) < 2:
        print(f"Warning: 文件 {file_path} 数据点不足2个，跳过处理")
        return None
    
    # 提取时间戳和位姿
    timestamps = data[:, 0]  # 时间戳
    poses = data[:, 1:]      # 位姿 [x, y, z, rx, ry, rz]
    
    # 计算位姿变化量
    pose_deltas = np.diff(poses, axis=0)  # 计算相邻位姿的差值
    
    # 组合结果：保留前n-1个时间戳，配对对应的位姿变化量
    # 时间戳[i] 对应 pose[i+1] - pose[i]
    action_data = np.column_stack([timestamps[:-1], pose_deltas])
    
    return action_data


def process_directory(directory_path):
    """
    处理单个目录中的_end_states.txt文件
    
    Args:
        directory_path: 目录路径
    """
    end_states_file = os.path.join(directory_path, '_end_states.txt')
    
    if not os.path.exists(end_states_file):
        print(f"Warning: {end_states_file} not found")
        return
    
    print(f"Processing: {directory_path}")
    
    # 处理位姿数据，计算动作
    action_data = process_end_states_to_actions(end_states_file)
    
    if action_data is None:
        return
    
    # 保存为txt格式
    action_txt_path = os.path.join(directory_path, '_action.txt')
    np.savetxt(action_txt_path, action_data, fmt='%.6f')
    
    # 保存为npy格式
    action_npy_path = os.path.join(directory_path, '_action.npy')
    np.save(action_npy_path, action_data)
    
    print(f"Saved: {action_txt_path}")
    print(f"Saved: {action_npy_path}")
    print(f"Generated {len(action_data)} action points from {len(action_data)+1} pose points\n")


def process_all_directories():
    """处理所有目录"""
    # 数据目录路径
    data_root = "/home/lyj/Program_python/Tactile_blind_operation/datasets/data25.7_aligned"
    
    # 获取所有子目录
    shape_dirs = ['cir_lar', 'cir_med', 'cir_sma', 'rect_lar', 'rect_med', 'rect_sma', 
                  'tri_lar', 'tri_med', 'tri_sma']
    
    total_processed = 0
    
    for shape_dir in shape_dirs:
        shape_path = os.path.join(data_root, shape_dir)
        if not os.path.exists(shape_path):
            print(f"Warning: Directory {shape_path} not found")
            continue
        
        print(f"\n=== Processing shape: {shape_dir} ===")
        
        # 获取该形状下的所有实验目录
        experiment_dirs = [d for d in os.listdir(shape_path) 
                          if os.path.isdir(os.path.join(shape_path, d))]
        
        for exp_dir in sorted(experiment_dirs):
            exp_path = os.path.join(shape_path, exp_dir)
            process_directory(exp_path)
            total_processed += 1
    
    print(f"\n=== Processing Complete ===")
    print(f"Total directories processed: {total_processed}")


def validate_action_data():
    """验证生成的动作数据"""
    print("\n=== Validation ===")
    
    # 找一个示例文件进行验证
    sample_dir = "/home/lyj/Program_python/Tactile_blind_operation/datasets/data25.7_aligned/cir_lar"
    
    # 获取第一个实验目录
    if os.path.exists(sample_dir):
        experiment_dirs = [d for d in os.listdir(sample_dir) 
                          if os.path.isdir(os.path.join(sample_dir, d))]
        
        if experiment_dirs:
            sample_path = os.path.join(sample_dir, sorted(experiment_dirs)[0])
            
            # 读取原始位姿数据
            end_states_file = os.path.join(sample_path, '_end_states.txt')
            action_file = os.path.join(sample_path, '_action.txt')
            
            if os.path.exists(end_states_file) and os.path.exists(action_file):
                end_states = np.loadtxt(end_states_file)
                actions = np.loadtxt(action_file)
                
                print(f"验证文件: {sample_path}")
                print(f"原始位姿数据点数: {len(end_states)}")
                print(f"动作数据点数: {len(actions)}")
                print(f"时间戳匹配: {np.allclose(end_states[:-1, 0], actions[:, 0])}")
                
                # 验证第一个动作计算是否正确
                if len(actions) > 0:
                    expected_action = end_states[1, 1:] - end_states[0, 1:]
                    actual_action = actions[0, 1:]
                    print(f"第一个动作计算正确: {np.allclose(expected_action, actual_action)}")
                    
                    print(f"第一个动作变化量: {actual_action}")
                    print(f"动作变化量范围:")
                    print(f"  dx: [{actions[:, 1].min():.6f}, {actions[:, 1].max():.6f}]")
                    print(f"  dy: [{actions[:, 2].min():.6f}, {actions[:, 2].max():.6f}]")
                    print(f"  dz: [{actions[:, 3].min():.6f}, {actions[:, 3].max():.6f}]")
                    print(f"  drx: [{actions[:, 4].min():.6f}, {actions[:, 4].max():.6f}]")
                    print(f"  dry: [{actions[:, 5].min():.6f}, {actions[:, 5].max():.6f}]")
                    print(f"  drz: [{actions[:, 6].min():.6f}, {actions[:, 6].max():.6f}]")


def main():
    """主函数"""
    print("开始处理末端位姿数据，计算位姿变化量...")
    
    # 处理所有目录
    process_all_directories()
    
    # 验证结果
    validate_action_data()
    
    print("\n处理完成！")


if __name__ == "__main__":
    main()
