#!/usr/bin/env python3
"""
KUKA iiwa14 Forward Kinematics
将关节位置数据转换为末端执行器的6轴位姿（位置[3] + 姿态[3]）
使用URDF文件进行前向运动学计算
"""

import numpy as np
import os
import glob
from pathlib import Path
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
# "logging.py" in utils will shade homonymous Standard Library of scipy


class KUKAiiwa14ForwardKinematics:
    """KUKA iiwa14 机器人前向运动学计算类"""
    
    def __init__(self, urdf_path):
        """
        初始化前向运动学计算器
        
        Args:
            urdf_path: URDF文件路径
        """
        self.urdf_path = urdf_path
        self.dh_params = self._parse_urdf()
        
    def _parse_urdf(self):
        """
        从URDF文件解析DH参数
        
        Returns:
            list: DH参数列表，每个元素包含 [a, alpha, d, theta_offset]
        """
        # KUKA iiwa14的DH参数（根据URDF文件）
        # 这些参数是从URDF文件中的关节和连杆信息提取的
        dh_params = [
            # [a, alpha, d, theta_offset]
            [0, 0, 0.1575, 0],                    # Joint 1
            [0, np.pi/2, 0.2025, np.pi/2],       # Joint 2  
            [0, -np.pi/2, 0, 0],                 # Joint 3
            [0, np.pi/2, 0.2155, 0],             # Joint 4
            [0, -np.pi/2, 0.1845, 0],            # Joint 5
            [0, np.pi/2, 0, 0],                  # Joint 6
            [0, -np.pi/2, 0.081, 0],             # Joint 7
        ]
        return dh_params
    
    def _dh_matrix(self, a, alpha, d, theta):
        """
        计算DH变换矩阵
        
        Args:
            a: 连杆长度
            alpha: 连杆扭转角
            d: 连杆偏移
            theta: 关节角度
            
        Returns:
            numpy.ndarray: 4x4变换矩阵
        """
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        
        T = np.array([
            [cos_theta, -sin_theta * cos_alpha,  sin_theta * sin_alpha, a * cos_theta],
            [sin_theta,  cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
            [0,          sin_alpha,              cos_alpha,             d],
            [0,          0,                      0,                     1]
        ])
        
        return T
    
    def forward_kinematics(self, joint_angles):
        """
        计算前向运动学
        
        Args:
            joint_angles: 7个关节角度的数组
            
        Returns:
            tuple: (position, orientation) 
                   position: [x, y, z] 位置
                   orientation: [rx, ry, rz] 欧拉角（XYZ顺序）
        """
        # 从基座坐标系开始
        T = np.eye(4)
        
        # 逐个关节计算变换矩阵
        for i, (a, alpha, d, theta_offset) in enumerate(self.dh_params):
            theta = joint_angles[i] + theta_offset
            T_i = self._dh_matrix(a, alpha, d, theta)
            T = T @ T_i
        
        # 提取位置（前3个元素是xyz坐标）
        position = T[:3, 3]
        
        # 提取旋转矩阵并转换为欧拉角
        rotation_matrix = T[:3, :3]
        
        # 使用scipy转换为欧拉角（XYZ顺序）
        r = Rotation.from_matrix(rotation_matrix)
        orientation = r.as_euler('xyz', degrees=False)
        
        return position, orientation
    
    def process_joint_data(self, joint_data):
        """
        处理关节数据，返回末端执行器位姿
        
        Args:
            joint_data: 包含时间戳和7个关节角度的数组 [timestamp, q1, q2, ..., q7]
            
        Returns:
            numpy.ndarray: [timestamp, x, y, z, rx, ry, rz]
        """
        timestamp = joint_data[0]
        joint_angles = joint_data[1:8]  # 前7个关节角度
        
        position, orientation = self.forward_kinematics(joint_angles)
        
        # 组合为 [timestamp, x, y, z, rx, ry, rz]
        end_effector_pose = np.concatenate(([timestamp], position, orientation))
        
        return end_effector_pose


def parse_lbr_joint_states_file(file_path):
    """
    解析_lbr_joint_states.txt文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        list: 每个元素为 [timestamp, q1, q2, ..., q7]
    """
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 按等号分隔符分割数据
    data_blocks = content.split('=' * 40)
    
    joint_data_list = []
    
    for block in data_blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 8:  # 至少需要时间戳 + 7个关节角度
            try:
                # 提取时间戳和7个关节角度
                values = [float(line.strip()) for line in lines[:8]]
                joint_data_list.append(values)
            except ValueError:
                # 跳过无效数据
                continue
    
    return joint_data_list


def process_directory(directory_path, fk_calculator):
    """
    处理单个目录中的_lbr_joint_states文件
    
    Args:
        directory_path: 目录路径
        fk_calculator: 前向运动学计算器实例
    """
    lbr_file_path = os.path.join(directory_path, '_lbr_joint_states.txt')
    
    if not os.path.exists(lbr_file_path):
        print(f"Warning: {lbr_file_path} not found")
        return
    
    print(f"Processing: {directory_path}")
    
    # 解析关节数据
    joint_data_list = parse_lbr_joint_states_file(lbr_file_path)
    
    if not joint_data_list:
        print(f"Warning: No valid data found in {lbr_file_path}")
        return
    
    # 计算末端执行器位姿
    end_effector_poses = []
    for joint_data in joint_data_list:
        pose = fk_calculator.process_joint_data(joint_data)
        end_effector_poses.append(pose)
    
    end_effector_poses = np.array(end_effector_poses)
    
    # 保存为txt格式
    txt_output_path = os.path.join(directory_path, '_end_states.txt')
    np.savetxt(txt_output_path, end_effector_poses, fmt='%.6f')
    
    # 保存为npy格式
    npy_output_path = os.path.join(directory_path, '_end_states.npy')
    np.save(npy_output_path, end_effector_poses)
    
    print(f"Saved: {txt_output_path}")
    print(f"Saved: {npy_output_path}")
    print(f"Processed {len(end_effector_poses)} data points\n")


def main():
    """主函数"""
    # 获取URDF文件路径
    urdf_path = "/home/lyj/Program_python/Tactile_blind_operation/utils/KUKA_iiwa14.urdf"
    
    # 初始化前向运动学计算器
    fk_calculator = KUKAiiwa14ForwardKinematics(urdf_path)
    
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
            process_directory(exp_path, fk_calculator)
            total_processed += 1
    
    print(f"\n=== Processing Complete ===")
    print(f"Total directories processed: {total_processed}")


if __name__ == "__main__":
    main()
