#!/usr/bin/env python3
"""
验证前向运动学计算结果
"""

import numpy as np
from joint_to_end_effector import KUKAiiwa14ForwardKinematics

def validate_forward_kinematics():
    """验证前向运动学计算结果"""
    
    # 初始化前向运动学计算器
    urdf_path = "/home/lyj/Program_python/Tactile_blind_operation/utils/KUKA_iiwa14.urdf"
    fk_calculator = KUKAiiwa14ForwardKinematics(urdf_path)
    
    # 测试零位姿
    zero_joints = np.zeros(7)
    position, orientation = fk_calculator.forward_kinematics(zero_joints)
    
    print("=== 前向运动学验证 ===")
    print(f"零位姿:")
    print(f"关节角度: {zero_joints}")
    print(f"末端位置 (x, y, z): {position}")
    print(f"末端姿态 (rx, ry, rz): {orientation}")
    print()
    
    # 加载实际数据进行验证
    sample_data = np.load("datasets/data25.7_aligned/cir_lar/20250722_152534_816_34917_50eb24f3/_end_states.npy")
    print(f"实际数据样本:")
    print(f"数据点数量: {len(sample_data)}")
    print(f"时间范围: {sample_data[0, 0]:.3f} - {sample_data[-1, 0]:.3f} 秒")
    print(f"位置范围:")
    print(f"  X: {sample_data[:, 1].min():.3f} - {sample_data[:, 1].max():.3f} m")
    print(f"  Y: {sample_data[:, 2].min():.3f} - {sample_data[:, 2].max():.3f} m")
    print(f"  Z: {sample_data[:, 3].min():.3f} - {sample_data[:, 3].max():.3f} m")
    print(f"姿态范围:")
    print(f"  RX: {sample_data[:, 4].min():.3f} - {sample_data[:, 4].max():.3f} rad")
    print(f"  RY: {sample_data[:, 5].min():.3f} - {sample_data[:, 5].max():.3f} rad")
    print(f"  RZ: {sample_data[:, 6].min():.3f} - {sample_data[:, 6].max():.3f} rad")
    print()
    
    # 检验工作空间合理性
    distances = np.sqrt(sample_data[:, 1]**2 + sample_data[:, 2]**2 + sample_data[:, 3]**2)
    print(f"末端到原点距离范围: {distances.min():.3f} - {distances.max():.3f} m")
    
    # KUKA iiwa14的大致工作半径是0.82m，检查是否合理
    if distances.max() < 1.0 and distances.min() > 0.3:
        print("✓ 工作空间范围合理")
    else:
        print("⚠ 工作空间范围可能存在问题")
    
    print("\n=== 验证完成 ===")

if __name__ == "__main__":
    validate_forward_kinematics()
