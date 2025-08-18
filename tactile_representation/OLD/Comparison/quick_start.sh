#!/bin/bash

# 触觉表征学习快速启动脚本

echo "=========================================="
echo "触觉表征学习项目快速启动"
echo "=========================================="

# 检查Python环境
echo "检查Python环境..."
python --version

# 检查依赖包
echo "检查关键依赖..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import numpy; print(f'NumPy版本: {numpy.__version__}')"

# 创建输出目录
echo "创建输出目录..."
mkdir -p outputs/vqvae
mkdir -p outputs/mae  
mkdir -p outputs/byol

# 运行测试
echo "运行模型测试..."
python test_models.py

if [ $? -eq 0 ]; then
    echo "✅ 测试通过！"
    echo ""
    echo "可用的训练命令:"
    echo "1. VQ-VAE: python main_train.py vqvae --data_root ./data/data25.7_aligned"
    echo "2. MAE:    python main_train.py mae --data_root ./data/data25.7_aligned"  
    echo "3. BYOL:   python main_train.py byol --data_root ./data/data25.7_aligned"
    echo ""
    echo "生成配置文件模板:"
    echo "python main_train.py vqvae --generate_config"
    echo ""
    echo "快速训练示例 (小数据集):"
    echo "python main_train.py vqvae --categories cir_lar --epochs 10 --batch_size 16"
else
    echo "❌ 测试失败，请检查环境配置"
    exit 1
fi
