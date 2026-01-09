#!/bin/bash

# 顺序执行训练脚本
# 使用 && 确保前一个成功后才执行下一个

echo "=== 开始训练任务序列 ==="
echo "时间: $(date)"

# 训练 rot_arc
echo ">>> 1/3 训练 rot_arc..."
python train_rot_arc.py
if [ $? -eq 0 ]; then
    echo "✓ rot_arc 训练完成"
else
    echo "✗ rot_arc 训练失败，停止后续任务"
    exit 1
fi

# 训练 fix_line
echo ">>> 2/3 训练 fix_line..."
python train_fix_line.py
if [ $? -eq 0 ]; then
    echo "✓ fix_line 训练完成"
else
    echo "✗ fix_line 训练失败，停止后续任务"
    exit 1
fi

# 训练 rot_line
echo ">>> 3/3 训练 rot_line..."
python train_rot_line.py
if [ $? -eq 0 ]; then
    echo "✓ rot_line 训练完成"
else
    echo "✗ rot_line 训练失败"
    exit 1
fi

echo "=== 所有训练任务完成 ==="
echo "时间: $(date)"
