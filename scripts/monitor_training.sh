#!/bin/bash
# 训练进度实时监控脚本

VERSION=$1

if [ -z "$VERSION" ]; then
    echo "用法: $0 <version>"
    echo "例如: $0 v1.1"
    exit 1
fi

EXPERIMENT_DIR="/home/aaa/ws/bioastModel/experiments/mobilenetv4_$VERSION"
LOG_FILE="$EXPERIMENT_DIR/training.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "错误: 训练日志不存在: $LOG_FILE"
    exit 1
fi

# 清屏
clear

echo "=========================================="
echo "MobileNetV4 $VERSION 训练监控"
echo "=========================================="
echo ""

# 提取最新的epoch信息
echo "=== 最近训练进度 ==="
grep -E "Epoch [0-9]+/[0-9]+" "$LOG_FILE" | tail -20

echo ""
echo "=== 最佳性能 ==="
grep "best:" "$LOG_FILE" | tail -5

echo ""
echo "=== 实时日志 (按Ctrl+C退出) ==="
tail -f "$LOG_FILE"
