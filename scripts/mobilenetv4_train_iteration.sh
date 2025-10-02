#!/bin/bash
# MobileNetV4 自动化训练迭代脚本
# 训练 → 分析 → 生成报告 → 错误分析

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# 参数解析
VERSION=""
MODEL_SIZE="small"
NUM_EPOCHS=20
LEARNING_RATE=0.0015
WARMUP_EPOCHS=3
PATIENCE=15
BATCH_SIZE=64

usage() {
    echo "Usage: $0 -v VERSION [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  -v VERSION        版本号 (e.g., v1.1)"
    echo ""
    echo "Options:"
    echo "  -s MODEL_SIZE     模型大小 [small|medium|large] (default: small)"
    echo "  -e NUM_EPOCHS     训练轮数 (default: 20)"
    echo "  -l LEARNING_RATE  学习率 (default: 0.0015)"
    echo "  -w WARMUP_EPOCHS  预热轮数 (default: 3)"
    echo "  -p PATIENCE       Early stopping patience (default: 15)"
    echo "  -b BATCH_SIZE     批量大小 (default: 64)"
    echo ""
    echo "Example:"
    echo "  $0 -v v1.1"
    echo "  $0 -v v1.2 -s medium -e 30"
    exit 1
}

# 解析命令行参数
while getopts "v:s:e:l:w:p:b:h" opt; do
    case $opt in
        v) VERSION="$OPTARG" ;;
        s) MODEL_SIZE="$OPTARG" ;;
        e) NUM_EPOCHS="$OPTARG" ;;
        l) LEARNING_RATE="$OPTARG" ;;
        w) WARMUP_EPOCHS="$OPTARG" ;;
        p) PATIENCE="$OPTARG" ;;
        b) BATCH_SIZE="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

# 检查必需参数
if [ -z "$VERSION" ]; then
    print_error "缺少版本号参数！"
    usage
fi

# 设置路径
PROJECT_ROOT="/home/aaa/ws/bioastModel"
EXPERIMENT_DIR="$PROJECT_ROOT/experiments/mobilenetv4_$VERSION"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

# 检查虚拟环境
if [ ! -f "$VENV_PYTHON" ]; then
    print_error "虚拟环境不存在: $VENV_PYTHON"
    exit 1
fi

print_info "======================================================================"
print_info "MobileNetV4 自动化训练流程"
print_info "======================================================================"
print_info "版本: $VERSION"
print_info "模型大小: $MODEL_SIZE"
print_info "训练轮数: $NUM_EPOCHS"
print_info "学习率: $LEARNING_RATE"
print_info "Warmup轮数: $WARMUP_EPOCHS"
print_info "Patience: $PATIENCE"
print_info "实验目录: $EXPERIMENT_DIR"
print_info "======================================================================"

# 步骤1: 训练模型
print_info ""
print_info "步骤 1/3: 开始训练..."
print_info ""

$VENV_PYTHON scripts/multilevel_training/train_mobilenetv4.py \
    --model_size "$MODEL_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --warmup_epochs "$WARMUP_EPOCHS" \
    --patience "$PATIENCE" \
    --batch_size "$BATCH_SIZE" \
    --experiment_dir "$EXPERIMENT_DIR"

if [ $? -ne 0 ]; then
    print_error "训练失败！"
    exit 1
fi

print_info "✅ 训练完成！"

# 步骤2: 错误样本分析
print_info ""
print_info "步骤 2/3: 错误样本分析..."
print_info ""

$VENV_PYTHON scripts/mobilenetv4_error_analysis.py \
    --checkpoint "$EXPERIMENT_DIR/best_model.pth" \
    --model_size "$MODEL_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$EXPERIMENT_DIR"

if [ $? -ne 0 ]; then
    print_warning "错误分析失败（非致命错误）"
else
    print_info "✅ 错误分析完成！"
fi

# 步骤3: 生成版本对比报告
print_info ""
print_info "步骤 3/3: 生成版本对比报告..."
print_info ""

# 读取训练历史
HISTORY_FILE="$EXPERIMENT_DIR/improved_training_history.json"

if [ -f "$HISTORY_FILE" ]; then
    # 提取最佳性能
    BEST_EPOCH=$(jq -r '.val_accuracy.growth_level | length' "$HISTORY_FILE")

    print_info "训练完成于 Epoch $BEST_EPOCH"
    print_info "结果文件位于: $EXPERIMENT_DIR"

    # 列出生成的文件
    print_info ""
    print_info "生成的文件:"
    ls -lh "$EXPERIMENT_DIR" | grep -E '\.(pth|json|md|log)$' | awk '{print "  - " $9 " (" $5 ")"}'
else
    print_warning "未找到训练历史文件"
fi

print_info ""
print_info "======================================================================"
print_info "✅ 完整流程执行完成！"
print_info "======================================================================"
print_info ""
print_info "下一步操作:"
print_info "1. 查看训练报告: cat $EXPERIMENT_DIR/TRAINING_ANALYSIS_REPORT.md"
print_info "2. 查看错误分析: cat $EXPERIMENT_DIR/ERROR_ANALYSIS_REPORT.md"
print_info "3. 更新版本历史: 编辑 docs/models/MOBILENETV4_VERSION_HISTORY.md"
print_info "4. 根据分析结果制定下一版本改进方案"
print_info ""
