#!/bin/bash
#
# Quick training script for MultiLevelMobileNetV4
# 快速训练 MobileNetV4 模型的脚本
#

set -e  # Exit on error

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}MultiLevelMobileNetV4 Quick Training${NC}"
echo -e "${BLUE}========================================${NC}"

# 默认配置
MODEL_SIZE=${1:-small}  # small, medium, or large
DATA_ROOT=${2:-/home/aaa/ws/bioastModel/ds/images}
JSON_PATH=${3:-/home/aaa/ws/bioastModel/ds/images/m9e1n170.json}

echo -e "\n${GREEN}Configuration:${NC}"
echo "  Model Size: $MODEL_SIZE"
echo "  Data Root: $DATA_ROOT"
echo "  JSON Path: $JSON_PATH"

# 检查数据集
if [ ! -d "$DATA_ROOT" ]; then
    echo -e "${YELLOW}Warning: Data root directory not found: $DATA_ROOT${NC}"
    echo "Please specify correct data root as second argument"
    exit 1
fi

if [ ! -f "$JSON_PATH" ]; then
    echo -e "${YELLOW}Warning: JSON file not found: $JSON_PATH${NC}"
    echo "Please specify correct JSON path as third argument"
    exit 1
fi

# 激活虚拟环境
if [ -d ".venv" ]; then
    echo -e "\n${GREEN}Activating virtual environment...${NC}"
    source .venv/bin/activate
fi

# 创建实验目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_DIR="experiments/mobilenetv4_${MODEL_SIZE}_${TIMESTAMP}"

echo -e "\n${GREEN}Creating experiment directory:${NC}"
echo "  $EXPERIMENT_DIR"
mkdir -p "$EXPERIMENT_DIR"

# 训练参数 (基于改进版的最佳配置)
BATCH_SIZE=64
LEARNING_RATE=0.002
WEIGHT_DECAY=0.01
NUM_EPOCHS=20
WARMUP_EPOCHS=5
PATIENCE=10
DROPOUT_RATE=0.3

echo -e "\n${GREEN}Training Parameters:${NC}"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Patience: $PATIENCE"

# 开始训练
echo -e "\n${GREEN}Starting training...${NC}"
python scripts/multilevel_training/train_mobilenetv4.py \
    --model_size "$MODEL_SIZE" \
    --data_root "$DATA_ROOT" \
    --json_path "$JSON_PATH" \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --num_epochs $NUM_EPOCHS \
    --warmup_epochs $WARMUP_EPOCHS \
    --patience $PATIENCE \
    --dropout_rate $DROPOUT_RATE \
    --growth_level_weight 1.0 \
    --growth_pattern_weight 1.0 \
    --interference_weight 1.0 \
    --experiment_dir "$EXPERIMENT_DIR"

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Training completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "\n${GREEN}Results saved to:${NC} $EXPERIMENT_DIR"
echo -e "\n${GREEN}Next steps:${NC}"
echo "  1. Check training curves: $EXPERIMENT_DIR/training_history.png"
echo "  2. View test results: $EXPERIMENT_DIR/test_results.json"
echo "  3. Compare with other models: python scripts/analysis/compare_models.py"
