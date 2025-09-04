#!/bin/bash
# FUA迭代平台部署脚本
# 自动化安装和配置FUA迭代平台

set -e  # 遇到错误立即退出

echo "=== FUA迭代平台部署脚本 ==="
echo "开始时间: $(date)"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Python版本
check_python() {
    log_info "检查Python版本..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        REQUIRED_VERSION="3.8"
        
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            log_info "Python版本: $PYTHON_VERSION ✓"
        else
            log_error "Python版本过低，需要3.8或更高版本，当前版本: $PYTHON_VERSION"
            exit 1
        fi
    else
        log_error "未找到Python3，请先安装Python 3.8或更高版本"
        exit 1
    fi
}

# 检查并安装uv
check_uv() {
    log_info "检查uv包管理器..."
    if command -v uv &> /dev/null; then
        log_info "uv已安装 ✓"
    else
        log_warn "uv未安装，正在安装..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source $HOME/.cargo/env
        log_info "uv安装完成 ✓"
    fi
}

# 创建虚拟环境
create_venv() {
    log_info "创建虚拟环境..."
    if [ ! -d ".venv" ]; then
        uv venv
        log_info "虚拟环境创建成功 ✓"
    else
        log_info "虚拟环境已存在 ✓"
    fi
}

# 激活虚拟环境
activate_venv() {
    log_info "激活虚拟环境..."
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
        log_info "虚拟环境已激活 ✓"
    elif [ -f ".venv/Scripts/activate" ]; then
        source .venv/Scripts/activate
        log_info "虚拟环境已激活 ✓"
    else
        log_error "无法找到虚拟环境激活脚本"
        exit 1
    fi
}

# 安装依赖
install_dependencies() {
    log_info "安装依赖包..."
    
    # 升级pip
    uv pip install --upgrade pip
    
    # 安装基础依赖
    uv pip install -r requirements.txt
    
    # 安装额外依赖
    uv pip install matplotlib seaborn scikit-learn pandas tqdm
    
    log_info "依赖安装完成 ✓"
}

# 创建必要目录
create_directories() {
    log_info "创建必要目录..."
    
    mkdir -p fua/workflows
    mkdir -p fua/validation_results
    mkdir -p fua/parameter_history
    mkdir -p experiments
    mkdir -p models
    
    log_info "目录创建完成 ✓"
}

# 下载示例数据集（可选）
download_sample_data() {
    log_info "检查示例数据集..."
    
    if [ ! -d "bioast_dataset" ]; then
        log_warn "未找到bioast_dataset目录"
        read -p "是否要创建示例数据集结构？(y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "创建示例数据集结构..."
            mkdir -p bioast_dataset/train/{positive,negative}
            mkdir -p bioast_dataset/val/{positive,negative}
            mkdir -p bioast_dataset/test/{positive,negative}
            
            # 创建一些示例文件（实际使用时需要替换为真实数据）
            for i in {1..5}; do
                touch bioast_dataset/train/positive/sample_${i}.jpg
                touch bioast_dataset/train/negative/sample_${i}.jpg
                touch bioast_dataset/val/positive/sample_${i}.jpg
                touch bioast_dataset/val/negative/sample_${i}.jpg
                touch bioast_dataset/test/positive/sample_${i}.jpg
                touch bioast_dataset/test/negative/sample_${i}.jpg
            done
            
            log_info "示例数据集创建完成 ✓"
        fi
    else
        log_info "数据集目录已存在 ✓"
    fi
}

# 运行测试
run_tests() {
    log_info "运行测试套件..."
    
    if python fua/tests/run_tests.py unit; then
        log_info "所有测试通过 ✓"
    else
        log_warn "部分测试失败，但继续部署..."
    fi
}

# 创建配置文件
create_config() {
    log_info "创建配置文件..."
    
    cat > fua_config.json << EOF
{
  "dataset_path": "bioast_dataset",
  "models_path": "models",
  "experiments_path": "experiments",
  "workflow_storage_path": "fua/workflows",
  "validation_results_path": "fua/validation_results",
  "parameter_history_path": "fua/parameter_history",
  "gpu_available": $([ -n "$CUDA_VISIBLE_DEVICES" ] && echo "true" || echo "false"),
  "max_concurrent_jobs": 1,
  "default_parameters": {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 50,
    "optimizer": "adam",
    "weight_decay": 0.0001
  },
  "bmad_config": {
    "max_iterations": 10,
    "improvement_threshold": 0.02,
    "auto_decision": true,
    "validation_datasets": {
      "primary": "bioast_dataset/test",
      "external": "external_validation_set"
    }
  }
}
EOF
    
    log_info "配置文件创建完成 ✓"
}

# 创建启动脚本
create_startup_scripts() {
    log_info "创建启动脚本..."
    
    # Web界面启动脚本
    cat > start_web.sh << 'EOF'
#!/bin/bash
source .venv/bin/activate
export PYTHONPATH=.
python fua/web/app.py
EOF
    
    # 工作流示例脚本
    cat > run_bmad_example.py << 'EOF'
#!/usr/bin/env python3
"""
FUA Bmad工作流示例
"""
import sys
import os
sys.path.insert(0, '.')

from fua.bmad_workflow_engine import BmadWorkflowEngine

def main():
    print("=== FUA Bmad工作流示例 ===")
    
    # 创建工作流引擎
    engine = BmadWorkflowEngine()
    
    # 创建工作流
    workflow_id = engine.create_workflow(
        "example_workflow",
        "resnet18",
        {
            "target_accuracy": 0.90,
            "max_iterations": 3
        }
    )
    
    print(f"创建工作流: {workflow_id}")
    
    # 注意：实际运行需要准备模型和数据集
    # engine.start_workflow(workflow_id)
    
    print("示例工作流已创建")
    print("使用 engine.start_workflow(workflow_id) 启动工作流")

if __name__ == "__main__":
    main()
EOF
    
    chmod +x start_web.sh
    chmod +x run_bmad_example.py
    
    log_info "启动脚本创建完成 ✓"
}

# 显示部署摘要
show_summary() {
    echo
    echo "=== 部署完成 ==="
    echo "完成时间: $(date)"
    echo
    echo "下一步操作："
    echo "1. 激活虚拟环境: source .venv/bin/activate"
    echo "2. 启动Web界面: ./start_web.sh"
    echo "3. 运行示例: python run_bmad_example.py"
    echo "4. 查看文档: cat FUA_Iteration_Deployment_Guide.md"
    echo
    echo "重要文件："
    echo "- 配置文件: fua_config.json"
    echo "- Web界面: fua/web/app.py"
    echo "- 测试套件: fua/tests/run_tests.py"
    echo
    echo "注意事项："
    echo "- 请确保数据集位于 bioast_dataset/ 目录"
    echo "- 根据需要调整 fua_config.json 中的配置"
    echo "- 运行前请准备好相应的模型文件"
}

# 主函数
main() {
    echo "开始部署FUA迭代平台..."
    echo
    
    check_python
    check_uv
    create_venv
    activate_venv
    install_dependencies
    create_directories
    download_sample_data
    run_tests
    create_config
    create_startup_scripts
    
    show_summary
    
    echo
    log_info "部署成功！"
}

# 捕获中断信号
trap 'log_error "部署被中断"; exit 1' INT

# 运行主函数
main