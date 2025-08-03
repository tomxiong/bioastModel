# 生物医学菌落检测模型训练项目

## 📋 项目概述

本项目专注于70×70像素生物医学图像中的菌落检测，使用深度学习技术进行二分类任务：
- **阳性**：图像中存在菌落
- **阴性**：图像中无菌落或仅有气孔

## 🏆 模型性能排名

| 排名 | 模型 | 准确率 | 状态 |
|------|------|--------|------|
| 🥇 | AirBubble_HybridNet | 98.02% | ✅ |
| 🥈 | ResNet18-Improved | 97.83% | ✅ |
| 🥉 | EfficientNet-B0 | 97.54% | ✅ |
| 4 | MIC_MobileNetV3 | 97.45% | ✅ |
| 5 | Micro-ViT | 97.36% | ✅ |
| 6 | ConvNext-Tiny | 97.07% | ✅ |
| 7 | ViT-Tiny | 96.60% | ✅ |
| 8 | CoAtNet | 96.13% | ✅ |

## 🚀 快速开始

### 环境要求
- Python 3.8+
- PyTorch 1.9+
- CUDA支持（推荐）

### 安装依赖
```bash
pip install torch torchvision torchaudio
pip install matplotlib seaborn pandas numpy pillow
pip install scikit-learn tqdm
```

### 数据准备
将数据集放置在 `bioast_dataset/` 目录下，结构如下：
```
bioast_dataset/
├── train/
│   ├── negative/
│   └── positive/
├── val/
│   ├── negative/
│   └── positive/
└── test/
    ├── negative/
    └── positive/
```

### 训练模型
```bash
# 训练单个模型
python scripts/train_model.py --model efficientnet_b0

# 批量训练所有模型
python scripts/auto_train_sequence.py
```

### 模型评估
```bash
# 批量测试所有模型
python scripts/batch_test_models.py

# 生成错误分析
python scripts/generate_error_analysis.py
```

## 📊 项目结构

```
bioastModel/
├── core/                   # 核心模块
│   ├── config/            # 配置文件
│   ├── data/              # 数据处理
│   └── training/          # 训练逻辑
├── models/                # 模型定义
├── scripts/               # 训练和评估脚本
├── reports/               # 分析报告
└── experiments/           # 实验结果（不包含在Git中）
```

## 🔍 模型架构

### 最佳性能模型：AirBubble_HybridNet
- **准确率**: 98.02%
- **特点**: 混合架构，结合CNN和注意力机制
- **错误样本**: 仅21个

### 传统架构
- **ResNet18-Improved**: 改进版ResNet，稳定可靠
- **EfficientNet-B0**: 轻量级模型，效率与性能平衡

### Vision Transformer
- **ViT-Tiny**: 小型Vision Transformer
- **CoAtNet**: 卷积与注意力结合
- **Micro-ViT**: 微型ViT架构

## 📈 训练特性

- ✅ 8个不同架构模型完整训练
- ✅ 完整的测试结果和性能指标
- ✅ 详细的错误样本分析
- ✅ 可视化分析图表
- ✅ 模型对比和性能排名

## 📋 分析报告

详细的模型对比分析请查看：
- [简单对比总结](reports/simple_model_comparison.md)
- [综合性能分析](reports/comprehensive_performance_analysis.md)
- [技术报告](reports/comprehensive_technical_report.html)

## 🛠️ 开发工具

- **训练监控**: 实时训练进度和指标监控
- **错误分析**: 自动生成错误样本分析
- **可视化**: 性能图表和混淆矩阵
- **批量处理**: 支持批量训练和测试

## 📝 许可证

本项目仅用于学术研究和教育目的。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目。

---
*最后更新: 2025-08-03*