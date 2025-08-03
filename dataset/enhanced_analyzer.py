import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

def enhanced_dataset_analysis():
    """增强版数据集分析，包含详细统计和可视化"""
    dataset_path = "./bioast_dataset"
    
    # 收集所有数据
    data_info = defaultdict(list)
    pixel_distributions = {'positive': [], 'negative': []}
    
    print("正在分析数据集...")
    
    for class_name in ['positive', 'negative']:
        for split in ['train', 'val', 'test']:
            split_path = os.path.join(dataset_path, class_name, split)
            if os.path.exists(split_path):
                files = [f for f in os.listdir(split_path) if f.endswith('.png')]
                
                for filename in files:
                    img_path = os.path.join(split_path, filename)
                    try:
                        img = Image.open(img_path)
                        img_array = np.array(img)
                        
                        # 收集统计信息
                        data_info['class'].append(class_name)
                        data_info['split'].append(split)
                        data_info['filename'].append(filename)
                        data_info['mean'].append(img_array.mean())
                        data_info['std'].append(img_array.std())
                        data_info['min'].append(img_array.min())
                        data_info['max'].append(img_array.max())
                        data_info['shape'].append(img_array.shape)
                        
                        # 收集像素分布（采样）
                        if len(pixel_distributions[class_name]) < 1000:  # 限制样本数量
                            pixel_distributions[class_name].extend(
                                img_array.flatten()[::10]  # 每10个像素采样一个
                            )
                            
                    except Exception as e:
                        print(f"读取图像 {img_path} 失败: {e}")
    
    # 转换为DataFrame
    df = pd.DataFrame(data_info)
    
    # 基本统计
    print("\n=== 数据集基本统计 ===")
    print(f"总图像数: {len(df)}")
    print("\n按类别和分割统计:")
    print(df.groupby(['class', 'split']).size().unstack(fill_value=0))
    
    print("\n=== 像素值统计 ===")
    stats_by_class = df.groupby('class')[['mean', 'std', 'min', 'max']].describe()
    print(stats_by_class)
    
    # 类别平衡性分析
    print("\n=== 类别平衡性分析 ===")
    class_counts = df['class'].value_counts()
    print(class_counts)
    print(f"正负样本比例: {class_counts['positive']:.0f}:{class_counts['negative']:.0f}")
    print(f"平衡度: {min(class_counts)/max(class_counts):.3f}")
    
    # 创建可视化
    create_visualizations(df, pixel_distributions)
    
    return df

def create_visualizations(df, pixel_distributions):
    """创建数据集可视化图表"""
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('生物抗菌素敏感性测试数据集分析', fontsize=16, fontweight='bold')
    
    # 1. 类别分布
    ax1 = axes[0, 0]
    class_counts = df['class'].value_counts()
    colors = ['#ff7f7f', '#7f7fff']  # 红色系表示positive，蓝色系表示negative
    bars = ax1.bar(class_counts.index, class_counts.values, color=colors, alpha=0.7)
    ax1.set_title('类别分布')
    ax1.set_ylabel('样本数量')
    for bar, count in zip(bars, class_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # 2. 分割分布
    ax2 = axes[0, 1]
    split_class = df.groupby(['split', 'class']).size().unstack()
    split_class.plot(kind='bar', ax=ax2, color=colors, alpha=0.7)
    ax2.set_title('训练/验证/测试集分布')
    ax2.set_ylabel('样本数量')
    ax2.set_xlabel('数据集分割')
    ax2.legend(title='类别')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. 像素均值分布
    ax3 = axes[0, 2]
    for i, class_name in enumerate(['positive', 'negative']):
        class_data = df[df['class'] == class_name]['mean']
        ax3.hist(class_data, bins=30, alpha=0.6, label=class_name, 
                color=colors[i], density=True)
    ax3.set_title('像素均值分布')
    ax3.set_xlabel('像素均值')
    ax3.set_ylabel('密度')
    ax3.legend()
    ax3.axvline(df[df['class']=='positive']['mean'].mean(), 
               color=colors[0], linestyle='--', alpha=0.8, label='positive均值')
    ax3.axvline(df[df['class']=='negative']['mean'].mean(), 
               color=colors[1], linestyle='--', alpha=0.8, label='negative均值')
    
    # 4. 像素标准差分布
    ax4 = axes[1, 0]
    for i, class_name in enumerate(['positive', 'negative']):
        class_data = df[df['class'] == class_name]['std']
        ax4.hist(class_data, bins=30, alpha=0.6, label=class_name, 
                color=colors[i], density=True)
    ax4.set_title('像素标准差分布')
    ax4.set_xlabel('像素标准差')
    ax4.set_ylabel('密度')
    ax4.legend()
    
    # 5. 像素值范围分析
    ax5 = axes[1, 1]
    df_melted = df.melt(id_vars=['class'], value_vars=['min', 'max'], 
                       var_name='stat', value_name='pixel_value')
    sns.boxplot(data=df_melted, x='stat', y='pixel_value', hue='class', ax=ax5)
    ax5.set_title('像素值范围分析')
    ax5.set_xlabel('统计量')
    ax5.set_ylabel('像素值')
    
    # 6. 整体像素分布对比
    ax6 = axes[1, 2]
    for i, (class_name, pixels) in enumerate(pixel_distributions.items()):
        if pixels:  # 确保有数据
            ax6.hist(pixels, bins=50, alpha=0.6, label=class_name, 
                    color=colors[i], density=True)
    ax6.set_title('整体像素分布对比')
    ax6.set_xlabel('像素值')
    ax6.set_ylabel('密度')
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 创建详细统计表
    create_detailed_stats_table(df)

def create_detailed_stats_table(df):
    """创建详细的统计表格"""
    
    print("\n=== 详细统计表 ===")
    
    # 按类别和分割的详细统计
    detailed_stats = df.groupby(['class', 'split']).agg({
        'mean': ['count', 'mean', 'std', 'min', 'max'],
        'std': ['mean', 'std'],
        'min': ['mean', 'min'],
        'max': ['mean', 'max']
    }).round(2)
    
    print("\n按类别和分割的统计:")
    print(detailed_stats)
    
    # 类别间差异分析
    print("\n=== 类别间差异分析 ===")
    pos_stats = df[df['class'] == 'positive'][['mean', 'std', 'min', 'max']].mean()
    neg_stats = df[df['class'] == 'negative'][['mean', 'std', 'min', 'max']].mean()
    
    print("Positive类别平均统计:")
    print(pos_stats.round(2))
    print("\nNegative类别平均统计:")
    print(neg_stats.round(2))
    
    print("\n类别间差异:")
    diff = pos_stats - neg_stats
    print(diff.round(2))
    
    # 数据质量评估
    print("\n=== 数据质量评估 ===")
    
    # 检查异常值
    for class_name in ['positive', 'negative']:
        class_data = df[df['class'] == class_name]
        
        # 使用IQR方法检测异常值
        Q1 = class_data['mean'].quantile(0.25)
        Q3 = class_data['mean'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = class_data[(class_data['mean'] < lower_bound) | 
                             (class_data['mean'] > upper_bound)]
        
        print(f"{class_name}类别异常值数量: {len(outliers)} ({len(outliers)/len(class_data)*100:.1f}%)")
        
        if len(outliers) > 0:
            print(f"  异常值范围: {outliers['mean'].min():.1f} - {outliers['mean'].max():.1f}")

def analyze_sample_images():
    """分析样本图像并显示"""
    dataset_path = "./bioast_dataset"
    
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    fig.suptitle('样本图像展示', fontsize=16, fontweight='bold')
    
    sample_count = 0
    for class_idx, class_name in enumerate(['positive', 'negative']):
        for split_idx, split in enumerate(['train', 'val', 'test']):
            split_path = os.path.join(dataset_path, class_name, split)
            if os.path.exists(split_path):
                files = [f for f in os.listdir(split_path) if f.endswith('.png')]
                if files:
                    # 选择第一个文件作为样本
                    sample_file = files[0]
                    img_path = os.path.join(split_path, sample_file)
                    
                    try:
                        img = Image.open(img_path)
                        img_array = np.array(img)
                        
                        ax = axes[class_idx, split_idx]
                        ax.imshow(img_array, cmap='gray')
                        ax.set_title(f'{class_name}\n{split}\n均值:{img_array.mean():.1f}')
                        ax.axis('off')
                        
                        sample_count += 1
                    except Exception as e:
                        print(f"无法显示图像 {img_path}: {e}")
    
    # 填充剩余的子图
    for i in range(sample_count, 12):
        row, col = divmod(i, 6)
        if row < 2:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("开始增强版数据集分析...")
    df = enhanced_dataset_analysis()
    
    print("\n显示样本图像...")
    analyze_sample_images()
    
    print("\n分析完成！生成的文件:")
    print("- dataset_analysis.png: 数据集统计图表")
    print("- sample_images.png: 样本图像展示")