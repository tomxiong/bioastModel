"""BioAst模型管理系统主入口

提供完整的模型生命周期管理解决方案。
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.integration import ModelLifecycleManager
from utils.config import ConfigManager
from utils.logger import setup_logger, get_logger
from utils.helpers import generate_model_id, format_timestamp


class BioAstModelSystem:
    """BioAst模型管理系统"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化系统
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config_manager = ConfigManager()
        if config_path:
            self.config = self.config_manager.load_config(config_path)
        else:
            self.config = self.config_manager.get_default_config()
        
        # 设置日志
        setup_logger(
            log_level=self.config.base.log_level,
            log_dir=self.config.base.log_dir
        )
        self.logger = get_logger(__name__)
        
        # 初始化生命周期管理器
        self.lifecycle_manager = ModelLifecycleManager(self.config)
        
        self.logger.info("BioAst模型管理系统初始化完成")
    
    def start_services(self):
        """启动所有服务"""
        self.logger.info("启动系统服务...")
        self.lifecycle_manager.start_services()
        self.logger.info("所有服务已启动")
    
    def stop_services(self):
        """停止所有服务"""
        self.logger.info("停止系统服务...")
        self.lifecycle_manager.stop_services()
        self.logger.info("所有服务已停止")
    
    def create_new_model_workflow(self, model_config: Dict[str, Any]) -> str:
        """创建新模型工作流
        
        Args:
            model_config: 模型配置
        
        Returns:
            工作流ID
        """
        self.logger.info(f"创建新模型工作流: {model_config.get('name', 'Unknown')}")
        
        # 生成模型ID
        model_id = generate_model_id()
        model_config['model_id'] = model_id
        
        # 创建训练工作流
        workflow_id = self.lifecycle_manager.create_training_workflow(
            model_config=model_config,
            data_config=model_config.get('data_config', {}),
            training_config=model_config.get('training_config', {})
        )
        
        self.logger.info(f"新模型工作流已创建: {workflow_id}")
        return workflow_id
    
    def execute_model_training(self, workflow_id: str) -> bool:
        """执行模型训练
        
        Args:
            workflow_id: 工作流ID
        
        Returns:
            是否执行成功
        """
        self.logger.info(f"执行模型训练工作流: {workflow_id}")
        
        try:
            # 执行工作流
            success = self.lifecycle_manager.execute_workflow(workflow_id)
            
            if success:
                self.logger.info(f"模型训练工作流执行成功: {workflow_id}")
            else:
                self.logger.error(f"模型训练工作流执行失败: {workflow_id}")
            
            return success
        
        except Exception as e:
            self.logger.error(f"执行模型训练时发生错误: {e}")
            return False
    
    def compare_models(self, model_ids: List[str]) -> str:
        """比较多个模型
        
        Args:
            model_ids: 模型ID列表
        
        Returns:
            比较报告路径
        """
        self.logger.info(f"比较模型: {model_ids}")
        
        try:
            # 生成比较报告
            report_path = self.lifecycle_manager.generate_comparison_report(
                model_ids=model_ids,
                output_format='html'
            )
            
            self.logger.info(f"模型比较报告已生成: {report_path}")
            return report_path
        
        except Exception as e:
            self.logger.error(f"生成模型比较报告时发生错误: {e}")
            return ""
    
    def get_model_registry_status(self) -> Dict[str, Any]:
        """获取模型注册表状态
        
        Returns:
            注册表状态信息
        """
        try:
            models = self.lifecycle_manager.list_models()
            experiments = self.lifecycle_manager.list_experiments()
            
            return {
                'total_models': len(models),
                'total_experiments': len(experiments),
                'models': models[:10],  # 最近10个模型
                'experiments': experiments[:10],  # 最近10个实验
                'timestamp': format_timestamp()
            }
        
        except Exception as e:
            self.logger.error(f"获取注册表状态时发生错误: {e}")
            return {}
    
    def generate_system_report(self) -> str:
        """生成系统报告
        
        Returns:
            报告路径
        """
        self.logger.info("生成系统报告")
        
        try:
            # 生成注册表报告
            report_path = self.lifecycle_manager.generate_registry_report(
                output_format='html'
            )
            
            self.logger.info(f"系统报告已生成: {report_path}")
            return report_path
        
        except Exception as e:
            self.logger.error(f"生成系统报告时发生错误: {e}")
            return ""


def create_sample_model_config() -> Dict[str, Any]:
    """创建示例模型配置
    
    Returns:
        示例模型配置
    """
    return {
        'name': 'SampleBioModel',
        'description': '示例生物信息学模型',
        'model_type': 'classification',
        'algorithm': 'random_forest',
        'version': '1.0.0',
        'author': 'BioAst Team',
        'tags': ['biology', 'classification', 'sample'],
        
        'data_config': {
            'data_path': 'data/sample_data.csv',
            'target_column': 'label',
            'feature_columns': None,  # 自动检测
            'test_size': 0.2,
            'validation_size': 0.1,
            'random_state': 42
        },
        
        'training_config': {
            'hyperparameters': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            },
            'cross_validation': {
                'enabled': True,
                'folds': 5,
                'scoring': 'accuracy'
            },
            'early_stopping': {
                'enabled': False
            }
        },
        
        'evaluation_config': {
            'metrics': ['accuracy', 'precision', 'recall', 'f1_score'],
            'generate_plots': True,
            'save_predictions': True
        }
    }


def solution_1_basic_pipeline():
    """解决方案1: 基础模型管道
    
    适用于简单的模型训练和管理需求。
    """
    print("\n=== 解决方案1: 基础模型管道 ===")
    print("特点:")
    print("- 简单易用的模型训练流程")
    print("- 基础的实验跟踪")
    print("- 标准化的模型注册")
    print("- 基本的报告生成")
    
    print("\n适用场景:")
    print("- 小团队或个人研究")
    print("- 模型数量较少")
    print("- 对自动化要求不高")
    
    print("\n实现步骤:")
    print("1. 初始化系统")
    print("2. 配置模型参数")
    print("3. 执行训练工作流")
    print("4. 生成分析报告")
    
    # 示例代码
    print("\n示例代码:")
    print("""
# 初始化系统
system = BioAstModelSystem()
system.start_services()

# 创建模型配置
model_config = create_sample_model_config()

# 创建并执行训练工作流
workflow_id = system.create_new_model_workflow(model_config)
success = system.execute_model_training(workflow_id)

# 生成报告
if success:
    report_path = system.generate_system_report()
    print(f"报告已生成: {report_path}")
    """)


def solution_2_automated_pipeline():
    """解决方案2: 自动化模型管道
    
    适用于需要自动化和批量处理的场景。
    """
    print("\n=== 解决方案2: 自动化模型管道 ===")
    print("特点:")
    print("- 全自动化的模型训练流程")
    print("- 智能的超参数优化")
    print("- 自动化的模型比较和选择")
    print("- 定时任务和批处理")
    print("- 完整的版本控制")
    
    print("\n适用场景:")
    print("- 中大型团队")
    print("- 需要处理多个模型")
    print("- 对自动化要求较高")
    print("- 需要定期重训练模型")
    
    print("\n核心组件:")
    print("- 工作流自动化引擎")
    print("- 任务调度器")
    print("- 模型版本控制")
    print("- 性能监控")
    
    print("\n实现步骤:")
    print("1. 配置自动化规则")
    print("2. 设置定时任务")
    print("3. 启动监控服务")
    print("4. 配置报告生成")


def solution_3_enterprise_platform():
    """解决方案3: 企业级模型平台
    
    适用于大型企业和复杂的模型管理需求。
    """
    print("\n=== 解决方案3: 企业级模型平台 ===")
    print("特点:")
    print("- 完整的MLOps流程")
    print("- 多用户权限管理")
    print("- 分布式训练支持")
    print("- 模型部署和服务")
    print("- 高级监控和告警")
    print("- 数据血缘追踪")
    
    print("\n适用场景:")
    print("- 大型企业")
    print("- 多团队协作")
    print("- 生产环境部署")
    print("- 严格的合规要求")
    
    print("\n核心功能:")
    print("- Web界面管理")
    print("- API接口")
    print("- 容器化部署")
    print("- 集群管理")
    print("- 安全审计")
    
    print("\n技术架构:")
    print("- 微服务架构")
    print("- 容器编排")
    print("- 消息队列")
    print("- 分布式存储")
    print("- 负载均衡")


def solution_4_ai_human_collaborative():
    """解决方案4: AI-人类协作平台
    
    专门设计用于AI和人类共同管控的场景。
    """
    print("\n=== 解决方案4: AI-人类协作平台 ===")
    print("特点:")
    print("- 双重接口设计 (AI结构化 + 人类可视化)")
    print("- 智能决策建议")
    print("- 人工审核节点")
    print("- 异常自动检测和人工确认")
    print("- 完整的操作审计")
    
    print("\n AI接口特性:")
    print("- JSON格式的结构化数据")
    print("- RESTful API")
    print("- 自动化工作流")
    print("- 机器可读的状态信息")
    
    print("\n 人类接口特性:")
    print("- 直观的Web仪表板")
    print("- 丰富的可视化图表")
    print("- Markdown格式报告")
    print("- 交互式操作界面")
    
    print("\n 协作机制:")
    print("- AI自动执行常规任务")
    print("- 人工审核关键决策")
    print("- 异常情况人工介入")
    print("- 学习反馈循环")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='BioAst模型管理系统')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--mode', type=str, choices=['demo', 'solutions', 'run'], 
                       default='solutions', help='运行模式')
    parser.add_argument('--solution', type=int, choices=[1, 2, 3, 4], 
                       help='解决方案编号')
    
    args = parser.parse_args()
    
    if args.mode == 'solutions':
        print("BioAst模型管理系统 - 解决方案概览")
        print("=" * 50)
        
        if args.solution:
            if args.solution == 1:
                solution_1_basic_pipeline()
            elif args.solution == 2:
                solution_2_automated_pipeline()
            elif args.solution == 3:
                solution_3_enterprise_platform()
            elif args.solution == 4:
                solution_4_ai_human_collaborative()
        else:
            solution_1_basic_pipeline()
            solution_2_automated_pipeline()
            solution_3_enterprise_platform()
            solution_4_ai_human_collaborative()
            
            print("\n=== 推荐选择 ===")
            print("根据您的需求选择合适的解决方案:")
            print("- 个人/小团队研究: 解决方案1")
            print("- 中型团队自动化: 解决方案2")
            print("- 企业级部署: 解决方案3")
            print("- AI-人类协作: 解决方案4")
    
    elif args.mode == 'demo':
        print("运行演示...")
        
        # 初始化系统
        system = BioAstModelSystem(args.config)
        system.start_services()
        
        try:
            # 创建示例模型
            model_config = create_sample_model_config()
            workflow_id = system.create_new_model_workflow(model_config)
            
            print(f"创建的工作流ID: {workflow_id}")
            
            # 获取系统状态
            status = system.get_model_registry_status()
            print(f"系统状态: {status}")
            
        finally:
            system.stop_services()
    
    elif args.mode == 'run':
        print("启动系统...")
        
        # 初始化并启动系统
        system = BioAstModelSystem(args.config)
        system.start_services()
        
        print("系统已启动，按Ctrl+C停止")
        
        try:
            # 保持运行
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n正在停止系统...")
            system.stop_services()
            print("系统已停止")


if __name__ == '__main__':
    main()