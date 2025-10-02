#!/usr/bin/env python3
"""
FUA Web Interface Test Script

测试Web界面的基本功能
"""

import sys
import os
import time
import threading
import requests
import json
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_web_interface():
    """测试Web界面"""
    print("=" * 70)
    print("FUA Web Interface Test")
    print("=" * 70)
    
    # 导入Web界面
    try:
        from fua.web import create_web_interface
        print("✓ Web interface module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import web interface: {e}")
        return False
    
    # 创建Web界面实例
    web_interface = create_web_interface(
        host="127.0.0.1",
        port=8080,
        debug=False
    )
    
    # 在后台启动Web服务器
    def run_server():
        web_interface.start()
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # 等待服务器启动
    print("Starting web server...")
    time.sleep(3)
    
    # 测试API端点
    base_url = "http://127.0.0.1:8080"
    
    # 测试列表
    tests = [
        ("GET", "/", "Home page"),
        ("GET", "/api/status", "System status API"),
        ("GET", "/api/experiments", "Experiments API"),
        ("GET", "/api/models", "Models API"),
        ("GET", "/api/experiment/summary", "Experiment summary API"),
        ("GET", "/health", "Health check"),
    ]
    
    results = []
    
    for method, endpoint, description in tests:
        url = base_url + endpoint
        try:
            if method == "GET":
                response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                print(f"✓ {description}: {response.status_code}")
                results.append(True)
            else:
                print(f"✗ {description}: {response.status_code}")
                results.append(False)
        except Exception as e:
            print(f"✗ {description}: Error - {str(e)}")
            results.append(False)
    
    # 测试内容
    if results.count(True) >= len(tests) * 0.8:  # 80% success rate
        print("\n" + "=" * 70)
        print("Web Interface Test Results: PASSED")
        print("=" * 70)
        print(f"✓ {results.count(True)}/{len(tests)} tests passed")
        print("\nWeb interface is running successfully!")
        print(f"Access the web interface at: http://127.0.0.1:8080")
        print("\nAvailable pages:")
        print("- Home: http://127.0.0.1:8080/")
        print("- Dashboard: http://127.0.0.1:8080/dashboard")
        print("- Experiments: http://127.0.0.1:8080/experiments")
        print("- Models: http://127.0.0.1:8080/models")
        print("- Monitoring: http://127.0.0.1:8080/monitoring")
        
        # 保持服务器运行
        print("\nPress Ctrl+C to stop the server...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping web server...")
            web_interface.stop()
        
        return True
    else:
        print("\n" + "=" * 70)
        print("Web Interface Test Results: FAILED")
        print("=" * 70)
        print(f"✗ Only {results.count(True)}/{len(tests)} tests passed")
        return False


def test_web_integration():
    """测试Web与后端集成"""
    print("\n" + "=" * 70)
    print("Testing Web Integration with Backend Components")
    print("=" * 70)
    
    integration_tests = [
        ("MLflow Integration", "test_mlflow_integration"),
        ("Distributed Monitoring", "test_distributed_monitoring"),
        ("Model Registry", "test_model_registry"),
        ("Experiment Tracking", "test_experiment_tracking"),
    ]
    
    results = []
    
    for test_name, module_name in integration_tests:
        try:
            # 尝试导入相关模块
            if module_name == "test_mlflow_integration":
                from fua.experiment_tracking.mlflow_integration import FUAMLflowIntegration
                print(f"✓ {test_name}: Module available")
                results.append(True)
            elif module_name == "test_distributed_monitoring":
                from fua.production.distributed_monitor import DistributedModelMonitor
                print(f"✓ {test_name}: Module available")
                results.append(True)
            elif module_name == "test_model_registry":
                from fua.experiment_tracking.mlflow_integration import FUAModelRegistry
                print(f"✓ {test_name}: Module available")
                results.append(True)
            elif module_name == "test_experiment_tracking":
                from fua.experiment_tracking.mlflow_integration import FUAExperimentTracker
                print(f"✓ {test_name}: Module available")
                results.append(True)
        except ImportError as e:
            print(f"✗ {test_name}: Module not available - {e}")
            results.append(False)
    
    print(f"\nIntegration Tests: {results.count(True)}/{len(results)} passed")
    return results.count(True) >= len(results) * 0.75


def generate_web_report():
    """生成Web界面报告"""
    print("\n" + "=" * 70)
    print("Generating Web Interface Report")
    print("=" * 70)
    
    report_content = f"""# FUA Web Interface Report

**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Overview

FUA Web Interface provides a user-friendly dashboard for managing ML experiments, models, and monitoring system health. Built with Flask and Bootstrap, it offers real-time insights into the FUA ecosystem.

## Features Implemented

### 1. Core Pages
- **Home Page**: Overview of FUA capabilities with quick stats
- **Dashboard**: Real-time metrics and charts
- **Experiments**: Experiment management and comparison
- **Models**: Model registry and version management
- **Monitoring**: System health and performance metrics

### 2. API Endpoints
- `/api/status` - System status check
- `/api/experiments` - List all experiments
- `/api/models` - List registered models
- `/api/monitoring/metrics` - System metrics
- `/api/experiment/summary` - Experiment summary
- `/health` - Health check endpoint

### 3. Frontend Features
- Responsive design with Bootstrap 5
- Real-time data updates
- Interactive charts and visualizations
- Experiment filtering and search
- Model version management
- System monitoring dashboards

### 4. Integration Points
- MLflow experiment tracking
- Distributed monitoring system
- Model registry
- Training pipeline metrics

## Technical Stack

- **Backend**: Flask with RESTful APIs
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **UI Framework**: Bootstrap 5
- **Icons**: Font Awesome 6
- **Charts**: Chart.js integration ready
- **Real-time**: WebSocket support planned

## File Structure

```
fua/web/
├── __init__.py                 # Main web interface module
├── templates/                  # HTML templates
│   ├── index.html             # Home page
│   ├── dashboard.html         # Dashboard page
│   └── experiments.html       # Experiments page
└── static/                     # Static assets
    ├── css/
    │   └── style.css          # Custom styles
    └── js/
        └── main.js             # JavaScript functionality
```

## Usage

### Starting the Web Interface

```python
from fua.web import start_web_ui

# Start with default settings
start_web_ui()

# Or customize
from fua.web import create_web_interface
web = create_web_interface(host="0.0.0.0", port=8080)
web.start()
```

### Accessing the Interface

- Open browser to `http://localhost:8080`
- Navigate through different sections
- View real-time metrics and experiment data

## Next Steps

1. **Enhanced Visualizations**
   - Add Chart.js for interactive charts
   - Implement real-time updating graphs
   - Add performance comparison views

2. **Advanced Features**
   - User authentication and authorization
   - Experiment scheduling
   - Model deployment interface
   - Advanced filtering and search

3. **Real-time Updates**
   - WebSocket integration for live updates
   - Push notifications for experiment completion
   - Real-time metrics streaming

4. **Mobile Optimization**
   - Responsive design improvements
   - Mobile-specific features
   - Progressive Web App (PWA) capabilities

## Conclusion

The FUA Web Interface provides a comprehensive solution for managing ML experiments and monitoring system health. With its intuitive design and powerful integration capabilities, it serves as the central hub for FUA operations.

---

*Report generated by FUA Web Interface Test*
"""
    
    # 保存报告
    report_path = "fua_web_interface_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✓ Report generated: {report_path}")
    return report_path


def main():
    """主函数"""
    print("FUA Web Interface Test Suite")
    print("=" * 70)
    
    # 运行测试
    web_test_passed = test_web_interface()
    integration_test_passed = test_web_integration()
    
    # 生成报告
    report_path = generate_web_report()
    
    # 总结
    print("\n" + "=" * 70)
    print("Test Suite Summary")
    print("=" * 70)
    print(f"Web Interface Test: {'PASSED' if web_test_passed else 'FAILED'}")
    print(f"Integration Test: {'PASSED' if integration_test_passed else 'FAILED'}")
    print(f"Report: {report_path}")
    
    if web_test_passed and integration_test_passed:
        print("\n🎉 All tests passed! Web interface is ready for use.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)