"""
FUA 推理服务器演示

展示如何使用 FUA 的高性能推理服务器
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fua
import tempfile
import time
import asyncio
import aiohttp
import json
from pathlib import Path


async def demo_inference_server():
    """演示推理服务器功能"""
    print("FUA 推理服务器演示")
    print("=" * 50)
    
    if not fua.DEPLOYMENT_AVAILABLE:
        print("❌ 部署模块不可用")
        return
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    print(f"使用临时目录: {temp_dir}")
    
    try:
        # 1. 创建并导出模型
        print("\n1. 创建并导出模型...")
        model_manager = fua.ModelManager()
        
        # 创建模型
        model_id = model_manager.create_model('mic_mobilenetv3', {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 5
        })
        
        model = model_manager.get_model(model_id)
        print(f"   ✓ 创建模型: {model_id}")
        
        # 导出为 ONNX
        exporter = fua.create_onnx_exporter()
        model_path = os.path.join(temp_dir, 'demo_model.onnx')
        
        success = exporter.export_model(
            model,
            model_path,
            optimization_level='advanced'
        )
        
        if not success:
            print("   ❌ 模型导出失败")
            return
        
        print("   ✓ 模型导出成功")
        
        # 2. 启动推理服务器
        print("\n2. 启动推理服务器...")
        from fua.deployment import create_inference_server
        
        # 创建服务器（但不运行，我们使用单独的进程）
        server = create_inference_server(max_models=5)
        
        # 3. 测试服务器 API
        print("\n3. 测试服务器 API...")
        
        # 模拟 API 调用（实际使用时需要启动服务器）
        print("\n   API 使用示例:")
        print("   ──────────────────────────────────────────────────")
        
        # 加载模型
        print("   POST /load_model")
        load_data = {
            "model_name": "demo_model",
            "model_path": model_path
        }
        print(f"   请求: {json.dumps(load_data, indent=2)}")
        print("   响应: {'message': 'Model demo_model loaded successfully'}")
        print()
        
        # 单个推理
        print("   POST /predict")
        predict_data = {
            "model_name": "demo_model",
            "input_data": [[0.1] * 70 * 70 * 3],  # 模拟 70x70x3 输入
            "threshold": 0.5
        }
        print(f"   请求: {json.dumps(predict_data, indent=2)}")
        print("   响应: {")
        print("     'predictions': [0.2, 0.8],")
        print("     'confidence': 0.8,")
        print("     'processing_time': 0.012,")
        print("     'request_id': 'req_1234567890'")
        print("   }")
        print()
        
        # 批量推理
        print("   POST /predict/batch")
        batch_data = {
            "model_name": "demo_model",
            "inputs": [
                [0.1] * 70 * 70 * 3,
                [0.2] * 70 * 70 * 3,
                [0.3] * 70 * 70 * 3
            ],
            "threshold": 0.5
        }
        print(f"   请求（部分）: 批量处理 3 个输入")
        print("   响应: {")
        print("     'results': [...],")
        print("     'total_time': 0.035,")
        print("     'average_time': 0.012,")
        print("     'throughput': 85.7")
        print("   }")
        print()
        
        # 获取模型列表
        print("   GET /models")
        print("   响应: {")
        print("     'models': [{")
        print("       'name': 'demo_model',")
        print("       'input_shape': [1, 3, 70, 70],")
        print("       'output_shape': [1, 2],")
        print("       'providers': ['CPUExecutionProvider']")
        print("     }],")
        print("     'count': 1,")
        print("     'max_capacity': 5")
        print("   }")
        print()
        
        # 获取性能指标
        print("   GET /metrics")
        print("   响应: {")
        print("     'total_requests': 100,")
        print("     'average_latency': 0.015,")
        print("     'p95_latency': 0.025,")
        print("     'p99_latency': 0.035,")
        print("     'throughput': 65.2,")
        print("     'error_rate': 0.0")
        print("   }")
        print()
        
        # 4. 性能测试模拟
        print("\n4. 性能特性:")
        print("   ✓ 异步处理 - 支持并发请求")
        print("   ✓ 批量推理 - 提高吞吐量")
        print("   ✓ 模型热加载 - 无需重启")
        print("   ✓ 内存管理 - LRU 模型卸载")
        print("   ✓ 性能监控 - 实时指标收集")
        print("   ✓ CORS 支持 - 跨域访问")
        print("   ✓ 健康检查 - 服务监控")
        print("   ✓ 模型预热 - 减少首次推理延迟")
        
        # 5. 部署建议
        print("\n5. 生产部署建议:")
        print("   - 使用 Gunicorn 或 Uvicorn 多 worker 模式")
        print("   - 配置负载均衡器")
        print("   - 设置适当的 max_models 限制")
        print("   - 启用访问日志和监控")
        print("   - 配置适当的超时时间")
        print("   - 使用 CDN 分发模型文件")
        
        # 6. 启动命令示例
        print("\n6. 启动服务器命令:")
        print("   # 基本启动")
        print("   python -m fua.deployment.inference_server")
        print()
        print("   # 多 worker 启动")
        print("   gunicorn -w 4 -k uvicorn.workers.UvicornWorker \\")
        print("            fua.deployment.inference_server:create_inference_server()")
        print()
        print("   # 自定义配置")
        print("   server = create_inference_server(max_models=20, metrics_window=5000)")
        print("   server.run(host='0.0.0.0', port=8000, workers=4)")
        
    except Exception as e:
        print(f"\n演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理临时目录
        import shutil
        shutil.rmtree(temp_dir)
        print(f"\n清理临时目录: {temp_dir}")


def demo_server_features():
    """演示服务器特性"""
    print("\n" + "=" * 50)
    print("推理服务器高级特性")
    print("=" * 50)
    
    features = [
        {
            "name": "异步处理",
            "description": "使用 FastAPI 的异步特性，支持高并发",
            "benefit": "提高吞吐量，减少响应时间"
        },
        {
            "name": "批量推理",
            "description": "单个请求处理多个输入",
            "benefit": "减少网络开销，提高 GPU 利用率"
        },
        {
            "name": "模型热加载",
            "description": "运行时动态加载/卸载模型",
            "benefit": "零停机更新模型"
        },
        {
            "name": "内存管理",
            "description": "LRU 策略自动卸载不常用模型",
            "benefit": "有效控制内存使用"
        },
        {
            "name": "性能监控",
            "description": "实时收集延迟、吞吐量、错误率等指标",
            "benefit": "便于性能优化和问题排查"
        },
        {
            "name": "模型预热",
            "description": "首次加载后执行预热推理",
            "benefit": "减少首次推理延迟"
        },
        {
            "name": "健康检查",
            "description": "提供服务健康状态接口",
            "benefit": "便于容器编排和监控"
        },
        {
            "name": "文件上传",
            "description": "支持直接上传 ONNX 模型文件",
            "benefit": "简化模型部署流程"
        }
    ]
    
    print("\n特性详情:")
    print("-" * 80)
    for i, feature in enumerate(features, 1):
        print(f"\n{i}. {feature['name']}")
        print(f"   描述: {feature['description']}")
        print(f"   优势: {feature['benefit']}")
    
    print("\n" + "-" * 80)
    print("\nAPI 端点:")
    endpoints = [
        ("GET /", "服务器信息"),
        ("GET /health", "健康检查"),
        ("POST /predict", "单个推理"),
        ("POST /predict/batch", "批量推理"),
        ("POST /load_model", "加载模型"),
        ("DELETE /models/{name}", "卸载模型"),
        ("GET /models", "模型列表"),
        ("GET /models/{name}", "模型详情"),
        ("POST /models/{name}/warmup", "模型预热"),
        ("GET /metrics", "性能指标"),
        ("POST /upload_model", "上传模型")
    ]
    
    for method, endpoint in endpoints:
        print(f"   {method:<20} {endpoint}")
    
    print("\n" + "-" * 80)
    print("\n性能指标:")
    metrics = [
        "total_requests - 总请求数",
        "average_latency - 平均延迟",
        "p95_latency - 95分位延迟",
        "p99_latency - 99分位延迟",
        "throughput - 吞吐量（请求/秒）",
        "error_rate - 错误率",
        "model_stats - 各模型统计信息"
    ]
    
    for metric in metrics:
        print(f"   • {metric}")


def main():
    """主函数"""
    print("FUA 推理服务器功能演示")
    print("这个演示展示了 FUA 的高性能推理服务器功能")
    
    # 基本演示
    asyncio.run(demo_inference_server())
    
    # 特性演示
    demo_server_features()
    
    print("\n" + "=" * 50)
    print("演示完成！")
    print("\n关键能力:")
    print("- ✓ 高性能异步推理服务器")
    print("- ✓ 完整的模型生命周期管理")
    print("- ✓ 实时性能监控和指标")
    print("- ✓ 生产级部署特性")
    print("- ✓ 易于使用的 RESTful API")
    print("=" * 50)


if __name__ == "__main__":
    main()