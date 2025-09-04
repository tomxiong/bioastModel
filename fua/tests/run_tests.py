#!/usr/bin/env python3
"""
FUA迭代平台测试运行器
运行单元测试和集成测试
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_tests(test_type="all"):
    """运行测试
    
    Args:
        test_type: 测试类型 ("unit", "integration", "all")
    """
    print(f"FUA迭代平台测试套件")
    print(f"测试类型: {test_type}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    results = {}
    
    if test_type in ["unit", "all"]:
        print("\n=== 运行单元测试 ===")
        start_time = time.time()
        
        # 导入并运行单元测试
        from fua.tests.test_fua_iteration import create_test_suite
        import unittest
        
        suite = create_test_suite()
        runner = unittest.TextTestRunner(verbosity=1)
        result = runner.run(suite)
        
        results["unit"] = {
            "time": time.time() - start_time,
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0
        }
        
        print(f"单元测试完成，耗时: {results['unit']['time']:.2f}秒")
    
    if test_type in ["integration", "all"]:
        print("\n=== 运行集成测试 ===")
        start_time = time.time()
        
        # 导入并运行集成测试
        from fua.tests.test_integration import create_integration_test_suite
        import unittest
        
        suite = create_integration_test_suite()
        runner = unittest.TextTestRunner(verbosity=1)
        result = runner.run(suite)
        
        results["integration"] = {
            "time": time.time() - start_time,
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0
        }
        
        print(f"集成测试完成，耗时: {results['integration']['time']:.2f}秒")
    
    # 生成汇总报告
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_time = 0
    
    for test_type, result in results.items():
        print(f"\n{test_type.upper()}测试:")
        print(f"  运行测试数: {result['tests_run']}")
        print(f"  失败数: {result['failures']}")
        print(f"  错误数: {result['errors']}")
        print(f"  成功率: {result['success_rate']:.1%}")
        print(f"  耗时: {result['time']:.2f}秒")
        
        total_tests += result['tests_run']
        total_failures += result['failures']
        total_errors += result['errors']
        total_time += result['time']
    
    if len(results) > 1:
        overall_success_rate = (total_tests - total_failures - total_errors) / total_tests if total_tests > 0 else 0
        print(f"\n总体结果:")
        print(f"  总测试数: {total_tests}")
        print(f"  总失败数: {total_failures}")
        print(f"  总错误数: {total_errors}")
        print(f"  总成功率: {overall_success_rate:.1%}")
        print(f"  总耗时: {total_time:.2f}秒")
    
    # 保存测试报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "test_type": test_type,
        "results": results,
        "summary": {
            "total_tests": total_tests,
            "total_failures": total_failures,
            "total_errors": total_errors,
            "overall_success_rate": (total_tests - total_failures - total_errors) / total_tests if total_tests > 0 else 0,
            "total_time": total_time
        }
    }
    
    report_file = f"fua_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细测试报告已保存到: {report_file}")
    
    # 返回成功状态
    return total_failures == 0 and total_errors == 0


def run_coverage():
    """运行代码覆盖率测试（如果coverage可用）"""
    try:
        import coverage
        print("\n=== 运行代码覆盖率测试 ===")
        
        # 创建coverage对象
        cov = coverage.Coverage(
            source=["fua"],
            omit=["*/tests/*", "*/test_*"],
            branch=True
        )
        
        # 开始收集覆盖率
        cov.start()
        
        # 运行所有测试
        success = run_tests("all")
        
        # 停止收集
        cov.stop()
        
        # 生成报告
        print("\n=== 代码覆盖率报告 ===")
        cov.report()
        
        # 生成HTML报告
        html_dir = "coverage_html"
        cov.html_report(directory=html_dir)
        print(f"HTML覆盖率报告已生成到: {html_dir}/index.html")
        
        return success
        
    except ImportError:
        print("coverage模块未安装，跳过覆盖率测试")
        print("可以使用以下命令安装: uv pip install coverage")
        return run_tests("all")


if __name__ == "__main__":
    # 解析命令行参数
    test_type = "all"
    coverage_mode = False
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ["unit", "integration", "all"]:
            test_type = sys.argv[1]
        elif sys.argv[1] == "--coverage":
            coverage_mode = True
    
    # 运行测试
    if coverage_mode:
        success = run_coverage()
    else:
        success = run_tests(test_type)
    
    # 设置退出码
    sys.exit(0 if success else 1)