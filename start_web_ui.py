#!/usr/bin/env python3
"""
FUA Web Interface Launcher

启动FUA Web界面的简单脚本
"""

import sys
import os
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description='FUA Web Interface Launcher')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print("Starting FUA Web Interface...")
    print(f"Access at: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        from fua.web import start_web_ui
        start_web_ui(host=args.host, port=args.port, debug=args.debug)
    except ImportError as e:
        print(f"Error: Failed to import web interface - {e}")
        print("Please ensure all dependencies are installed")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)