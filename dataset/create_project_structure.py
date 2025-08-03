import os

def create_project_structure():
    """创建项目目录结构"""
    
    directories = [
        "models",
        "data", 
        "training",
        "evaluation",
        "utils",
        "configs",
        "results",
        "results/checkpoints",
        "results/logs",
        "results/plots",
        "results/reports"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")
    
    # 创建__init__.py文件
    init_files = [
        "models/__init__.py",
        "data/__init__.py", 
        "training/__init__.py",
        "evaluation/__init__.py",
        "utils/__init__.py"
    ]
    
    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write("# Package initialization\n")
        print(f"创建文件: {init_file}")
    
    print("\n项目结构创建完成!")

if __name__ == "__main__":
    create_project_structure()