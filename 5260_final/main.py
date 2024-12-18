import os
import random
import numpy as np
import torch
from maze import Maze
from system import MultiAgentSystem

def setup_environment():
    """设置环境变量和随机种子"""
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

def check_cuda():
    """检查CUDA可用性并打印设备信息"""
    if torch.cuda.is_available():
        print(f"CUDA is available. Found {torch.cuda.device_count()} GPU(s)")
        print(f"Current device: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("CUDA is not available. Using CPU instead")
        return False

def create_maze() -> Maze:
    """创建并初始化迷宫"""
    print("Generating maze...")
    maze = Maze()
    maze.generate_maze()
    return maze

def init_system(maze: Maze) -> MultiAgentSystem:
    """初始化多智能体系统"""
    print("Initializing multi-agent system...")
    return MultiAgentSystem(maze, num_agents=10)

def train_system(system: MultiAgentSystem) -> None:
    """训练多智能体系统"""
    print("Training agents...")
    system.train()
    
    print("Visualizing exploration process...")
    system.visualize_exploration_process()
    
    print("Training completed!")

def main():
    """主程序入口点"""
    # 设置环境
    setup_environment()
    
    # 检查CUDA
    use_cuda = check_cuda()
    
    try:
        # 创建迷宫环境
        maze = create_maze()
        
        # 初始化多智能体系统
        system = init_system(maze)
        
        # 训练系统
        train_system(system)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    finally:
        print("\nProgram finished")

if __name__ == "__main__":
    main()