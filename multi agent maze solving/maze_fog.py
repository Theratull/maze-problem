import numpy as np
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import random
import time
import heapq
from tqdm import tqdm

class EnhancedMazeFog:
    def __init__(self, height=81, width=121):
        self.height = height if height % 2 == 1 else height + 1
        self.width = width if width % 2 == 1 else width + 1
        self.maze = np.ones((self.height, self.width), dtype=np.uint8)
        self.start = (1, 1)
        self.end = (self.height-2, self.width-2)
        self.vision_range = 6  # 5*5视野

    def generate_maze_prim(self):
        """使用 Prim 算法生成基本迷宫"""
        def add_walls(cell):
            for dy, dx in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                ny, nx = cell[0] + dy, cell[1] + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    if self.maze[ny, nx] == 1:
                        wall = ((cell[0] + ny) // 2, (cell[1] + nx) // 2)
                        walls.add((ny, nx, wall[0], wall[1]))

        walls = set()
        self.maze[self.start] = 0
        add_walls(self.start)

        while walls:
            ny, nx, wy, wx = random.choice(list(walls))
            walls.remove((ny, nx, wy, wx))
            
            if self.maze[ny, nx] == 1:
                self.maze[wy, wx] = 0
                self.maze[ny, nx] = 0
                add_walls((ny, nx))

        self.maze[self.end] = 0

    def get_local_observation(self, current_pos):
        """获取局部观察 (5*5区域)"""
        y, x = current_pos
        size = 2 * self.vision_range + 1  # 5*5的观察窗口
        observation = np.ones((size, size), dtype=np.uint8) * -1  # -1表示未知区域
        
        for dy in range(-self.vision_range, self.vision_range + 1):
            for dx in range(-self.vision_range, self.vision_range + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    observation[dy + self.vision_range, dx + self.vision_range] = self.maze[ny, nx]
                    
        return observation

    def visualize_maze_with_fog(self, path=None, title="Maze with Fog of War", current_pos=None):
        """可视化带有动态迷雾的迷宫"""
        # 创建RGB图像
        rgb_maze = np.zeros((self.height, self.width, 3))
        rgb_maze.fill(0.7)  # 默认为迷雾（灰色）
        
        # 如果提供了当前位置，显示当前位置周围的5*5区域
        if current_pos is not None:
            y, x = current_pos
            for dy in range(-self.vision_range, self.vision_range + 1):
                for dx in range(-self.vision_range, self.vision_range + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        if self.maze[ny, nx] == 1:
                            rgb_maze[ny, nx] = [0, 0, 0]  # 墙是黑色
                        else:
                            rgb_maze[ny, nx] = [1, 1, 1]  # 通道是白色
        
        # 绘制路径
        if path:
            path_y, path_x = zip(*path)
            plt.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7)
        
        plt.imshow(rgb_maze)
        plt.plot(self.start[1], self.start[0], 'go', markersize=10, label='Start')
        plt.plot(self.end[1], self.end[0], 'ro', markersize=10, label='End')
        
        plt.title(title)
        plt.legend()
        plt.axis('off')
        plt.tight_layout()

class RLSolverFog:
    def __init__(self, maze):
        self.maze = maze
        self.height = maze.height
        self.width = maze.width
        self.start = maze.start
        self.end = maze.end
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # 上右下左
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.memory = deque(maxlen=10000)
        self.vision_range = maze.vision_range

    def get_state_features(self, state, observation):
        """获取状态特征"""
        y, x = state
        features = [
            y / self.height,  # 归一化的y坐标
            x / self.width,   # 归一化的x坐标
            abs(y - self.end[0]) / self.height,  # 到终点的y距离
            abs(x - self.end[1]) / self.width,   # 到终点的x距离
        ]
        
        # 添加局部观察信息
        obs_flat = observation.flatten() / 4  # 归一化观察值
        features.extend(obs_flat)
                
        return tuple(features)

    def get_valid_actions(self, state, observation):
        """获取有效动作"""
        valid_actions = []
        for action_idx, (dy, dx) in enumerate(self.actions):
            y, x = state
            ny, nx = y + dy, x + dx
            
            # 检查是否在观察范围内
            if abs(dy) <= self.vision_range and abs(dx) <= self.vision_range:
                obs_y, obs_x = dy + self.vision_range, dx + self.vision_range
                # 如果在视野内且不是墙
                if observation[obs_y, obs_x] != 1:
                    valid_actions.append(action_idx)
                    
        return valid_actions

    def train(self, episodes, max_steps=2000):
        """训练智能体"""
        alpha = 0.2          # 学习率
        gamma = 0.99         # 折扣因子
        epsilon = 0.5        # 初始探索率
        min_epsilon = 0.05   # 最小探索率
        epsilon_decay = 0.99 # 探索率衰减
        batch_size = 64      # 批次大小
        
        best_path = None
        best_reward = float('-inf')
        no_improvement_count = 0
        
        for episode in tqdm(range(episodes), desc="Training RL with Fog of War"):
            state = self.start
            path = [state]
            episode_reward = 0
            
            for step in range(max_steps):
                # 获取当前观察
                observation = self.maze.get_local_observation(state)
                state_features = self.get_state_features(state, observation)
                
                valid_actions = self.get_valid_actions(state, observation)
                if not valid_actions:
                    break
                
                if random.random() < epsilon:
                    action = random.choice(valid_actions)
                else:
                    q_values = [self.q_table[state_features][a] for a in valid_actions]
                    action = valid_actions[np.argmax(q_values)]
                
                dy, dx = self.actions[action]
                next_state = (state[0] + dy, state[1] + dx)
                
                reward = -0.1  
                if next_state == self.end:
                    reward = 1000
                    done = True
                else:
                    curr_dist = abs(state[0] - self.end[0]) + abs(state[1] - self.end[1])
                    next_dist = abs(next_state[0] - self.end[0]) + abs(next_state[1] - self.end[1])
                    reward += (curr_dist - next_dist) * 0.1
                    done = False
                
                episode_reward += reward
                
                next_observation = self.maze.get_local_observation(next_state)
                next_state_features = self.get_state_features(next_state, next_observation)
                
                self.memory.append((state_features, action, reward, next_state_features, done))
                
                if len(self.memory) >= batch_size:
                    batch = random.sample(self.memory, batch_size)
                    for s_f, a, r, next_s_f, d in batch:
                        if not d:
                            next_observation = self.maze.get_local_observation(next_state)
                            next_valid_actions = self.get_valid_actions(next_state, next_observation)
                            if next_valid_actions:
                                next_max_q = max([self.q_table[next_s_f][next_a] 
                                               for next_a in next_valid_actions])
                                target = r + gamma * next_max_q
                            else:
                                target = r
                        else:
                            target = r
                        
                        self.q_table[s_f][a] = self.q_table[s_f][a] + alpha * (target - self.q_table[s_f][a])
                
                path.append(next_state)
                state = next_state
                
                if done:
                    if episode_reward > best_reward:
                        best_reward = episode_reward
                        best_path = path.copy()
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                    break
            
            if no_improvement_count > 50:
                print(f"Early stopping at episode {episode} due to no improvement")
                break
                
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        return best_path if best_path is not None else []

def main():
    print("\n" + "="*50)
    print("Starting Maze Experiment with RL")
    print("="*50)
    
    random.seed(42)
    np.random.seed(42)
    
    # Create a small maze
    print("\n1. Generating maze...")
    maze_size = (17, 21)  
    maze_fog = EnhancedMazeFog(*maze_size)
    maze_fog.generate_maze_prim()
    print(f"Maze size: {maze_size[0]} x {maze_size[1]}")
    print(f"Start: {maze_fog.start}, End: {maze_fog.end}")
    
    # RL solving
    print("\n2. Training RL agent...")
    start_time = time.time()
    rl_solver_fog = RLSolverFog(maze_fog)
    rl_path_fog = rl_solver_fog.train(episodes=60)
    rl_time = time.time() - start_time
    print(f"\n[RL Results]")
    print(f"Total time: {rl_time:.3f} seconds")
    print(f"Path length: {len(rl_path_fog)}")
    
    reached = False
    if rl_path_fog and len(rl_path_fog) > 0:
        reached = (rl_path_fog[-1] == maze_fog.end)
    
    if not reached:
        print("Failed to find valid path!")
    
    print("\n3. Results:")
    print("="*50)
    print(f"Solving time: {rl_time:.3f} seconds")
    print(f"Path length: {len(rl_path_fog)}")
    print(f"Reached goal: {'Yes' if reached else 'No'}")
    
    print("\n4. Visualizing results...")
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    maze_fog.visualize_maze_with_fog(None, "Complete Maze")
    
    plt.subplot(132)
    maze_fog.visualize_maze_with_fog(None, "View with Fog", maze_fog.start)
    
    plt.subplot(133)
    if rl_path_fog:
        maze_fog.visualize_maze_with_fog(rl_path_fog, "RL Solution Path", rl_path_fog[-1])
    else:
        maze_fog.visualize_maze_with_fog(None, "RL Solution Path (Failed)")
    
    plt.tight_layout()
    plt.show()
    
    if rl_path_fog and input("\nShow path progression? (y/n): ").lower() == 'y':
        print("\nShowing path progression...")
        plt.figure(figsize=(8, 8))
        for i, pos in enumerate(rl_path_fog):
            plt.clf()
            maze_fog.visualize_maze_with_fog(rl_path_fog[:i+1], f"Step {i+1}/{len(rl_path_fog)}", pos)
            plt.pause(0.1)
        plt.show()

if __name__ == "__main__":
    main()