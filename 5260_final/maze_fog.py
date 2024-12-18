import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque, defaultdict
import random
import time
import heapq
from tqdm import tqdm

class EnhancedMazeFog:
    def __init__(self, height=17, width=21):
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
        
        # 如果提供了当前位置，显示当前位置周围的区域
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
        
        return rgb_maze

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

    def train(self, episodes, max_steps=1000):
        """训练智能体"""
        alpha = 0.3          # 降低学习率，使学习更稳定
        gamma = 0.99         # 保持折扣因子不变
        epsilon = 0.5        # 增加初始探索率
        min_epsilon = 0.1    # 提高最小探索率
        epsilon_decay = 0.98 # 减缓探索率衰减
        batch_size = 32
        
        best_path = None
        best_reward = float('-inf')
        no_improvement_count = 0
        early_stop_count = 30  # 增加早停次数
        
        # 降低启发式移动概率
        heuristic_prob = 0.2
        
        for episode in tqdm(range(episodes), desc="Training RL with Fog of War"):
            state = self.start
            path = [state]
            episode_reward = 0
            visited = set([state])
            stuck_count = 0  # 添加卡住计数器
            
            for step in range(max_steps):
                observation = self.maze.get_local_observation(state)
                state_features = self.get_state_features(state, observation)
                valid_actions = self.get_valid_actions(state, observation)
                
                if not valid_actions:
                    break
                
                # Action selection
                if random.random() < heuristic_prob:
                    # 使用启发式选择
                    distances = []
                    for action in valid_actions:
                        dy, dx = self.actions[action]
                        ny, nx = state[0] + dy, state[1] + dx
                        next_pos = (ny, nx)
                        dist = abs(ny - self.end[0]) + abs(nx - self.end[1])
                        # 对未访问的位置给予优势
                        if next_pos not in visited:
                            dist *= 0.8
                        distances.append((action, dist))
                    action = min(distances, key=lambda x: x[1])[0]
                elif random.random() < epsilon:
                    # 随机探索
                    action = random.choice(valid_actions)
                else:
                    # 基于Q值选择
                    q_values = [self.q_table[state_features][a] for a in valid_actions]
                    max_q = max(q_values)
                    # 处理多个相同的最大Q值
                    max_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
                    action = random.choice(max_actions)
                
                dy, dx = self.actions[action]
                next_state = (state[0] + dy, state[1] + dx)
                
                # 改进的奖励机制
                reward = -0.1  # 减少基础惩罚
                if next_state == self.end:
                    reward = 1000
                    done = True
                else:
                    curr_dist = abs(state[0] - self.end[0]) + abs(state[1] - self.end[1])
                    next_dist = abs(next_state[0] - self.end[0]) + abs(next_state[1] - self.end[1])
                    
                    # 距离奖励
                    if next_dist < curr_dist:
                        reward += 2
                    elif next_dist > curr_dist:
                        reward -= 1
                    
                    # 新位置奖励
                    if next_state not in visited:
                        reward += 10
                        stuck_count = 0
                    else:
                        stuck_count += 1
                    
                    # 如果卡住太久，给予额外惩罚
                    if stuck_count > 10:
                        reward -= stuck_count * 0.5
                    
                    done = False
                
                visited.add(next_state)
                episode_reward += reward
                
                # 批量更新Q值
                self.memory.append((state_features, action, reward, 
                                  self.get_state_features(next_state, 
                                  self.maze.get_local_observation(next_state)), done))
                
                if len(self.memory) >= batch_size:
                    batch = random.sample(self.memory, batch_size)
                    for s_f, a, r, next_s_f, d in batch:
                        if not d:
                            next_obs = self.maze.get_local_observation(next_state)
                            next_valid = self.get_valid_actions(next_state, next_obs)
                            if next_valid:
                                next_max_q = max([self.q_table[next_s_f][next_a] 
                                               for next_a in next_valid])
                                target = r + gamma * next_max_q
                            else:
                                target = r
                        else:
                            target = r
                        
                        self.q_table[s_f][a] += alpha * (target - self.q_table[s_f][a])
                
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
                
                # 如果走太多步还没找到终点，提前结束
                if step > max_steps * 0.8:
                    break
            
            if no_improvement_count > early_stop_count:
                if best_path:  # 只有在找到过路径的情况下才早停
                    print(f"\nEarly stopping at episode {episode} due to no improvement")
                    break
                
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        return best_path if best_path is not None else []

def main():
    print("Starting Maze problem with RL:")
    
    # 迷宫生成
    print("\n1. Generating maze...")
    maze_size = (17, 21)  
    maze_fog = EnhancedMazeFog(*maze_size)
    maze_fog.generate_maze_prim()
    print(f"Maze size: {maze_size[0]} x {maze_size[1]}")
    print(f"Start: {maze_fog.start}, End: {maze_fog.end}")
    
    # 训练部分
    print("\n2. Training RL agent...")
    start_time = time.time()
    rl_solver_fog = RLSolverFog(maze_fog)
    rl_path_fog = rl_solver_fog.train(episodes=30)
    elapsed_time = time.time() - start_time
    
    # 计算训练时间
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    
    # 显示训练结果
    print(f"\n[Training Results]")
    print(f"Training time: {minutes} minutes {seconds:.2f} seconds")
    print(f"Path length: {len(rl_path_fog)}")
    print(f"Reached goal: {'Yes' if rl_path_fog else 'No'}")
    
    # 创建一个3x1的图，显示训练结果
    plt.figure(figsize=(20, 6))
    
    # 1. 完整迷宫
    plt.subplot(131)
    complete_maze = maze_fog.visualize_maze_with_fog()
    plt.imshow(complete_maze)
    plt.title("Complete Maze")
    plt.plot(maze_fog.start[1], maze_fog.start[0], 'go', markersize=10, label='Start')
    plt.plot(maze_fog.end[1], maze_fog.end[0], 'ro', markersize=10, label='End')
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    
    # 2. 起始视图（带迷雾）
    plt.subplot(132)
    initial_view = maze_fog.visualize_maze_with_fog(current_pos=maze_fog.start)
    plt.imshow(initial_view)
    plt.title("Initial View with Fog")
    plt.plot(maze_fog.start[1], maze_fog.start[0], 'go', markersize=10, label='Start')
    plt.plot(maze_fog.end[1], maze_fog.end[0], 'ro', markersize=10, label='End')
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    
    # 3. 最终路径
    plt.subplot(133)
    final_view = maze_fog.visualize_maze_with_fog()
    plt.imshow(final_view)
    if rl_path_fog:
        path_y, path_x = zip(*rl_path_fog)
        plt.plot(path_x, path_y, 'r-', linewidth=3, label='Path')
    plt.title(f"Solution Path\nLength: {len(rl_path_fog) if rl_path_fog else 'N/A'}")
    plt.plot(maze_fog.start[1], maze_fog.start[0], 'go', markersize=10, label='Start')
    plt.plot(maze_fog.end[1], maze_fog.end[0], 'ro', markersize=10, label='End')
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    
    plt.suptitle(f"Training Time: {minutes}m {seconds:.2f}s | Path Length: {len(rl_path_fog)}" + 
                (" (Goal Reached)" if rl_path_fog else " (Failed)"), 
                fontsize=14)
    plt.tight_layout()
    plt.show()

    # 如果找到路径，显示动态探索过程
    if rl_path_fog:
        print("\nShowing exploration animation...")
        fig, ax = plt.subplots(figsize=(10, 10))
        
        def update(frame):
            current_path = rl_path_fog[:frame+1]
            current_pos = current_path[-1]
            
            rgb_maze = maze_fog.visualize_maze_with_fog(current_pos=current_pos)
            
            ax.clear()
            ax.imshow(rgb_maze)
            
            if len(current_path) > 1:
                path_y, path_x = zip(*current_path)
                ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.5)
            
            ax.plot(current_pos[1], current_pos[0], 'o', 
                   color='yellow', markersize=15, 
                   markeredgecolor='black', markeredgewidth=2,
                   zorder=10, label='Agent')
            
            ax.plot(maze_fog.start[1], maze_fog.start[0], 'go', 
                   markersize=10, label='Start')
            ax.plot(maze_fog.end[1], maze_fog.end[0], 'ro', 
                   markersize=10, label='End')
            
            ax.set_title(f"Step {frame+1}/{len(rl_path_fog)}")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
            ax.set_xlim(-0.5, maze_fog.width - 0.5)
            ax.set_ylim(maze_fog.height - 0.5, -0.5)
            
            plt.tight_layout()
            
            return ax.get_children()

        anim = animation.FuncAnimation(fig, update, frames=len(rl_path_fog), interval=200, repeat=False)
        plt.show()

if __name__ == "__main__":
    main()