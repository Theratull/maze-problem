import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict, deque

class RLSolver:
    def __init__(self, maze_array, start, end, portals, traps):
        """初始化RL求解器
        Args:
            maze_array: 迷宫数组 (0:通道, 1:墙, 2:陷阱, 3+:传送门)
            start: 起点坐标 (y,x)
            end: 终点坐标 (y,x)
            portals: 传送门字典 {位置: 目标位置}
            traps: 陷阱位置集合
        """
        self.maze = maze_array
        self.height, self.width = maze_array.shape
        self.start = start
        self.end = end
        self.portals = portals
        self.traps = traps
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # 上右下左
        self.q_table = defaultdict(lambda: defaultdict(float))
        # 添加经验回放缓冲区
        self.memory = deque(maxlen=10000)
        
    def get_state_actions(self, state):
        """获取当前状态下的合法动作"""
        valid_actions = []
        y, x = state
        for action_idx, (dy, dx) in enumerate(self.actions):
            new_y, new_x = y + dy, x + dx
            if (0 <= new_x < self.width and 0 <= new_y < self.height 
                and self.maze[new_y, new_x] != 1):  # 不是墙就是可行的
                valid_actions.append(action_idx)
        return valid_actions

    def get_next_state(self, state, action):
        """获取执行动作后的下一个状态
        Returns:
            next_state: 下一个状态
            reward: 奖励
            done: 是否结束
        """
        dy, dx = self.actions[action]
        y, x = state
        next_state = (y + dy, x + dx)
        
        # 检查是否有效移动
        if not (0 <= next_state[0] < self.height and 0 <= next_state[1] < self.width):
            return state, -1, False
        
        # 检查是否撞墙
        if self.maze[next_state] == 1:
            return state, -1, False
            
        # 检查是否踩到陷阱
        if next_state in self.traps:
            return next_state, -100, True
            
        # 检查是否是传送门
        if next_state in self.portals:
            next_state = self.portals[next_state]
            
        # 检查是否到达终点
        if next_state == self.end:
            return next_state, 1000, True
            
        # 计算到终点的距离奖励
        curr_dist = abs(state[0] - self.end[0]) + abs(state[1] - self.end[1])
        next_dist = abs(next_state[0] - self.end[0]) + abs(next_state[1] - self.end[1])
        dist_reward = (curr_dist - next_dist) * 0.1
        
        return next_state, -0.1 + dist_reward, False

    def get_state_features(self, state):
        """获取状态的特征表示，用于泛化学习"""
        y, x = state
        features = [
            y / self.height,  # 归一化的y坐标
            x / self.width,   # 归一化的x坐标
            abs(y - self.end[0]) / self.height,  # 到终点的y距离
            abs(x - self.end[1]) / self.width,   # 到终点的x距离
        ]
        # 添加周围环境信息
        for dy, dx in self.actions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.height and 0 <= nx < self.width:
                features.append(self.maze[ny, nx] / 4)  # 归一化的格子类型
            else:
                features.append(1)  # 墙
        return np.array(features)

    def train(self, episodes, max_steps=3000):  # 减少轮次和步数
        """训练Q-learning算法并返回最佳路径"""
        alpha = 0.1          # 提高学习率
        gamma = 0.99         # 保持不变
        epsilon = 0.5        # 降低初始探索率
        min_epsilon = 0.05   # 提高最小探索率
        epsilon_decay = 0.99 # 加快探索率衰减
        batch_size = 64      # 增大批次大小
        
        best_path = None
        best_path_length = float('inf')
        no_improvement_count = 0
        
        # 提前停止条件
        early_stop_threshold = 600  
        
        for episode in tqdm(range(episodes), desc="Training Q-Learning"):
            state = self.start
            path = [state]
            episode_reward = 0
            
            for step in range(max_steps):
                valid_actions = self.get_state_actions(state)
                if not valid_actions:
                    break
                
                if random.random() < epsilon:
                    action = random.choice(valid_actions)
                else:
                    state_features = self.get_state_features(state)
                    q_values = [self.q_table[tuple(state_features)][a] for a in valid_actions]
                    action = valid_actions[np.argmax(q_values)]
                
                next_state, reward, done = self.get_next_state(state, action)
                episode_reward += reward
                
                # 存储经验
                self.memory.append((state, action, reward, next_state, done))
                
                # 经验回放学习，减少学习频率
                if len(self.memory) >= batch_size and episode % 2 == 0:  # 每两轮学习一次
                    minibatch = random.sample(self.memory, batch_size)
                    for s, a, r, next_s, d in minibatch:
                        s_features = self.get_state_features(s)
                        if not d:
                            next_valid_actions = self.get_state_actions(next_s)
                            next_s_features = self.get_state_features(next_s)
                            next_max_q = max([self.q_table[tuple(next_s_features)][next_a] 
                                        for next_a in next_valid_actions])
                            target = r + gamma * next_max_q
                        else:
                            target = r
                        
                        current_q = self.q_table[tuple(s_features)][a]
                        self.q_table[tuple(s_features)][a] = current_q + alpha * (target - current_q)
                
                path.append(next_state)
                state = next_state
                
                if done:
                    if reward > 0:  # 成功到达终点
                        if len(path) < best_path_length:
                            best_path_length = len(path)
                            best_path = path.copy()
                            no_improvement_count = 0
                        else:
                            no_improvement_count += 1
                    break
            
            # 提前停止检查
            if no_improvement_count > early_stop_threshold:
                print(f"Early stopping at episode {episode} due to no improvement")
                break
                
            # 动态调整探索率
            epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        return best_path if best_path else self._find_backup_path()
    
    def _find_backup_path(self):
        """使用BFS找到一条基准路径"""
        queue = deque([(self.start, [self.start])])
        visited = {self.start}
        
        while queue:
            current, path = queue.popleft()
            if current == self.end:
                return path
                
            for action_idx in self.get_state_actions(current):
                dy, dx = self.actions[action_idx]
                next_pos = (current[0] + dy, current[1] + dx)
                
                if next_pos not in visited:
                    next_state, _, done = self.get_next_state(current, action_idx)
                    if not done or next_state == self.end:  # 允许到达终点
                        visited.add(next_pos)
                        new_path = path + [next_pos]
                        if next_state != next_pos:  # 如果经过传送门
                            new_path.append(next_state)
                        queue.append((next_state, new_path))
        
        return [self.start]  # 如果找不到路径，至少返回起点