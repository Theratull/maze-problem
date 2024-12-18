import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict, deque

class RLSolver:
    def __init__(self, maze_array, start, end, portals, traps, movement_costs):
        """初始化RL求解器
        Args:
            maze_array: 迷宫数组
            start: 起点坐标
            end: 终点坐标
            portals: 传送门字典
            traps: 陷阱位置集合
            movement_costs: 特殊区域移动代价字典
        """
        self.maze = maze_array
        self.height, self.width = maze_array.shape
        self.start = start
        self.end = end
        self.portals = portals
        self.traps = traps
        self.movement_costs = movement_costs
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.memory = deque(maxlen=10000)
        
    def get_movement_cost(self, pos):
        """获取指定位置的移动代价"""
        max_cost = 1
        y, x = pos
        for (y1, x1, y2, x2), cost in self.movement_costs.items():
            if y1 <= y <= y2 and x1 <= x <= x2:
                max_cost = max(max_cost, cost)
        return max_cost

    def get_state_actions(self, state):
        """获取当前状态下的合法动作"""
        valid_actions = []
        y, x = state
        for action_idx, (dy, dx) in enumerate(self.actions):
            new_y, new_x = y + dy, x + dx
            if (0 <= new_x < self.width and 0 <= new_y < self.height 
                and self.maze[new_y, new_x] != 1):
                valid_actions.append(action_idx)
        return valid_actions

    def get_next_state(self, state, action):
        """获取执行动作后的下一个状态"""
        dy, dx = self.actions[action]
        y, x = state
        next_state = (y + dy, x + dx)
        
        if not (0 <= next_state[0] < self.height and 0 <= next_state[1] < self.width):
            return state, -1, False
        
        if self.maze[next_state] == 1:
            return state, -1, False
            
        if next_state in self.traps:
            return next_state, -100, True
            
        if next_state in self.portals:
            next_state = self.portals[next_state]
            
        if next_state == self.end:
            return next_state, 1000, True
        
        # 考虑移动代价的距离奖励
        move_cost = self.get_movement_cost(next_state)
        curr_dist = abs(state[0] - self.end[0]) + abs(state[1] - self.end[1])
        next_dist = abs(next_state[0] - self.end[0]) + abs(next_state[1] - self.end[1])
        dist_reward = (curr_dist - next_dist) * 0.1
        
        # 根据移动代价调整奖励
        step_penalty = -0.1 * move_cost
        
        return next_state, step_penalty + dist_reward, False

    def get_state_features(self, state):
        """获取状态的特征表示"""
        y, x = state
        features = [
            y / self.height,
            x / self.width,
            abs(y - self.end[0]) / self.height,
            abs(x - self.end[1]) / self.width,
            self.get_movement_cost((y, x))  # 添加当前位置的移动代价作为特征
        ]
        # 添加周围环境信息
        for dy, dx in self.actions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.height and 0 <= nx < self.width:
                features.append(self.maze[ny, nx] / 4)
                features.append(self.get_movement_cost((ny, nx)))  # 添加相邻位置的移动代价
            else:
                features.append(1)
                features.append(3)  # 最大移动代价
        return np.array(features)

    def train(self, episodes=1000, max_steps=3000):
        """训练Q-learning算法并返回最佳路径及其总代价"""
        alpha = 0.1
        gamma = 0.99
        epsilon = 0.5
        min_epsilon = 0.05
        epsilon_decay = 0.99
        batch_size = 64
        
        best_path = None
        best_path_cost = float('inf')
        no_improvement_count = 0
        early_stop_threshold = 600
        
        for episode in tqdm(range(episodes), desc="Training Q-Learning"):
            state = self.start
            path = [state]
            path_cost = 0
            episode_reward = 0
            
            for step in range(max_steps):
                valid_actions = self.get_state_actions(state)
                if not valid_actions:
                    break
                
                # epsilon-greedy策略选择动作
                if random.random() < epsilon:
                    action = random.choice(valid_actions)
                else:
                    state_features = self.get_state_features(state)
                    q_values = [self.q_table[tuple(state_features)][a] for a in valid_actions]
                    action = valid_actions[np.argmax(q_values)]
                
                next_state, reward, done = self.get_next_state(state, action)
                path_cost += self.get_movement_cost(next_state)
                episode_reward += reward
                
                # 存储经验
                self.memory.append((state, action, reward, next_state, done))
                
                # 经验回放学习
                if len(self.memory) >= batch_size and episode % 2 == 0:
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
                        if path_cost < best_path_cost:
                            best_path_cost = path_cost
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
        
        if best_path:
            return best_path, best_path_cost
        else:
            backup_path = self._find_backup_path()
            if backup_path:
                backup_cost = sum(self.get_movement_cost(pos) for pos in backup_path)
                return backup_path, backup_cost
            return [self.start], float('inf')
    
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
                    if not done or next_state == self.end:
                        visited.add(next_pos)
                        new_path = path + [next_pos]
                        if next_state != next_pos:
                            new_path.append(next_state)
                        queue.append((next_state, new_path))
        
        return [self.start]