import numpy as np
import torch
import random
from collections import deque
from typing import List, Tuple, Set, Dict, Optional
from torch.cuda.amp import autocast

class Agent:
    def __init__(self, maze, start_pos: Tuple[int, int], agent_id: int, device_id: int,
                 global_dead_ends: Set[Tuple[int, int]], global_visit_count: np.ndarray,
                 global_decision_points: Dict, global_unexplored_branches: Dict):
        self.maze = maze
        self.position = start_pos
        self.agent_id = agent_id
        self.device = f'cuda:{device_id}' if device_id >= 0 else 'cpu'
        
        # 全局共享记忆
        self.dead_ends = global_dead_ends
        self.visit_count = global_visit_count
        self.decision_points = global_decision_points
        self.unexplored_branches = global_unexplored_branches
        
        # 状态相关
        self.memory = deque(maxlen=10000)
        self.q_values = torch.zeros((maze.height, maze.width, 4), device=self.device)
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
        self.previous_positions = deque(maxlen=1000)  # 增加历史记录长度
        self.backtracking = False
        self.path_to_safe = []

    def select_action(self, state_tensor: torch.Tensor, epsilon: float, 
                     other_positions: List[Tuple[int, int]]) -> int:
        # 如果在回溯模式且有路径，继续回溯
        if self.backtracking and self.path_to_safe:
            next_pos = self.path_to_safe[0]
            dy = next_pos[0] - self.position[0]
            dx = next_pos[1] - self.position[1]
            for i, (action_dy, action_dx) in enumerate(self.actions):
                if (action_dy, action_dx) == (dy, dx):
                    self.path_to_safe.pop(0)
                    return i

        valid_actions = self.get_valid_actions(other_positions)
        if not valid_actions:
            return random.randint(0, 3)

        # 如果当前位置是死胡同，尝试回溯到决策点
        if self.detect_dead_end(self.position):
            self.dead_ends.add(self.position)
            if not self.backtracking:
                if self.backtrack_to_decision_point():
                    next_pos = self.path_to_safe[0]
                    dy = next_pos[0] - self.position[0]
                    dx = next_pos[1] - self.position[1]
                    for i, (action_dy, action_dx) in enumerate(self.actions):
                        if (action_dy, action_dx) == (dy, dx):
                            return i

        # 随机探索
        if random.random() < epsilon:
            return random.choice(valid_actions)

        # 计算每个动作的分数
        action_scores = []
        for action in valid_actions:
            dy, dx = self.actions[action]
            ny, nx = self.position[0] + dy, self.position[1] + dx
            next_pos = (ny, nx)
            
            score = self.q_values[self.position[0], self.position[1], action].item()

            # 考虑其他智能体的探索情况
            other_explored = False
            for other_pos in other_positions:
                if self.maze.can_communicate(next_pos, other_pos):
                    if self.maze.is_explored(next_pos):
                        score -= 200  # 惩罚已被其他智能体探索的区域
                    other_explored = True

            # 奖励未探索区域
            if not other_explored and not self.maze.is_explored(next_pos):
                score += 300

            # 死胡同惩罚
            if next_pos in self.dead_ends:
                score -= 1000
            if self.is_potential_dead_end(next_pos):
                score -= 500

            # 访问频率惩罚
            visit_penalty = self.visit_count[ny, nx] * 200
            score -= visit_penalty

            # 目标导向奖励
            distance_to_goal = self._manhattan_distance(next_pos, self.maze.end)
            score += max(0, (50 - distance_to_goal) * 10)  # 距离目标越近奖励越高

            action_scores.append((action, score))

        # 选择分数最高的动作
        return max(action_scores, key=lambda x: x[1])[0]

    def _find_last_unexplored_decision(self) -> Optional[Tuple[int, int]]:
        """找到最近的有未探索方向的决策点"""
        if not self.previous_positions:
            return None
        
        # 逆序遍历历史路径
        for pos in reversed(list(self.previous_positions)):
            unexplored_directions = []
            valid_moves = 0
            
            for dy, dx in self.actions:
                ny, nx = pos[0] + dy, pos[1] + dx
                next_pos = (ny, nx)
                
                if (0 <= ny < self.maze.height and 
                    0 <= nx < self.maze.width and 
                    self.maze.maze[ny, nx] == 0):
                    valid_moves += 1
                    # 检查该方向是否未探索且不是死胡同
                    if (not self.maze.is_explored(next_pos) and 
                        next_pos not in self.dead_ends and 
                        not self.is_potential_dead_end(next_pos)):
                        unexplored_directions.append(next_pos)
            
            # 如果这是一个有未探索方向的决策点
            if len(unexplored_directions) > 0 and valid_moves > 2:
                return pos
            
        return None

    def backtrack_to_decision_point(self) -> bool:
        """回溯到最近的决策点"""
        decision_point = self._find_last_unexplored_decision()
        if decision_point:
            path = self._get_path_to_position(self.position, decision_point)
            if path:
                self.backtracking = True
                self.path_to_safe = path[1:]  # 不包括当前位置
                return True
        return False

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        self.previous_positions.append(self.position)
        
        dy, dx = self.actions[action]
        ny, nx = self.position[0] + dy, self.position[1] + dx
        new_pos = (ny, nx)
        
        if (0 <= ny < self.maze.height and 
            0 <= nx < self.maze.width and 
            self.maze.maze[ny, nx] == 0):
            self.position = (ny, nx)
            
        self.visit_count[ny, nx] += 1
        
        # 死胡同处理
        if self.detect_dead_end(self.position):
            self.dead_ends.add(self.position)
            if not self.backtracking:
                self.backtrack_to_decision_point()
        elif self.backtracking and not self.path_to_safe:  # 如果回溯完成
            self.backtracking = False
        
        observation = self.maze.get_local_observation(self.position)
        reward = self._calculate_reward()
        
        return observation, reward, self.position == self.maze.end

    def get_state_tensor(self, observation: np.ndarray) -> torch.Tensor:
        state_features = [
            self.position[0] / self.maze.height,
            self.position[1] / self.maze.width,
            abs(self.position[0] - self.maze.end[0]) / self.maze.height,
            abs(self.position[1] - self.maze.end[1]) / self.maze.width
        ]
        
        obs_flat = observation.flatten() / max(1, observation.max())
        features = state_features + obs_flat.tolist()
        
        return torch.tensor(features, device=self.device)

    def detect_dead_end(self, position: Tuple[int, int]) -> bool:
        """改进的死胡同检测"""
        if position == self.maze.end:
            return False
            
        valid_moves = 0
        total_moves = 0
        
        for dy, dx in self.actions:
            ny, nx = position[0] + dy, position[1] + dx
            if (0 <= ny < self.maze.height and 
                0 <= nx < self.maze.width and 
                self.maze.maze[ny, nx] == 0):
                total_moves += 1
                if (ny, nx) not in self.dead_ends:
                    valid_moves += 1
        
        return valid_moves <= 1 and total_moves > 1

    def is_potential_dead_end(self, position: Tuple[int, int]) -> bool:
        if position == self.maze.end:
            return False
            
        valid_moves = 0
        unvisited_moves = 0
        explored_empty_neighbors = 0
        
        for dy, dx in self.actions:
            ny, nx = position[0] + dy, position[1] + dx
            if 0 <= ny < self.maze.height and 0 <= nx < self.maze.width:
                if self.maze.maze[ny, nx] == 0:
                    next_pos = (ny, nx)
                    if self.maze.is_explored(next_pos):
                        explored_empty_neighbors += 1
                        if next_pos not in self.dead_ends:
                            valid_moves += 1
                    else:
                        unvisited_moves += 1
        
        return (valid_moves + unvisited_moves <= 1) or (explored_empty_neighbors <= 1 and unvisited_moves == 0)

    def get_valid_actions(self, other_positions: List[Tuple[int, int]]) -> List[int]:
        valid_actions = []
        for i, (dy, dx) in enumerate(self.actions):
            ny, nx = self.position[0] + dy, self.position[1] + dx
            new_pos = (ny, nx)
            if self._is_valid_move(new_pos, other_positions):
                valid_actions.append(i)
        return valid_actions if valid_actions else list(range(4))

    def _is_valid_move(self, new_pos: Tuple[int, int], 
                      other_positions: List[Tuple[int, int]]) -> bool:
        ny, nx = new_pos
        if not (0 <= ny < self.maze.height and 
                0 <= nx < self.maze.width and 
                self.maze.maze[ny, nx] == 0):
            return False
            
        for other_pos in other_positions:
            if self._euclidean_distance(new_pos, other_pos) < 2:
                return False
        return True

    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _euclidean_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        return np.sqrt(((pos1[0] - pos2[0]) ** 2) + ((pos1[1] - pos2[1]) ** 2))

    def _get_path_to_position(self, start: Tuple[int, int], 
                            target: Tuple[int, int]) -> List[Tuple[int, int]]:
        if start == target:
            return [start]
            
        visited = {start}
        queue = deque([(start, [start])])
        
        while queue:
            pos, path = queue.popleft()
            for dy, dx in self.actions:
                ny, nx = pos[0] + dy, pos[1] + dx
                next_pos = (ny, nx)
                
                if (next_pos not in visited and 
                    0 <= ny < self.maze.height and 
                    0 <= nx < self.maze.width and 
                    self.maze.maze[ny, nx] == 0):
                    
                    if next_pos == target:
                        return path + [next_pos]
                        
                    visited.add(next_pos)
                    queue.append((next_pos, path + [next_pos]))
        
        return []


    def get_batch_state_tensor(self, observation: torch.Tensor) -> torch.Tensor:
        """批量获取状态张量"""
        return observation.to(self.device)
    
    def select_batch_action(self, state_tensor: torch.Tensor, epsilon: float, 
                          other_positions: torch.Tensor) -> int:
        """批量选择动作"""
        if random.random() < epsilon:
            return random.choice(self.get_valid_actions(other_positions))
            
        # 批量计算所有动作的分数
        action_scores = self.q_values[state_tensor[:, 0].long(), 
                                    state_tensor[:, 1].long()]
        
        # 使用掩码处理无效动作
        valid_actions = self.get_valid_actions(other_positions)
        action_mask = torch.zeros_like(action_scores, dtype=torch.bool)
        action_mask[:, valid_actions] = True
        
        # 给无效动作赋予极小值
        action_scores[~action_mask] = float('-inf')
        
        return action_scores.argmax(dim=1)
    
    def _calculate_reward(self) -> float:
        reward = -0.1  # 基础步数惩罚
        
        if self.position == self.maze.end:
            reward = 1000  # 到达目标的主要奖励
        else:
            # 基于到终点距离的奖励
            curr_dist = self._manhattan_distance(self.position, self.maze.end)
            prev_dist = self._manhattan_distance(
                self.previous_positions[-1] if self.previous_positions else self.position,
                self.maze.end)
            
            if curr_dist < prev_dist:  # 如果距离终点更近了
                reward += 2
            elif curr_dist > prev_dist:  # 如果距离终点更远了
                reward -= 2

            # 探索新区域的奖励
            if not self.maze.is_explored(self.position):
                reward += 5

            # 回溯过程中的奖励调整
            if self.backtracking:
                if self.path_to_safe:  # 如果正在按计划回溯
                    reward += 1  # 给予轻微正奖励以鼓励完成回溯
                else:  # 如果偏离回溯路径
                    reward -= 2

            # 根据周围未探索区域数量给予奖励
            unexplored_neighbors = 0
            for dy, dx in self.actions:
                ny, nx = self.position[0] + dy, self.position[1] + dx
                if (0 <= ny < self.maze.height and 
                    0 <= nx < self.maze.width and 
                    not self.maze.is_explored((ny, nx))):
                    unexplored_neighbors += 1
            reward += unexplored_neighbors * 0.5

        return reward