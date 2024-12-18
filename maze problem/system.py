import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from tqdm import tqdm
from collections import deque
from typing import List, Dict, Any, Tuple, Optional
from agent import Agent
from maze import Maze
import time  # 添加到文件开头的导入部分

class MultiAgentSystem:
    def __init__(self, maze: Maze, num_agents: int = 6):  # 默认6个智能体
        self.maze = maze
        self.num_agents = num_agents
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 全局状态
        self.global_dead_ends = set()
        self.global_visit_count = np.zeros((maze.height, maze.width))
        self.global_decision_points = {}
        self.global_unexplored_branches = {}
        self.shared_knowledge = np.full((maze.height, maze.width), -1)
        self.actual_maze = maze.maze.copy()
        
        print(f"Initializing system with {num_agents} agents...")
        
        # 初始化智能体
        device_id = 0 if torch.cuda.is_available() else -1
        self.agents = [
            Agent(maze, maze.get_start_position(i), i, device_id,
                 self.global_dead_ends,
                 self.global_visit_count,
                 self.global_decision_points,
                 self.global_unexplored_branches)
            for i in range(self.num_agents)
        ]
        
        # 训练相关
        self.exploration_history = []
        self.max_steps = 2000
        self.connected_path = None
        self.training_complete = False

    def _record_state(self, step: int, path_found: bool = False) -> None:
        """记录当前状态"""
        self.exploration_history.append({
            'step': step,
            'positions': [agent.position for agent in self.agents],
            'knowledge': self.shared_knowledge.copy(),
            'path_found': path_found
        })

    def _update_visit_counts(self) -> None:
        """更新全局访问计数"""
        for agent in self.agents:
            self.global_visit_count[agent.position[0], agent.position[1]] += 1

    def _update_shared_knowledge(self, position: Tuple[int, int], observation: np.ndarray) -> None:
        """更新共享知识地图"""
        y, x = position
        for dy in range(-self.maze.vision_range, self.maze.vision_range + 1):
            for dx in range(-self.maze.vision_range, self.maze.vision_range + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.maze.height and 0 <= nx < self.maze.width:
                    obs_y, obs_x = dy + self.maze.vision_range, dx + self.maze.vision_range
                    if observation[obs_y, obs_x] != -1:
                        self.shared_knowledge[ny, nx] = observation[obs_y, obs_x]

    def _find_connected_path(self) -> Optional[List[Tuple[int, int]]]:
        """使用完整迷宫信息查找从起点到终点的路径"""
        start = self.maze.start
        end = self.maze.end
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            if current == end:
                return path
                
            for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ny, nx = current[0] + dy, current[1] + dx
                next_pos = (ny, nx)
                
                if (0 <= ny < self.maze.height and 
                    0 <= nx < self.maze.width and 
                    next_pos not in visited and 
                    self.actual_maze[ny, nx] == 0):
                    
                    visited.add(next_pos)
                    queue.append((next_pos, path + [next_pos]))
        
        return None

    def _calculate_exploration_ratio(self) -> float:
        """计算探索比率"""
        explored = np.sum(self.shared_knowledge != -1)
        total = self.maze.height * self.maze.width
        return explored / total

    def _check_improvement(self, current_ratio: float, best_ratio: float, 
                         no_improvement_steps: int) -> int:
        """检查探索进度是否有改善"""
        return 0 if current_ratio > best_ratio else no_improvement_steps + 1

    def _agents_step(self, current_paths: List[List], current_rewards: List[float], 
                        best_rewards: List[float]) -> Tuple[bool, List[Tuple[int, int]], bool]:
        """并行执行所有智能体的行动"""
        if self.training_complete:
            return True, [agent.position for agent in self.agents], False

        # 1. 批量获取所有智能体的观察和状态
        all_observations = torch.tensor([
            self.maze.get_local_observation(agent.position) 
            for agent in self.agents
        ], device=self.device)  # Shape: [num_agents, vision_size, vision_size]
        
        all_positions = torch.tensor([
            [agent.position[0], agent.position[1]] 
            for agent in self.agents
        ], device=self.device)  # Shape: [num_agents, 2]
        
        # 2. 批量计算状态张量
        batch_state_features = torch.cat([
            all_positions / torch.tensor([self.maze.height, self.maze.width], device=self.device),
            torch.abs(all_positions - torch.tensor([self.maze.end[0], self.maze.end[1]], device=self.device)) / 
            torch.tensor([self.maze.height, self.maze.width], device=self.device)
        ], dim=1)  # Shape: [num_agents, 4]
        
        obs_flat = all_observations.reshape(self.num_agents, -1) / torch.max(all_observations)
        all_state_tensors = torch.cat([batch_state_features, obs_flat], dim=1)
        
        # 3. 批量计算epsilon
        epsilon = max(0.05, 0.5 * np.exp(-len(self.exploration_history)/500))
        
        # 4. 并行计算所有智能体的动作
        # 为每个智能体创建其他智能体位置的掩码
        positions_matrix = all_positions.unsqueeze(1).repeat(1, self.num_agents, 1)  # [num_agents, num_agents, 2]
        agent_mask = ~torch.eye(self.num_agents, device=self.device).bool()
        other_positions = positions_matrix[agent_mask].view(self.num_agents, self.num_agents-1, 2)
        
        # 并行计算所有可能的动作分数
        actions = torch.tensor([(0, 1), (1, 0), (0, -1), (-1, 0)], device=self.device)
        next_positions = all_positions.unsqueeze(1) + actions.unsqueeze(0)  # [num_agents, 4, 2]
        
        # 并行计算有效动作掩码
        valid_moves = (
            (next_positions[..., 0] >= 0) & 
            (next_positions[..., 0] < self.maze.height) & 
            (next_positions[..., 1] >= 0) & 
            (next_positions[..., 1] < self.maze.width)
        )
        
        # 5. 并行更新状态和奖励
        # 使用scatter操作批量更新访问计数
        visit_indices = all_positions.long()
        self.global_visit_count = self.global_visit_count.to(self.device)
        self.global_visit_count[visit_indices[:, 0], visit_indices[:, 1]] += 1
        
        # 并行计算奖励
        rewards = torch.zeros(self.num_agents, device=self.device)
        done_mask = torch.tensor([
            agent.position == self.maze.end for agent in self.agents
        ], device=self.device)
        
        # 6. 检查是否有智能体到达终点
        if done_mask.any():
            path = self._find_connected_path()
            if path is not None:
                self.connected_path = path
                self.training_complete = True
                print(f"\nFound connected path! Length: {len(path)}")
                return True, [agent.position for agent in self.agents], True
        
        # 7. 更新共享知识
        for i, agent in enumerate(self.agents):
            self._update_shared_knowledge(agent.position, all_observations[i].cpu().numpy())
        
        new_positions = [agent.position for agent in self.agents]
        return False, new_positions, False


    def train(self) -> None:
        """训练多智能体系统"""
        print("Training agents...")
        
        # 开始计时
        start_time = time.time()
        
        step = 0
        no_improvement_steps = 0
        best_exploration_ratio = 0
        current_paths = [[] for _ in range(self.num_agents)]
        best_rewards = [float('-inf')] * self.num_agents
        current_rewards = [0] * self.num_agents
        
        self._record_state(step)
        
        with tqdm(total=self.max_steps) as pbar:
            while step < self.max_steps and not self.training_complete:
                self._update_visit_counts()
                
                all_done, new_positions, path_found = self._agents_step(
                    current_paths, current_rewards, best_rewards)
                
                self._record_state(step, path_found)
                
                explored_ratio = self._calculate_exploration_ratio()
                no_improvement_steps = self._check_improvement(
                    explored_ratio, best_exploration_ratio, no_improvement_steps)
                best_exploration_ratio = max(best_exploration_ratio, explored_ratio)
                
                explored = np.sum(self.shared_knowledge != -1)
                total = self.maze.height * self.maze.width
                pbar.update(1)
                pbar.set_postfix({
                    'Explored': f"{explored}/{total}",
                    'Ratio': f"{explored_ratio:.2%}",
                    'NoImprove': no_improvement_steps
                })
                
                if self.training_complete:
                    # 结束计时并计算用时
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    
                    # 格式化时间显示
                    minutes = int(elapsed_time // 60)
                    seconds = elapsed_time % 60
                    
                    print("\nTraining completed - Found path to goal!")
                    print(f"Time taken: {minutes} minutes {seconds:.2f} seconds")
                    print(f"Total steps: {step}")
                    print(f"Path length: {len(self.connected_path)}")
                    break
                    
                if no_improvement_steps >= 200 and not self.training_complete:
                    print("\nNo improvement detected, attempting backtracking...")
                    backtrack_success = False
                    for agent in self.agents:
                        if agent.backtrack_to_decision_point():
                            backtrack_success = True
                    
                    if backtrack_success:
                        print("Successfully backtracked to decision point")
                        no_improvement_steps = 0
                        current_rewards = [0] * self.num_agents
                    else:
                        print("No valid backtrack points found, continuing exploration")
                
                step += 1
            
            # 如果达到最大步数仍未找到路径
            if not self.training_complete:
                end_time = time.time()
                elapsed_time = end_time - start_time
                minutes = int(elapsed_time // 60)
                seconds = elapsed_time % 60
                print("\nTraining incomplete - Maximum steps reached")
                print(f"Time taken: {minutes} minutes {seconds:.2f} seconds")
                print(f"Total steps: {step}")
                
    def visualize_exploration_process(self) -> None:
        """可视化探索过程"""
        if not self.exploration_history:
            print("No exploration history to visualize")
            return
        
        fig = plt.figure(figsize=(12, 8))
        ax = plt.gca()
        
        # 找到最终发现路径的帧
        path_found_frame = len(self.exploration_history) - 1
        for i, state in enumerate(self.exploration_history):
            if state.get('path_found', False):
                path_found_frame = i
                break
        
        def update(frame):
            ax.clear()
            state = self.exploration_history[frame]
            positions = state['positions']
            
            # 判断是否是最后一帧
            is_last_frame = frame == len(self.exploration_history) - 1
            
            if is_last_frame:
                # 最后一帧显示完整迷宫
                maze_image = np.zeros((self.maze.height, self.maze.width))
                maze_image[self.actual_maze == 0] = 1.0
                ax.imshow(maze_image, cmap='gray')
                
                # 显示完整路径
                if self.connected_path:
                    path_y, path_x = zip(*self.connected_path)
                    ax.plot(path_x, path_y, 'r-', linewidth=6, alpha=0.8, 
                           solid_capstyle='round', zorder=100, label='Found Path')
            else:
                # 正常帧显示带迷雾的迷宫
                fog_image = np.ones((self.maze.height, self.maze.width)) * 0.1
                visibility_mask = np.zeros((self.maze.height, self.maze.width), dtype=bool)
                
                # 更新每个智能体周围的可见区域
                for pos in positions:
                    y, x = pos
                    for dy in range(-self.maze.vision_range, self.maze.vision_range + 1):
                        for dx in range(-self.maze.vision_range, self.maze.vision_range + 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < self.maze.height and 0 <= nx < self.maze.width:
                                distance = np.sqrt(dy**2 + dx**2)
                                if distance <= self.maze.vision_range:
                                    visibility_mask[ny, nx] = True
                                    fog_image[ny, nx] = 1.0 if self.maze.maze[ny, nx] == 0 else 0.0
                
                ax.imshow(fog_image, cmap='gray')
                
                # 在找到路径后显示可见部分的路径
                if self.connected_path and frame >= path_found_frame:
                    path_y, path_x = zip(*self.connected_path)
                    visible_path = [(y, x) for y, x in zip(path_y, path_x) if visibility_mask[y, x]]
                    if visible_path:
                        visible_y, visible_x = zip(*visible_path)
                        ax.plot(visible_x, visible_y, 'r-', linewidth=6, alpha=0.8, 
                               solid_capstyle='round', zorder=100, label='Found Path')
            
            colors = plt.cm.rainbow(np.linspace(0, 1, self.num_agents))
            
            for i, pos in enumerate(positions):
                if not is_last_frame:
                    current_segment = []
                    for j in range(max(0, frame-50), frame+1):
                        if j >= len(self.exploration_history):
                            break
                        curr_pos = self.exploration_history[j]['positions'][i]
                        current_segment.append(curr_pos)
                    
                    if len(current_segment) > 1:
                        seg_y, seg_x = zip(*current_segment)
                        if not is_last_frame:
                            visible_trail = [(y, x) for y, x in zip(seg_y, seg_x) 
                                           if visibility_mask[y, x]]
                            if visible_trail:
                                trail_y, trail_x = zip(*visible_trail)
                                ax.plot(trail_x, trail_y, '-', color=colors[i], 
                                       alpha=0.4, linewidth=2)
                
                ax.plot(pos[1], pos[0], 'o', color=colors[i], 
                       markersize=10, label=f'Agent {i}')
            
            if is_last_frame or visibility_mask[self.maze.start[0], self.maze.start[1]]:
                ax.plot(self.maze.start[1], self.maze.start[0], 'gs', 
                       markersize=10, label='Start')
            if is_last_frame or visibility_mask[self.maze.end[0], self.maze.end[1]]:
                ax.plot(self.maze.end[1], self.maze.end[0], 'rs', 
                       markersize=10, label='Goal')
            
            if is_last_frame:
                title = f'Final Path (Total Steps: {len(self.exploration_history)})'
                if self.connected_path:
                    title += f'\nPath Length: {len(self.connected_path)}'
            else:
                explored = np.sum(visibility_mask)
                total = self.maze.height * self.maze.width
                title = f'Step {frame}/{len(self.exploration_history)-1}\n' \
                        f'Visible: {explored}/{total} ({explored/total:.1%})'
                if frame >= path_found_frame and self.connected_path:
                    title += '\nPath Found!'
            
            ax.set_title(title)
            if not is_last_frame:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            
            return ax.get_children
        
        
        ani = animation.FuncAnimation(fig, update,frames=len(self.exploration_history),interval=50,repeat=False)
        plt.show()