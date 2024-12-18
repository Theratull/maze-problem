import numpy as np
import random
from typing import Tuple, List

class Maze:
    def __init__(self, height: int = 41, width: int = 71):
        """初始化迷宫环境"""
        self.height = height if height % 2 == 1 else height + 1
        self.width = width if width % 2 == 1 else width + 1
        
        # 迷宫相关属性
        self.maze = np.ones((self.height, self.width), dtype=np.uint8)
        self.exploration_map = np.full((self.height, self.width), -1)
        self.shared_knowledge = np.full((self.height, self.width), -1)
        
        # 起点、终点和辅助智能体位置
        self.start = (1, 1)
        self.helper_agent1 = (self.height-2, 1)
        self.helper_agent2 = (1, self.width-2)
        self.end = (self.height-2, self.width-2)
        
        # 视野和通信范围
        self.vision_range = 6
        self.communication_range = 20

    def generate_maze(self) -> None:
        """使用Prim算法生成迷宫"""
        # 设置随机种子
        random.seed(None)
        
        # 初始化迷宫为全墙
        self.maze.fill(1)
        
        # 随机选择起点（必须是奇数坐标）
        start_y = random.randrange(1, self.height - 1, 2)
        start_x = random.randrange(1, self.width - 1, 2)
        self.maze[start_y, start_x] = 0
        
        # 初始化墙列表
        walls = []
        # 将起点周围的墙加入列表
        self._add_walls_to_list(start_y, start_x, walls)
        
        # Prim算法主循环
        while walls:
            # 随机选择一面墙
            wall_index = random.randint(0, len(walls) - 1)
            wall_y, wall_x, from_y, from_x = walls[wall_index]
            walls.pop(wall_index)
            
            # 计算对面的格子
            to_y = wall_y + (wall_y - from_y)
            to_x = wall_x + (wall_x - from_x)
            
            # 如果对面的格子未被访问且在范围内
            if (0 < to_y < self.height-1 and 0 < to_x < self.width-1 and 
                self.maze[to_y, to_x] == 1):
                # 打通这面墙
                self.maze[wall_y, wall_x] = 0
                self.maze[to_y, to_x] = 0
                
                # 将新单元格周围的墙加入列表
                self._add_walls_to_list(to_y, to_x, walls)
        
        # 添加额外的路径
        self._create_additional_paths()
        
        # 确保关键位置可达
        self._ensure_key_positions_accessible()
        
        # 确保外墙完整
        for i in range(self.height):
            self.maze[i, 0] = 1
            self.maze[i, self.width-1] = 1
        for j in range(self.width):
            self.maze[0, j] = 1
            self.maze[self.height-1, j] = 1

    def _create_additional_paths(self) -> None:
        """创建额外的通道来增加路径选择"""
        # 计算需要添加的额外路径数量
        path_factor = 40  # 可以调整这个参数来控制额外路径的数量，数字越小路径越多
        num_extra_paths = (self.height * self.width) // path_factor
        paths_added = 0
        max_attempts = num_extra_paths * 4  # 防止无限循环
        attempts = 0
        
        while paths_added < num_extra_paths and attempts < max_attempts:
            # 随机选择一个位置（奇数坐标）
            y = random.randrange(1, self.height - 2)
            x = random.randrange(1, self.width - 2)
            
            # 随机选择一个方向
            dy = random.choice([-1, 1])
            dx = random.choice([-1, 1])
            
            # 检查当前位置和目标位置是否都是墙
            if (self.maze[y, x] == 1 and 
                self.maze[y + dy, x + dx] == 1 and 
                0 < y + dy < self.height - 1 and 
                0 < x + dx < self.width - 1):
                
                # 检查是否会创建大的开放空间
                wall_count = 0
                for cy in [y-1, y, y+1]:
                    for cx in [x-1, x, x+1]:
                        if (0 <= cy < self.height and 
                            0 <= cx < self.width and 
                            self.maze[cy, cx] == 1):
                            wall_count += 1
                
                # 如果周围有足够的墙，则打通路径
                if wall_count >= 5:  # 可以调整这个数值来控制通道的稀疏程度
                    self.maze[y, x] = 0
                    self.maze[y + dy, x + dx] = 0
                    paths_added += 1
            
            attempts += 1

    def _add_walls_to_list(self, y: int, x: int, walls: list) -> None:
        """将单元格周围的墙添加到列表中"""
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  
        random.shuffle(directions)  # 随机打乱方向以增加随机性
        
        for dy, dx in directions:
            wall_y = y + dy
            wall_x = x + dx
            
            if (0 < wall_y < self.height-1 and 0 < wall_x < self.width-1 and 
                self.maze[wall_y, wall_x] == 1):
                walls.append((wall_y, wall_x, y, x))

    def _ensure_key_positions_accessible(self) -> None:
        """确保所有关键位置可达"""
        # 只需要确保起点和终点可达
        key_positions = [self.start, self.end]
        
        for pos in key_positions:
            y, x = pos
            self.maze[y, x] = 0  # 确保位置本身是通道
            
            # 确保至少有一个相邻位置是通道
            for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ny, nx = y + dy, x + dx
                if 0 < ny < self.height-1 and 0 < nx < self.width-1:
                    self.maze[ny, nx] = 0
                    break

    def get_local_observation(self, current_pos: Tuple[int, int]) -> np.ndarray:
        """获取局部观察并更新探索地图"""
        y, x = current_pos
        size = 2 * self.vision_range + 1
        observation = np.ones((size, size), dtype=np.int32) * -1

        for dy in range(-self.vision_range, self.vision_range + 1):
            for dx in range(-self.vision_range, self.vision_range + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    observation[dy + self.vision_range, dx + self.vision_range] = self.maze[ny, nx]
                    self.exploration_map[ny, nx] = self.maze[ny, nx]
        
        return observation

    def is_explored(self, position: Tuple[int, int]) -> bool:
        """检查位置是否已被探索"""
        y, x = position
        if not (0 <= y < self.height and 0 <= x < self.width):
            return False
        return self.exploration_map[y, x] != -1

    def can_communicate(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """检查两个位置是否在通信范围内"""
        return np.sqrt(((pos1[0] - pos2[0]) ** 2) + ((pos1[1] - pos2[1]) ** 2)) <= self.communication_range


    def get_start_position(self, agent_id: int) -> Tuple[int, int]:
        """获取指定智能体的起始位置，现在所有智能体都从起点开始"""
        return self.start  # 所有智能体都从同一个起点出发