import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import time
import heapq
from rl_solver import RLSolver

class EnhancedMaze:
    def __init__(self, height=81, width=121):
        self.height = height if height % 2 == 1 else height + 1
        self.width = width if width % 2 == 1 else width + 1
        # 0: 通道, 1: 墙, 3+: 传送门对
        self.maze = np.ones((self.height, self.width), dtype=np.uint8)
        self.start = (1, 1)
        self.end = (self.height-2, self.width-2)
        self.portals = {}  # 存储传送门配对信息，格式：{位置: 目标位置}

    def _add_portal_pairs(self, portal_density=0.005):
        """添加传送门对
        Args:
            portal_density: 传送门密度，占总地图面积的比例
        """
        # 计算传送门对数量
        total_area = self.height * self.width
        num_portal_pairs = max(1, int(total_area * portal_density))
        
        # 获取所有可用的位置（排除墙壁、起点和终点）
        valid_positions = [(y, x) for y in range(self.height) 
                          for x in range(self.width) 
                          if self.maze[y, x] == 0 
                          and (y, x) != self.start 
                          and (y, x) != self.end]
        
        # 随机选择位置放置传送门对
        portal_id = 3  # 从3开始编号传送门
        for _ in range(num_portal_pairs):
            if len(valid_positions) < 2:
                break
            
            # 选择第一个传送门位置
            pos1 = random.choice(valid_positions)
            valid_positions.remove(pos1)
            
            # 选择第二个传送门位置（距离第一个有一定距离）
            min_dist = min(self.height, self.width) // 4  # 最小距离要求
            valid_pos2 = [pos for pos in valid_positions 
                         if abs(pos[0] - pos1[0]) + abs(pos[1] - pos1[1]) >= min_dist]
            
            if not valid_pos2:
                continue
                
            pos2 = random.choice(valid_pos2)
            valid_positions.remove(pos2)
            
            # 设置传送门对
            self.maze[pos1] = portal_id
            self.maze[pos2] = portal_id
            self.portals[pos1] = pos2
            self.portals[pos2] = pos1
            
            portal_id += 1

    def generate_maze_prim(self):
        """使用 Prim 算法生成基本迷宫"""
        def add_walls(cell):
            for neighbor in self.get_neighbors(cell):
                if self.maze[neighbor] == 1:
                    wall = ((cell[0] + neighbor[0]) // 2,
                           (cell[1] + neighbor[1]) // 2)
                    walls.add((neighbor, wall))

        walls = set()
        self.maze[self.start] = 0
        add_walls(self.start)

        while walls:
            cell, wall = random.choice(list(walls))
            walls.remove((cell, wall))
            
            if self.maze[cell] == 1:
                self.maze[wall] = 0
                self.maze[cell] = 0
                add_walls(cell)

        self._add_extra_paths()
        self.maze[self.end] = 0
        self._ensure_path_near_endpoints()

    def get_neighbors(self, cell):
        """获取相邻单元格"""
        y, x = cell
        neighbors = []
        for dy, dx in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            ny, nx = y + dy, x + dx
            if 0 < ny < self.height-1 and 0 < nx < self.width-1:
                neighbors.append((ny, nx))
        return neighbors

    def _add_extra_paths(self):
        """添加额外的通道增加路径多样性"""
        extra_paths = (self.height * self.width) // 20
        for _ in range(extra_paths):
            y = random.randrange(1, self.height-1)
            x = random.randrange(1, self.width-1)
            if self.maze[y, x] == 1:
                self.maze[y, x] = 0

    def _ensure_path_near_endpoints(self):
        """确保起点和终点周围有通路"""
        for point in [self.start, self.end]:
            y, x = point
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    self.maze[ny, nx] = 0

    def get_next_position(self, current_pos, next_pos):
        """获取下一个位置，考虑传送门效果"""
        if next_pos in self.portals:
            return self.portals[next_pos]
        return next_pos

    def solve_dfs(self):
        """使用DFS求解迷宫"""
        path = []
        visited = set()
        
        def dfs(current):
            if current == self.end:
                return True
            
            visited.add(current)
            path.append(current)
            
            for dy, dx in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                new_y, new_x = current[0] + dy, current[1] + dx
                next_pos = (new_y, new_x)
                
                if (0 <= new_x < self.width and 0 <= new_y < self.height 
                    and self.maze[next_pos] != 1
                    and next_pos not in visited):
                    
                    actual_next = self.get_next_position(current, next_pos)
                    if actual_next not in visited:
                        if actual_next != next_pos:
                            path.append(next_pos)  # 记录传送门入口
                        if dfs(actual_next):
                            return True
                        if actual_next != next_pos:
                            path.pop()  # 传送门入口也需要弹出
            
            path.pop()
            return False
        
        dfs(self.start)
        return path

    def solve_bfs(self):
        """使用BFS求解迷宫"""
        queue = deque([(self.start, [self.start])])
        visited = {self.start}
        
        while queue:
            current, path = queue.popleft()
            
            if current == self.end:
                return path
            
            for dy, dx in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                new_y, new_x = current[0] + dy, current[1] + dx
                next_pos = (new_y, new_x)
                
                if (0 <= new_x < self.width and 0 <= new_y < self.height 
                    and self.maze[next_pos] != 1
                    and next_pos not in visited):
                    
                    actual_next = self.get_next_position(current, next_pos)
                    if actual_next not in visited:
                        visited.add(actual_next)
                        new_path = path + [next_pos]
                        if actual_next != next_pos:
                            new_path.append(actual_next)
                        queue.append((actual_next, new_path))
        
        return []

    def solve_astar(self):
        """使用A*算法求解迷宫"""
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        frontier = [(0, self.start)]
        came_from = {self.start: None}
        cost_so_far = {self.start: 0}
        
        while frontier:
            current = heapq.heappop(frontier)[1]
            
            if current == self.end:
                break
                
            for dy, dx in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                next_pos = (current[0] + dy, current[1] + dx)
                
                if (0 <= next_pos[0] < self.height and 0 <= next_pos[1] < self.width 
                    and self.maze[next_pos] != 1):
                    
                    actual_next = self.get_next_position(current, next_pos)
                    new_cost = cost_so_far[current] + 1
                    
                    if actual_next not in cost_so_far or new_cost < cost_so_far[actual_next]:
                        cost_so_far[actual_next] = new_cost
                        priority = new_cost + heuristic(self.end, actual_next)
                        heapq.heappush(frontier, (priority, actual_next))
                        came_from[actual_next] = current

        path = []
        current = self.end
        while current is not None:
            path.append(current)
            current = came_from.get(current)
        path.reverse()
        return path

    def visualize_maze(self, dfs_path=None, bfs_path=None, astar_path=None, rl_path=None):
        """可视化迷宫和路径"""
        paths = [p for p in [dfs_path, bfs_path, astar_path, rl_path] if p is not None]
        num_plots = len(paths)
        if num_plots == 0:
            num_plots = 1
            
        plt.figure(figsize=(5*num_plots, 5))
        
        titles = ['DFS Path', 'BFS Path', 'A* Path', 'RL Path']
        colors = ['r-', 'b-', 'g-', 'y-']
        
        for i, (path, title, color) in enumerate(zip(paths, titles, colors)):
            plt.subplot(1, num_plots, i+1)
            
            # 创建RGB图像
            rgb_maze = np.zeros((self.height, self.width, 3))
            
            # 设置基本迷宫颜色
            rgb_maze[self.maze == 1] = [0, 0, 0]  # 墙是黑色
            rgb_maze[self.maze == 0] = [1, 1, 1]  # 通道是白色
            
            # 为每对传送门设置相同的颜色
            if len(self.portals) > 0:
                portal_colors = plt.cm.rainbow(np.linspace(0, 1, len(self.portals)//2))
                portal_id = 3
                for j in range(len(self.portals)//2):
                    mask = (self.maze == portal_id)
                    rgb_maze[mask] = portal_colors[j][:3]
                    portal_id += 1
            
            plt.imshow(rgb_maze)
            
            if path:
                path_y, path_x = zip(*path)
                plt.plot(path_x, path_y, color, linewidth=2, alpha=0.7)
            
            plt.plot(self.start[1], self.start[0], 'go', markersize=10, label='Start')
            plt.plot(self.end[1], self.end[0], 'ro', markersize=10, label='End')
            plt.title(f'{title}\n(length: {len(path) if path else 0})')
            plt.legend()
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    # 创建迷宫实例
    maze = EnhancedMaze(41, 61)  # 使用较小的尺寸以便观察传送门效果
    print("生成迷宫中...")
    
    # 生成基本迷宫
    maze.generate_maze_prim()
    
    # 添加传送门
    # maze._add_portal_pairs(portal_density=0.01)  # 可以调整传送门密度
    
    print("使用不同算法求解中...")
    
    # DFS求解
    start_time = time.time()
    dfs_path = maze.solve_dfs()
    dfs_time = time.time() - start_time
    print(f"DFS求解耗时: {dfs_time:.3f}秒")
    
    # BFS求解
    start_time = time.time()
    bfs_path = maze.solve_bfs()
    bfs_time = time.time() - start_time
    print(f"BFS求解耗时: {bfs_time:.3f}秒")
    
    # A*求解
    start_time = time.time()
    astar_path = maze.solve_astar()
    astar_time = time.time() - start_time
    print(f"A*求解耗时: {astar_time:.3f}秒")
    
    # RL求解
    start_time = time.time()
    # 创建RL求解器实例
    rl_solver = RLSolver(maze.maze, maze.start, maze.end, maze.portals, set())  # 没有陷阱，所以传入空集合
    print("训练RL模型中...")
    rl_path = rl_solver.train(episodes=1000)  # 可以调整训练轮数
    rl_time = time.time() - start_time
    print(f"RL求解耗时: {rl_time:.3f}秒")
    
    print("\n路径长度比较:")
    print(f"DFS路径长度: {len(dfs_path)}")
    print(f"BFS路径长度: {len(bfs_path)}")
    print(f"A*路径长度: {len(astar_path)}")
    print(f"RL路径长度: {len(rl_path)}")
    
    print("\n求解时间比较:")
    print(f"DFS耗时: {dfs_time:.3f}秒")
    print(f"BFS耗时: {bfs_time:.3f}秒")
    print(f"A*耗时: {astar_time:.3f}秒")
    print(f"RL耗时: {rl_time:.3f}秒")
    
    print("\n可视化结果...")
    maze.visualize_maze(dfs_path, bfs_path, astar_path, rl_path)

if __name__ == "__main__":
    main()