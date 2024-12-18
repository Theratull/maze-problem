import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import time
import heapq
from rl_solver_block import RLSolver

class EnhancedMaze:
    def __init__(self, height=81, width=121):
        self.height = height if height % 2 == 1 else height + 1
        self.width = width if width % 2 == 1 else width + 1
        self.maze = np.ones((self.height, self.width), dtype=np.uint8)
        self.start = (1, 1)
        self.end = (self.height-2, self.width-2)
        self.portals = {}
        self.movement_costs = {}

    def _add_special_zones(self, density=0.2, min_size=4, max_size=8):
        """添加特殊移动代价区域"""
        # 计算大致需要多少个区域来达到目标密度
        avg_size = (min_size + max_size) / 2
        total_area = self.height * self.width
        free_area = sum(1 for i in range(self.height) for j in range(self.width) if self.maze[i,j] == 0)
        target_special_area = free_area * density
        num_zones = int(target_special_area / (avg_size * avg_size))
        
        print(f"添加约 {num_zones} 个特殊区域，目标覆盖率 {density*100:.1f}%")
        
        for _ in range(num_zones):
            size = random.randint(min_size, max_size)
            y = random.randint(1, self.height - size - 1)
            x = random.randint(1, self.width - size - 1)
            cost_multiplier = random.choice([3, 4])
            self.movement_costs[(y, x, y+size, x+size)] = cost_multiplier

    def get_movement_cost(self, pos):
        """获取指定位置的移动代价"""
        max_cost = 1
        y, x = pos
        for (y1, x1, y2, x2), cost in self.movement_costs.items():
            if y1 <= y <= y2 and x1 <= x <= x2:
                max_cost = max(max_cost, cost)
        return max_cost

    def generate_maze_prim(self, path_density=0.3):
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

        self._add_extra_paths(density=path_density)
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

    def _add_extra_paths(self, density=0.3):
        """添加额外的通道"""
        for y in range(1, self.height-1):
            for x in range(1, self.width-1):
                if self.maze[y, x] == 1:
                    has_path = False
                    for (dy1, dx1), (dy2, dx2) in [
                        ((-1, 0), (1, 0)),
                        ((0, -1), (0, 1)),
                        ((-1, -1), (1, 1)),
                        ((-1, 1), (1, -1))
                    ]:
                        p1 = (y + dy1, x + dx1)
                        p2 = (y + dy2, x + dx2)
                        if (0 <= p1[0] < self.height and 0 <= p1[1] < self.width and
                            0 <= p2[0] < self.height and 0 <= p2[1] < self.width and
                            self.maze[p1] == 0 and self.maze[p2] == 0):
                            has_path = True
                            break
                    
                    if has_path and random.random() < density:
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

    def solve_bfs(self):
        """使用考虑代价的BFS算法"""
        # 使用优先队列来按照总代价排序
        pq = [(0, self.start, [self.start])]  # (总代价, 当前位置, 路径)
        visited = {self.start: 0}  # 记录到达每个位置的最小代价
        
        while pq:
            # 获取当前代价最小的路径
            current_cost, current, path = heapq.heappop(pq)
            
            # 如果到达终点，返回路径和总代价
            if current == self.end:
                return path, current_cost
                
            # 检查四个方向
            for dy, dx in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                next_pos = (current[0] + dy, current[1] + dx)
                
                # 检查是否在迷宫范围内且不是墙
                if (0 <= next_pos[0] < self.height and 
                    0 <= next_pos[1] < self.width and 
                    self.maze[next_pos] != 1):
                    
                    # 获取实际的下一个位置（考虑传送门）
                    actual_next = self.get_next_position(current, next_pos)
                    
                    # 计算移动到下一个位置的代价
                    move_cost = self.get_movement_cost(next_pos)
                    new_cost = current_cost + move_cost
                    
                    # 如果找到更低代价的路径或是第一次访问该位置
                    if actual_next not in visited or new_cost < visited[actual_next]:
                        visited[actual_next] = new_cost
                        new_path = path + [next_pos]
                        if actual_next != next_pos:
                            new_path.append(actual_next)
                        heapq.heappush(pq, (new_cost, actual_next, new_path))
        
        # 如果没有找到路径
        return [], float('inf')

    def solve_astar(self):
        """使用A*算法求解迷宫（考虑移动代价）"""
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
                    move_cost = self.get_movement_cost(next_pos)
                    new_cost = cost_so_far[current] + move_cost
                    
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
        
        return path, cost_so_far[self.end] if self.end in cost_so_far else float('inf')

    def solve_dfs(self):
        """使用DFS求解迷宫（考虑移动代价）"""
        path = []
        visited = set()
        path_costs = {self.start: 0}
        
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
                        move_cost = self.get_movement_cost(next_pos)
                        path_costs[actual_next] = path_costs[current] + move_cost
                        
                        if actual_next != next_pos:
                            path.append(next_pos)
                        if dfs(actual_next):
                            return True
                        if actual_next != next_pos:
                            path.pop()
            
            path.pop()
            return False
        
        dfs(self.start)
        return path, path_costs[self.end] if self.end in path_costs else float('inf')

    def visualize_maze(self, dfs_path=None, bfs_path=None, astar_path=None, rl_path=None):
        """可视化迷宫和路径"""
        paths = [(p[0] if isinstance(p, tuple) else p) for p in [dfs_path, bfs_path, astar_path, rl_path] if p is not None]
        costs = [p[1] if isinstance(p, tuple) else None for p in [dfs_path, bfs_path, astar_path, rl_path] if p is not None]
        
        num_plots = len(paths)
        if num_plots == 0:
            num_plots = 1
            
        plt.figure(figsize=(5*num_plots, 5))
        
        titles = ['DFS Path', 'BFS Path', 'A* Path', 'RL Path']
        colors = ['r-', 'b-', 'g-', 'y-']
        
        for i, (path, title, color, cost) in enumerate(zip(paths, titles, colors, costs)):
            plt.subplot(1, num_plots, i+1)
            
            rgb_maze = np.zeros((self.height, self.width, 3))
            rgb_maze[self.maze == 1] = [0, 0, 0]
            rgb_maze[self.maze == 0] = [1, 1, 1]
            
            for (y1, x1, y2, x2), multiplier in self.movement_costs.items():
                alpha = 0.3 if multiplier == 3 else 0.5
                color_map = [1, 0, 0] if multiplier == 3 else [0, 1, 0]
                for y in range(y1, y2+1):
                    for x in range(x1, x2+1):
                        if self.maze[y, x] != 1:
                            rgb_maze[y, x] = [c * (1-alpha) + color_map[j] * alpha for j, c in enumerate(rgb_maze[y, x])]
            
            plt.imshow(rgb_maze)
            
            if path:
                path_y, path_x = zip(*path)
                plt.plot(path_x, path_y, color, linewidth=2, alpha=0.7)
            
            plt.plot(self.start[1], self.start[0], 'go', markersize=10, label='Start')
            plt.plot(self.end[1], self.end[0], 'ro', markersize=10, label='End')
            title_text = f'{title}\n(length: {len(path) if path else 0})'
            if cost is not None:
                title_text += f'\n(cost: {cost:.1f})'
            plt.title(title_text)
            plt.legend()
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    # Create maze instance
    maze = EnhancedMaze(41, 61)  # Using smaller dimensions for observation
    print("Generating maze...")
    
    # Generate basic maze with higher path density
    maze.generate_maze_prim(path_density=0.15)
    
    # Add special zones covering about 20% of walkable area
    maze._add_special_zones(density=0.50, min_size=4, max_size=8)
    print("Special zones:", maze.movement_costs)
    
    print("Solving with different algorithms...")
    
    # DFS solution
    start_time = time.time()
    dfs_path = maze.solve_dfs()
    dfs_time = time.time() - start_time
    print(f"DFS solution time: {dfs_time:.3f} seconds")
    
    # BFS solution
    start_time = time.time()
    bfs_path = maze.solve_bfs()
    bfs_time = time.time() - start_time
    print(f"BFS solution time: {bfs_time:.3f} seconds")
    
    # A* solution
    start_time = time.time()
    astar_path = maze.solve_astar()
    astar_time = time.time() - start_time
    print(f"A* solution time: {astar_time:.3f} seconds")
    
    # RL solution
    start_time = time.time()
    # Create RL solver instance - Fixed initialization
    traps = set()  # Define traps set
    rl_solver = RLSolver(maze_array=maze.maze, 
                        start=maze.start, 
                        end=maze.end, 
                        portals=maze.portals,
                        traps=traps,
                        movement_costs=maze.movement_costs)
    
    print("Training RL model...")
    rl_path, rl_cost = rl_solver.train(episodes=1200)
    rl_time = time.time() - start_time
    print(f"RL solution time: {rl_time:.3f} seconds")
    
    print("\nPath length and cost comparison:")
    print(f"DFS - Length: {len(dfs_path[0])}, Cost: {dfs_path[1]:.1f}")
    print(f"BFS - Length: {len(bfs_path[0])}, Cost: {bfs_path[1]:.1f}")
    print(f"A* - Length: {len(astar_path[0])}, Cost: {astar_path[1]:.1f}")
    print(f"RL - Length: {len(rl_path)}, Cost: {rl_cost:.1f}")
    
    print("\nSolution time comparison:")
    print(f"DFS time: {dfs_time:.3f} seconds")
    print(f"BFS time: {bfs_time:.3f} seconds")
    print(f"A* time: {astar_time:.3f} seconds")
    print(f"RL time: {rl_time:.3f} seconds")
    
    print("\nVisualizing results...")
    maze.visualize_maze(dfs_path=dfs_path, bfs_path=bfs_path, 
                       astar_path=astar_path, rl_path=(rl_path, rl_cost))

if __name__ == "__main__":
    main()