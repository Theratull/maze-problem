import numpy as np
from collections import deque
import random
import time
import heapq
import matplotlib.pyplot as plt

class EnhancedMaze:
    def __init__(self, height=81, width=121):
        self.height = height if height % 2 == 1 else height + 1
        self.width = width if width % 2 == 1 else width + 1
        self.maze = np.ones((self.height, self.width), dtype=np.uint8)
        self.start = (1, 1)
        self.end = (self.height-2, self.width-2)
        self.portals = {}

    def generate_maze_prim(self):
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
        y, x = cell
        neighbors = []
        for dy, dx in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            ny, nx = y + dy, x + dx
            if 0 < ny < self.height-1 and 0 < nx < self.width-1:
                neighbors.append((ny, nx))
        return neighbors

    def _add_extra_paths(self):
        extra_paths = (self.height * self.width) // 20
        for _ in range(extra_paths):
            y = random.randrange(1, self.height-1)
            x = random.randrange(1, self.width-1)
            if self.maze[y, x] == 1:
                self.maze[y, x] = 0

    def _ensure_path_near_endpoints(self):
        for point in [self.start, self.end]:
            y, x = point
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    self.maze[ny, nx] = 0

    def solve_dfs(self):
        """Solve maze using iterative DFS"""
        stack = [(self.start, [self.start])]
        visited = {self.start}
        
        while stack:
            current, path = stack.pop()
            
            if current == self.end:
                return path
                
            for dy, dx in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                new_y, new_x = current[0] + dy, current[1] + dx
                next_pos = (new_y, new_x)
                
                if (0 <= new_x < self.width and 0 <= new_y < self.height 
                    and self.maze[next_pos] != 1
                    and next_pos not in visited):
                    visited.add(next_pos)
                    stack.append((next_pos, path + [next_pos]))
        
        return []

    def solve_bfs(self):
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
                    visited.add(next_pos)
                    queue.append((next_pos, path + [next_pos]))
        return []

    def solve_astar(self):
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
                    
                    new_cost = cost_so_far[current] + 1
                    
                    if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                        cost_so_far[next_pos] = new_cost
                        priority = new_cost + heuristic(self.end, next_pos)
                        heapq.heappush(frontier, (priority, next_pos))
                        came_from[next_pos] = current

        path = []
        current = self.end
        while current is not None:
            path.append(current)
            current = came_from.get(current)
        path.reverse()
        return path

def compare_algorithms(sizes):
    times = {'DFS': [], 'BFS': [], 'A*': []}
    path_lengths = {'DFS': [], 'BFS': [], 'A*': []}
    
    for size in sizes:
        print(f"\nTesting maze size: {size}x{size}")
        maze = EnhancedMaze(size, size)
        maze.generate_maze_prim()
        
        # DFS
        start_time = time.time()
        dfs_path = maze.solve_dfs()
        dfs_time = time.time() - start_time
        times['DFS'].append(dfs_time)
        path_lengths['DFS'].append(len(dfs_path))
        print(f"DFS - Time: {dfs_time:.3f}s, Path length: {len(dfs_path)}")
        
        # BFS
        start_time = time.time()
        bfs_path = maze.solve_bfs()
        bfs_time = time.time() - start_time
        times['BFS'].append(bfs_time)
        path_lengths['BFS'].append(len(bfs_path))
        print(f"BFS - Time: {bfs_time:.3f}s, Path length: {len(bfs_path)}")
        
        # A*
        start_time = time.time()
        astar_path = maze.solve_astar()
        astar_time = time.time() - start_time
        times['A*'].append(astar_time)
        path_lengths['A*'].append(len(astar_path))
        print(f"A* - Time: {astar_time:.3f}s, Path length: {len(astar_path)}")
    
    return times, path_lengths

def plot_results(sizes, times, path_lengths):
    plt.figure(figsize=(12, 5))
    
    # Plot solving times
    plt.subplot(1, 2, 1)
    for algorithm in times:
        plt.plot(sizes, times[algorithm], marker='o', label=algorithm)
    plt.title('Algorithm Solving Times')
    plt.xlabel('Maze Size')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    for algorithm in path_lengths:
        plt.plot(sizes, path_lengths[algorithm], marker='o', label=algorithm)
    plt.title('Path Lengths')
    plt.xlabel('Maze Size')
    plt.ylabel('Path Length')
    plt.grid(True)
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png')
    plt.close()

def main():
    # Use more reasonable sizes for testing
    sizes = [11, 51, 101, 201, 401, 801, 1601]
    times, path_lengths = compare_algorithms(sizes)
    plot_results(sizes, times, path_lengths)

if __name__ == "__main__":
    main()