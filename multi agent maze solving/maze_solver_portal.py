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
        # 0: path, 1: wall, 3+: portal pairs
        self.maze = np.ones((self.height, self.width), dtype=np.uint8)
        self.start = (1, 1)
        self.end = (self.height-2, self.width-2)
        self.portals = {}  # Store portal pair information, format: {position: target_position}

    def _add_portal_pairs(self, portal_density=0.005):
        """Add portal pairs
        Args:
            portal_density: Portal density as a proportion of total map area
        """
        # Calculate number of portal pairs
        total_area = self.height * self.width
        num_portal_pairs = max(1, int(total_area * portal_density))
        
        # Get all available positions (excluding walls, start and end points)
        valid_positions = [(y, x) for y in range(self.height) 
                          for x in range(self.width) 
                          if self.maze[y, x] == 0 
                          and (y, x) != self.start 
                          and (y, x) != self.end]
        
        # Randomly select positions for portal pairs
        portal_id = 3  # Start portal numbering from 3
        for _ in range(num_portal_pairs):
            if len(valid_positions) < 2:
                break
            
            # Select first portal position
            pos1 = random.choice(valid_positions)
            valid_positions.remove(pos1)
            
            # Select second portal position (with minimum distance from first)
            min_dist = min(self.height, self.width) // 4  # Minimum distance requirement
            valid_pos2 = [pos for pos in valid_positions 
                         if abs(pos[0] - pos1[0]) + abs(pos[1] - pos1[1]) >= min_dist]
            
            if not valid_pos2:
                continue
                
            pos2 = random.choice(valid_pos2)
            valid_positions.remove(pos2)
            
            # Set portal pair
            self.maze[pos1] = portal_id
            self.maze[pos2] = portal_id
            self.portals[pos1] = pos2
            self.portals[pos2] = pos1
            
            portal_id += 1

    def generate_maze_prim(self):
        """Generate basic maze using Prim's algorithm"""
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
        """Get adjacent cells"""
        y, x = cell
        neighbors = []
        for dy, dx in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            ny, nx = y + dy, x + dx
            if 0 < ny < self.height-1 and 0 < nx < self.width-1:
                neighbors.append((ny, nx))
        return neighbors

    def _add_extra_paths(self):
        """Add extra paths to increase path diversity"""
        extra_paths = (self.height * self.width) // 20
        for _ in range(extra_paths):
            y = random.randrange(1, self.height-1)
            x = random.randrange(1, self.width-1)
            if self.maze[y, x] == 1:
                self.maze[y, x] = 0

    def _ensure_path_near_endpoints(self):
        """Ensure paths exist near start and end points"""
        for point in [self.start, self.end]:
            y, x = point
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    self.maze[ny, nx] = 0

    def get_next_position(self, current_pos, next_pos):
        """Get next position, considering portal effects"""
        if next_pos in self.portals:
            return self.portals[next_pos]
        return next_pos

    def solve_dfs(self):
        """Solve maze using DFS"""
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
                            path.append(next_pos)  # Record portal entrance
                        if dfs(actual_next):
                            return True
                        if actual_next != next_pos:
                            path.pop()  # Portal entrance needs to be popped too
            
            path.pop()
            return False
        
        dfs(self.start)
        return path

    def solve_bfs(self):
        """Solve maze using BFS"""
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
        """Solve maze using A* algorithm"""
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
        """Visualize maze and paths"""
        paths = [p for p in [dfs_path, bfs_path, astar_path, rl_path] if p is not None]
        num_plots = len(paths)
        if num_plots == 0:
            num_plots = 1
            
        plt.figure(figsize=(5*num_plots, 5))
        
        titles = ['DFS Path', 'BFS Path', 'A* Path', 'RL Path']
        colors = ['r-', 'b-', 'g-', 'y-']
        
        for i, (path, title, color) in enumerate(zip(paths, titles, colors)):
            plt.subplot(1, num_plots, i+1)
            
            # Create RGB image
            rgb_maze = np.zeros((self.height, self.width, 3))
            
            # Set basic maze colors
            rgb_maze[self.maze == 1] = [0, 0, 0]  # Walls are black
            rgb_maze[self.maze == 0] = [1, 1, 1]  # Paths are white
            
            # Set same color for each portal pair
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
    # Create maze instance
    maze = EnhancedMaze(41, 61)  # Use smaller dimensions for better portal observation
    print("Generating maze...")
    
    # Generate basic maze
    maze.generate_maze_prim()
    
    # Add portals
    maze._add_portal_pairs(portal_density=0.01)  # Can adjust portal density
    
    print("Solving with different algorithms...")
    
    # DFS solution
    start_time = time.time()
    dfs_path = maze.solve_dfs()
    dfs_time = time.time() - start_time
    print(f"DFS solving time: {dfs_time:.3f} seconds")
    
    # BFS solution
    start_time = time.time()
    bfs_path = maze.solve_bfs()
    bfs_time = time.time() - start_time
    print(f"BFS solving time: {bfs_time:.3f} seconds")
    
    # A* solution
    start_time = time.time()
    astar_path = maze.solve_astar()
    astar_time = time.time() - start_time
    print(f"A* solving time: {astar_time:.3f} seconds")
    
    # RL solution
    start_time = time.time()
    # Create RL solver instance
    rl_solver = RLSolver(maze.maze, maze.start, maze.end, maze.portals, set())  # No traps, so pass empty set
    print("Training RL model...")
    rl_path = rl_solver.train(episodes=2000)  # Can adjust training episodes
    rl_time = time.time() - start_time
    print(f"RL solving time: {rl_time:.3f} seconds")
    
    print("\nPath length comparison:")
    print(f"DFS path length: {len(dfs_path)}")
    print(f"BFS path length: {len(bfs_path)}")
    print(f"A* path length: {len(astar_path)}")
    print(f"RL path length: {len(rl_path)}")
    
    print("\nSolving time comparison:")
    print(f"DFS time: {dfs_time:.3f} seconds")
    print(f"BFS time: {bfs_time:.3f} seconds")
    print(f"A* time: {astar_time:.3f} seconds")
    print(f"RL time: {rl_time:.3f} seconds")
    
    print("\nVisualizing results...")
    maze.visualize_maze(dfs_path, bfs_path, astar_path, rl_path)

if __name__ == "__main__":
    main()