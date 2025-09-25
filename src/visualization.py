"""
Visualization system for the autonomous delivery agent.
Provides both text-based and graphical visualization of paths and environments.
"""

import sys
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import List, Optional, Dict, Any
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import GridEnvironment, Position


class PathVisualizer:
    """Visualization system for paths and environments."""
    
    def __init__(self, environment: GridEnvironment):
        self.environment = environment
        self.fig = None
        self.ax = None
        
        # Color scheme
        self.colors = {
            'empty': '#FFFFFF',
            'obstacle': '#000000',
            'terrain': '#8B4513',
            'dynamic_obstacle': '#FF0000',
            'agent': '#00FF00',
            'path': '#0000FF',
            'start': '#00FF00',
            'goal': '#FF0000'
        }
    
    def create_grid_visualization(self, agent_pos: Optional[Position] = None, 
                                path: Optional[List[Position]] = None,
                                time_step: int = 0) -> np.ndarray:
        """Create a 2D array representing the environment for visualization."""
        height, width = self.environment.height, self.environment.width
        grid = np.zeros((height, width, 3), dtype=float)
        
        # Set environment time step
        self.environment.set_time(time_step)
        
        for y in range(height):
            for x in range(width):
                pos = Position(x, y)
                
                if pos in self.environment.static_obstacles:
                    # Static obstacle - black
                    grid[y, x] = [0, 0, 0]
                elif pos in self.environment.get_all_dynamic_obstacle_positions(time_step):
                    # Dynamic obstacle - red
                    grid[y, x] = [1, 0, 0]
                elif self.environment.grid[y, x] > 1:
                    # Terrain with cost > 1 - brown (darker for higher cost)
                    cost = self.environment.grid[y, x]
                    intensity = min(0.5 + (cost - 2) * 0.1, 1.0)
                    grid[y, x] = [intensity * 0.55, intensity * 0.27, intensity * 0.07]  # Brown
                else:
                    # Empty cell - white
                    grid[y, x] = [1, 1, 1]
        
        # Overlay path
        if path:
            for i, pos in enumerate(path):
                if 0 <= pos.x < width and 0 <= pos.y < height:
                    # Path - blue (gradient from light to dark)
                    intensity = 0.3 + 0.7 * (i / len(path))
                    grid[pos.y, pos.x] = [0, 0, intensity]
        
        # Overlay agent position
        if agent_pos and 0 <= agent_pos.x < width and 0 <= agent_pos.y < height:
            grid[agent_pos.y, agent_pos.x] = [0, 1, 0]  # Green
        
        # Overlay start and goal if provided
        if path:
            start_pos = path[0]
            goal_pos = path[-1]
            if 0 <= start_pos.x < width and 0 <= start_pos.y < height:
                grid[start_pos.y, start_pos.x] = [0, 1, 0]  # Green
            if 0 <= goal_pos.x < width and 0 <= goal_pos.y < height:
                grid[goal_pos.y, goal_pos.x] = [1, 0, 0]  # Red
        
        return grid
    
    def plot_static_environment(self, agent_pos: Optional[Position] = None, 
                              path: Optional[List[Position]] = None,
                              time_step: int = 0, title: str = "Environment"):
        """Create a static plot of the environment."""
        grid = self.create_grid_visualization(agent_pos, path, time_step)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(grid, origin='upper')
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor='white', label='Empty'),
            plt.Rectangle((0, 0), 1, 1, facecolor='black', label='Static Obstacle'),
            plt.Rectangle((0, 0), 1, 1, facecolor='red', label='Dynamic Obstacle'),
            plt.Rectangle((0, 0), 1, 1, facecolor='brown', label='Terrain (cost > 1)'),
            plt.Rectangle((0, 0), 1, 1, facecolor='green', label='Agent/Start'),
            plt.Rectangle((0, 0), 1, 1, facecolor='blue', label='Path')
        ]
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
        
        plt.tight_layout()
        plt.show()
    
    def animate_dynamic_environment(self, paths: Dict[int, List[Position]], 
                                  agent_positions: Dict[int, Position],
                                  title: str = "Dynamic Environment Animation"):
        """Create an animation of the dynamic environment."""
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        def animate(frame):
            self.ax.clear()
            
            if frame in agent_positions and frame in paths:
                grid = self.create_grid_visualization(
                    agent_positions[frame], paths[frame], frame
                )
                self.ax.imshow(grid, origin='upper')
                self.ax.set_title(f"{title} - Time Step {frame}")
                self.ax.set_xlabel('X')
                self.ax.set_ylabel('Y')
            
            return self.ax.images
        
        # Create animation
        frames = max(len(agent_positions), len(paths))
        anim = animation.FuncAnimation(self.fig, animate, frames=frames, 
                                     interval=500, blit=False, repeat=True)
        
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def compare_algorithms_visualization(self, results: Dict[str, Dict[str, Any]], 
                                       start: Position, goal: Position):
        """Create a comparison visualization of different algorithms."""
        num_algorithms = len(results)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, (alg_name, result) in enumerate(results.items()):
            if i >= 4:  # Limit to 4 algorithms
                break
            
            ax = axes[i]
            
            if result.get('path_found', False) and result.get('path'):
                # Convert string positions back to Position objects
                path_positions = []
                for pos_str in result['path']:
                    x, y = map(int, pos_str.strip('()').split(', '))
                    path_positions.append(Position(x, y))
                
                grid = self.create_grid_visualization(None, path_positions)
                ax.imshow(grid, origin='upper')
                ax.set_title(f"{alg_name.upper()}\n"
                           f"Cost: {result.get('path_cost', 0):.2f}, "
                           f"Nodes: {result.get('nodes_expanded', 0)}, "
                           f"Time: {result.get('execution_time', 0):.3f}s")
            else:
                # No path found
                grid = self.create_grid_visualization()
                ax.imshow(grid, origin='upper')
                ax.set_title(f"{alg_name.upper()}\nNo path found")
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
        
        # Hide unused subplots
        for i in range(len(results), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_performance_comparison(self, benchmark_results: Dict[str, Any]):
        """Create performance comparison plots."""
        algorithms = []
        costs = []
        nodes_expanded = []
        execution_times = []
        
        for test_name, test_data in benchmark_results.items():
            results = test_data.get('results', {})
            for alg_name, result in results.items():
                if result.get('path_found', False):
                    algorithms.append(f"{alg_name}_{test_name}")
                    costs.append(result.get('path_cost', 0))
                    nodes_expanded.append(result.get('nodes_expanded', 0))
                    execution_times.append(result.get('execution_time', 0))
        
        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # Path cost comparison
        ax1.bar(range(len(algorithms)), costs)
        ax1.set_title('Path Cost Comparison')
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Path Cost')
        ax1.set_xticks(range(len(algorithms)))
        ax1.set_xticklabels(algorithms, rotation=45, ha='right')
        
        # Nodes expanded comparison
        ax2.bar(range(len(algorithms)), nodes_expanded)
        ax2.set_title('Nodes Expanded Comparison')
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel('Nodes Expanded')
        ax2.set_xticks(range(len(algorithms)))
        ax2.set_xticklabels(algorithms, rotation=45, ha='right')
        
        # Execution time comparison
        ax3.bar(range(len(algorithms)), execution_times)
        ax3.set_title('Execution Time Comparison')
        ax3.set_xlabel('Algorithm')
        ax3.set_ylabel('Execution Time (s)')
        ax3.set_xticks(range(len(algorithms)))
        ax3.set_xticklabels(algorithms, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
    
    def save_visualization(self, filename: str, agent_pos: Optional[Position] = None,
                          path: Optional[List[Position]] = None, time_step: int = 0):
        """Save visualization to file."""
        grid = self.create_grid_visualization(agent_pos, path, time_step)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(grid, origin='upper')
        plt.title(f"Environment Visualization - Time Step {time_step}")
        plt.xlabel('X')
        plt.ylabel('Y')
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()


def create_dynamic_replanning_visualization(agent, algorithm_name: str = 'local'):
    """Create a visualization of dynamic replanning."""
    print(f"Creating dynamic replanning visualization for {algorithm_name}...")
    
    # Load dynamic environment
    agent.load_environment('maps/dynamic.txt')
    visualizer = PathVisualizer(agent.environment)
    
    start = Position(0, 0)
    goal = Position(19, 19)
    
    # Simulate agent movement with replanning
    current_pos = start
    agent_positions = {}
    paths = {}
    time_step = 0
    
    while current_pos != goal and time_step < 15:
        # Find path from current position
        result = agent.run_algorithm(algorithm_name, current_pos, goal, time_step)
        
        if not result['path_found'] or len(result['path']) < 2:
            break
        
        # Convert string positions back to Position objects
        path_positions = []
        for pos_str in result['path']:
            x, y = map(int, pos_str.strip('()').split(', '))
            path_positions.append(Position(x, y))
        
        # Store for animation
        agent_positions[time_step] = current_pos
        paths[time_step] = path_positions
        
        # Move one step along the path
        next_pos_str = result['path'][1]  # Skip current position
        x, y = map(int, next_pos_str.strip('()').split(', '))
        next_pos = Position(x, y)
        
        # Check if next position is still valid
        if agent.environment.is_obstacle(next_pos, time_step + 1):
            print(f"Time step {time_step}: Path blocked! Replanning...")
            time_step += 1
            agent.environment.advance_time()
            continue
        
        # Move to next position
        current_pos = next_pos
        time_step += 1
        agent.environment.advance_time()
    
    # Create animation
    anim = visualizer.animate_dynamic_environment(paths, agent_positions, 
                                                f"Dynamic Replanning - {algorithm_name}")
    
    return anim


if __name__ == "__main__":
    # Test visualization
    from environment import create_test_environment
    
    env = create_test_environment("small")
    visualizer = PathVisualizer(env)
    
    # Test static visualization
    start = Position(0, 0)
    goal = Position(9, 9)
    test_path = [Position(0, 0), Position(1, 0), Position(2, 0), Position(3, 0), Position(4, 0),
                 Position(5, 0), Position(6, 0), Position(7, 0), Position(8, 0), Position(9, 0),
                 Position(9, 1), Position(9, 2), Position(9, 3), Position(9, 4), Position(9, 5),
                 Position(9, 6), Position(9, 7), Position(9, 8), Position(9, 9)]
    
    visualizer.plot_static_environment(start, test_path, title="Test Visualization")
