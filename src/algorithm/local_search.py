"""
Local search replanning strategies for dynamic environments.
Implements hill-climbing with random restarts and simulated annealing.
"""

import sys
import os
import random
import math
from typing import List, Tuple, Optional, Dict, Set
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import GridEnvironment, Position
from .astar import AStarPlanner


class LocalSearchPlanner:
    """
    Local search planner with hill-climbing and random restarts.
    
    This planner is designed for dynamic environments where obstacles
    may appear or move, requiring frequent replanning.
    """
    
    def __init__(self, environment: GridEnvironment, max_iterations: int = 1000, 
                 restart_probability: float = 0.1):
        self.environment = environment
        self.max_iterations = max_iterations
        self.restart_probability = restart_probability
        self.nodes_expanded = 0
        self.path_cost = 0
        self.execution_time = 0.0
        self.replan_count = 0
        
        # A* planner for finding initial paths
        self.astar = AStarPlanner(environment)
    
    def generate_random_path(self, start: Position, goal: Position, max_length: int = 100) -> List[Position]:
        """Generate a random valid path from start to goal."""
        path = [start]
        current = start
        
        for _ in range(max_length):
            if current == goal:
                break
            
            # Get valid neighbors
            neighbors = self.environment.get_neighbors(current, self.environment.time_step)
            valid_neighbors = [(pos, cost) for pos, cost in neighbors if pos not in path]
            
            if not valid_neighbors:
                break
            
            # Choose random neighbor
            next_pos, _ = random.choice(valid_neighbors)
            path.append(next_pos)
            current = next_pos
        
        return path if current == goal else None
    
    def calculate_path_cost(self, path: List[Position]) -> float:
        """Calculate total cost of a path."""
        if len(path) < 2:
            return 0.0
        
        total_cost = 0.0
        for i in range(len(path) - 1):
            neighbors = self.environment.get_neighbors(path[i], self.environment.time_step)
            for neighbor_pos, cost in neighbors:
                if neighbor_pos == path[i + 1]:
                    total_cost += cost
                    break
        
        return total_cost
    
    def is_valid_path(self, path: List[Position], time_step: int) -> bool:
        """Check if path is valid (no obstacles, connected)."""
        if not path or len(path) < 2:
            return False
        
        # Check if path is connected
        for i in range(len(path) - 1):
            current = path[i]
            next_pos = path[i + 1]
            
            # Check if positions are neighbors
            neighbors = self.environment.get_neighbors(current, time_step)
            neighbor_positions = [pos for pos, _ in neighbors]
            
            if next_pos not in neighbor_positions:
                return False
            
            # Check if next position is not blocked
            if self.environment.is_obstacle(next_pos, time_step):
                return False
        
        return True
    
    def hill_climbing_step(self, current_path: List[Position], goal: Position) -> Optional[List[Position]]:
        """Perform one step of hill-climbing to improve the path."""
        if len(current_path) < 3:
            return current_path
        
        best_path = current_path
        best_cost = self.calculate_path_cost(current_path)
        
        # Try local modifications
        for i in range(1, len(current_path) - 1):
            # Try to bypass this point
            new_path = current_path[:i] + current_path[i+1:]
            
            if self.is_valid_path(new_path, self.environment.time_step):
                new_cost = self.calculate_path_cost(new_path)
                if new_cost < best_cost:
                    best_path = new_path
                    best_cost = new_cost
        
        # Try to add intermediate points
        for i in range(len(current_path) - 1):
            current = current_path[i]
            next_pos = current_path[i + 1]
            
            # Try to find a shorter route between current and next
            neighbors_current = [pos for pos, _ in self.environment.get_neighbors(current, self.environment.time_step)]
            neighbors_next = [pos for pos, _ in self.environment.get_neighbors(next_pos, self.environment.time_step)]
            
            # Find common neighbors
            common_neighbors = set(neighbors_current) & set(neighbors_next)
            
            for intermediate in common_neighbors:
                if intermediate not in current_path:
                    new_path = current_path[:i+1] + [intermediate] + current_path[i+1:]
                    
                    if self.is_valid_path(new_path, self.environment.time_step):
                        new_cost = self.calculate_path_cost(new_path)
                        if new_cost < best_cost:
                            best_path = new_path
                            best_cost = new_cost
        
        return best_path
    
    def find_path_with_hill_climbing(self, start: Position, goal: Position, 
                                   time_step: int = 0) -> Optional[List[Position]]:
        """Find path using hill-climbing with random restarts."""
        start_time = time.time()
        self.nodes_expanded = 0
        self.path_cost = 0
        self.replan_count = 0
        
        # Set environment time step
        self.environment.set_time(time_step)
        
        # Try to find initial path with A*
        initial_path = self.astar.find_path(start, goal, time_step)
        if initial_path is None:
            # If A* fails, try random path generation
            initial_path = self.generate_random_path(start, goal)
            if initial_path is None:
                self.execution_time = time.time() - start_time
                return None
        
        best_path = initial_path
        best_cost = self.calculate_path_cost(best_path)
        self.nodes_expanded += self.astar.nodes_expanded
        
        # Hill-climbing with random restarts
        for iteration in range(self.max_iterations):
            # Random restart with some probability
            if random.random() < self.restart_probability:
                # Generate new random starting path
                current_path = self.generate_random_path(start, goal)
                if current_path is None:
                    continue
                self.replan_count += 1
            else:
                current_path = best_path.copy()
            
            # Hill-climbing
            improved_path = self.hill_climbing_step(current_path, goal)
            
            if improved_path is not None:
                improved_cost = self.calculate_path_cost(improved_path)
                
                if improved_cost < best_cost:
                    best_path = improved_path
                    best_cost = improved_cost
        
        self.path_cost = best_cost
        self.execution_time = time.time() - start_time
        return best_path
    
    def find_path(self, start: Position, goal: Position, time_step: int = 0) -> Optional[List[Position]]:
        """Main pathfinding method."""
        return self.find_path_with_hill_climbing(start, goal, time_step)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get algorithm performance statistics."""
        return {
            'nodes_expanded': self.nodes_expanded,
            'path_cost': self.path_cost,
            'execution_time': self.execution_time,
            'replan_count': self.replan_count
        }


class SimulatedAnnealingPlanner:
    """
    Simulated annealing planner for dynamic environments.
    """
    
    def __init__(self, environment: GridEnvironment, initial_temperature: float = 100.0,
                 cooling_rate: float = 0.95, max_iterations: int = 1000):
        self.environment = environment
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.nodes_expanded = 0
        self.path_cost = 0
        self.execution_time = 0.0
        
        self.astar = AStarPlanner(environment)
    
    def calculate_path_cost(self, path: List[Position]) -> float:
        """Calculate total cost of a path."""
        if len(path) < 2:
            return 0.0
        
        total_cost = 0.0
        for i in range(len(path) - 1):
            neighbors = self.environment.get_neighbors(path[i], self.environment.time_step)
            for neighbor_pos, cost in neighbors:
                if neighbor_pos == path[i + 1]:
                    total_cost += cost
                    break
        
        return total_cost
    
    def generate_neighbor_path(self, current_path: List[Position]) -> List[Position]:
        """Generate a neighbor path by making a small modification."""
        if len(current_path) < 3:
            return current_path.copy()
        
        new_path = current_path.copy()
        
        # Randomly choose modification type
        modification = random.choice(['remove', 'add', 'swap'])
        
        if modification == 'remove' and len(new_path) > 2:
            # Remove a random intermediate point
            idx = random.randint(1, len(new_path) - 2)
            new_path.pop(idx)
            
        elif modification == 'add' and len(new_path) > 1:
            # Add a random intermediate point
            idx = random.randint(0, len(new_path) - 2)
            current = new_path[idx]
            next_pos = new_path[idx + 1]
            
            neighbors_current = [pos for pos, _ in self.environment.get_neighbors(current, self.environment.time_step)]
            neighbors_next = [pos for pos, _ in self.environment.get_neighbors(next_pos, self.environment.time_step)]
            
            common_neighbors = set(neighbors_current) & set(neighbors_next)
            if common_neighbors:
                intermediate = random.choice(list(common_neighbors))
                if intermediate not in new_path:
                    new_path.insert(idx + 1, intermediate)
        
        elif modification == 'swap' and len(new_path) > 2:
            # Swap two random intermediate points
            idx1 = random.randint(1, len(new_path) - 2)
            idx2 = random.randint(1, len(new_path) - 2)
            new_path[idx1], new_path[idx2] = new_path[idx2], new_path[idx1]
        
        return new_path
    
    def find_path(self, start: Position, goal: Position, time_step: int = 0) -> Optional[List[Position]]:
        """Find path using simulated annealing."""
        start_time = time.time()
        self.nodes_expanded = 0
        self.path_cost = 0
        
        # Set environment time step
        self.environment.set_time(time_step)
        
        # Get initial solution with A*
        current_path = self.astar.find_path(start, goal, time_step)
        if current_path is None:
            self.execution_time = time.time() - start_time
            return None
        
        best_path = current_path.copy()
        best_cost = self.calculate_path_cost(best_path)
        current_cost = best_cost
        
        temperature = self.initial_temperature
        self.nodes_expanded += self.astar.nodes_expanded
        
        for iteration in range(self.max_iterations):
            # Generate neighbor
            neighbor_path = self.generate_neighbor_path(current_path)
            
            # Check if neighbor is valid
            if self.is_valid_path(neighbor_path, time_step):
                neighbor_cost = self.calculate_path_cost(neighbor_path)
                
                # Accept or reject based on simulated annealing criteria
                if neighbor_cost < current_cost or random.random() < math.exp(-(neighbor_cost - current_cost) / temperature):
                    current_path = neighbor_path
                    current_cost = neighbor_cost
                    
                    # Update best solution
                    if current_cost < best_cost:
                        best_path = current_path.copy()
                        best_cost = current_cost
            
            # Cool down
            temperature *= self.cooling_rate
        
        self.path_cost = best_cost
        self.execution_time = time.time() - start_time
        return best_path
    
    def is_valid_path(self, path: List[Position], time_step: int) -> bool:
        """Check if path is valid."""
        if not path or len(path) < 2:
            return False
        
        for i in range(len(path) - 1):
            current = path[i]
            next_pos = path[i + 1]
            
            neighbors = self.environment.get_neighbors(current, time_step)
            neighbor_positions = [pos for pos, _ in neighbors]
            
            if next_pos not in neighbor_positions:
                return False
            
            if self.environment.is_obstacle(next_pos, time_step):
                return False
        
        return True
    
    def get_statistics(self) -> Dict[str, float]:
        """Get algorithm performance statistics."""
        return {
            'nodes_expanded': self.nodes_expanded,
            'path_cost': self.path_cost,
            'execution_time': self.execution_time
        }


def run_local_search_test():
    """Test function for local search implementations."""
    from environment import create_test_environment
    
    # Create test environment
    env = create_test_environment("small")
    
    start = Position(0, 0)
    goal = Position(9, 9)
    
    # Test hill-climbing with random restarts
    print("Testing Hill-climbing with random restarts...")
    hill_climber = LocalSearchPlanner(env, max_iterations=500)
    
    path = hill_climber.find_path(start, goal)
    
    if path:
        print(f"Path found with cost {hill_climber.path_cost}")
        print(f"Nodes expanded: {hill_climber.nodes_expanded}")
        print(f"Execution time: {hill_climber.execution_time:.4f}s")
        print(f"Replan count: {hill_climber.replan_count}")
        print("Path:", [str(p) for p in path])
        
        # Visualize
        env.visualize(agent_pos=start, path=path)
    else:
        print("No path found!")
    
    # Test simulated annealing
    print("\nTesting Simulated Annealing...")
    sa_planner = SimulatedAnnealingPlanner(env)
    
    path_sa = sa_planner.find_path(start, goal)
    
    if path_sa:
        print(f"Path found with cost {sa_planner.path_cost}")
        print(f"Nodes expanded: {sa_planner.nodes_expanded}")
        print(f"Execution time: {sa_planner.execution_time:.4f}s")


if __name__ == "__main__":
    run_local_search_test()
