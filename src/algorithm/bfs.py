"""
Breadth-First Search (BFS) pathfinding algorithm implementation.
BFS guarantees finding the shortest path in terms of number of steps (not cost).
"""

import sys
import os
from collections import deque
from typing import List, Tuple, Optional, Dict
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import GridEnvironment, Position


class BFSPlanner:
    """
    Breadth-First Search pathfinding planner.
    
    BFS explores all nodes at distance d before exploring nodes at distance d+1.
    This guarantees finding the shortest path in terms of number of steps.
    
    Time Complexity: O(V + E) where V is vertices and E is edges
    Space Complexity: O(V) for the queue and visited set
    """
    
    def __init__(self, environment: GridEnvironment):
        self.environment = environment
        self.nodes_expanded = 0
        self.path_cost = 0
        self.execution_time = 0.0
    
    def find_path(self, start: Position, goal: Position, time_step: int = 0) -> Optional[List[Position]]:
        """
        Find path from start to goal using BFS.
        
        Args:
            start: Starting position
            goal: Goal position
            time_step: Time step for dynamic obstacles
            
        Returns:
            List of positions forming the path, or None if no path exists
        """
        start_time = time.time()
        self.nodes_expanded = 0
        self.path_cost = 0
        
        # Initialize data structures
        queue = deque([(start, [start])])  # (position, path)
        visited = {start}
        
        # Set environment time step for dynamic obstacles
        self.environment.set_time(time_step)
        
        while queue:
            current_pos, path = queue.popleft()
            self.nodes_expanded += 1
            
            # Check if goal reached
            if current_pos == goal:
                self.path_cost = len(path) - 1  # Number of steps
                self.execution_time = time.time() - start_time
                return path
            
            # Explore neighbors
            neighbors = self.environment.get_neighbors(current_pos, time_step)
            
            for neighbor_pos, _ in neighbors:
                if neighbor_pos not in visited:
                    visited.add(neighbor_pos)
                    new_path = path + [neighbor_pos]
                    queue.append((neighbor_pos, new_path))
        
        # No path found
        self.execution_time = time.time() - start_time
        return None
    
    def get_statistics(self) -> Dict[str, float]:
        """Get algorithm performance statistics."""
        return {
            'nodes_expanded': self.nodes_expanded,
            'path_cost': self.path_cost,
            'execution_time': self.execution_time,
            'path_length': self.path_cost + 1 if self.path_cost > 0 else 0
        }


class BFSWithCosts(BFSPlanner):
    """
    BFS variant that considers movement costs.
    While still using BFS exploration order, it tracks actual path costs.
    """
    
    def find_path(self, start: Position, goal: Position, time_step: int = 0) -> Optional[List[Position]]:
        """
        Find path from start to goal using BFS with cost tracking.
        
        Args:
            start: Starting position
            goal: Goal position
            time_step: Time step for dynamic obstacles
            
        Returns:
            List of positions forming the path, or None if no path exists
        """
        start_time = time.time()
        self.nodes_expanded = 0
        self.path_cost = 0
        
        # Initialize data structures
        queue = deque([(start, [start], 0)])  # (position, path, cost)
        visited = {start}
        
        # Set environment time step for dynamic obstacles
        self.environment.set_time(time_step)
        
        while queue:
            current_pos, path, cost = queue.popleft()
            self.nodes_expanded += 1
            
            # Check if goal reached
            if current_pos == goal:
                self.path_cost = cost
                self.execution_time = time.time() - start_time
                return path
            
            # Explore neighbors
            neighbors = self.environment.get_neighbors(current_pos, time_step)
            
            for neighbor_pos, movement_cost in neighbors:
                if neighbor_pos not in visited:
                    visited.add(neighbor_pos)
                    new_path = path + [neighbor_pos]
                    new_cost = cost + movement_cost
                    queue.append((neighbor_pos, new_path, new_cost))
        
        # No path found
        self.execution_time = time.time() - start_time
        return None


def run_bfs_test():
    """Test function for BFS implementation."""
    from environment import create_test_environment
    
    # Create test environment
    env = create_test_environment("small")
    
    # Test BFS
    print("Testing BFS...")
    bfs = BFSPlanner(env)
    
    start = Position(0, 0)
    goal = Position(9, 9)
    
    path = bfs.find_path(start, goal)
    
    if path:
        print(f"Path found with {len(path)} steps")
        print(f"Nodes expanded: {bfs.nodes_expanded}")
        print(f"Execution time: {bfs.execution_time:.4f}s")
        print("Path:", [str(p) for p in path])
        
        # Visualize
        env.visualize(agent_pos=start, path=path)
    else:
        print("No path found!")
    
    # Test BFS with costs
    print("\nTesting BFS with costs...")
    bfs_cost = BFSWithCosts(env)
    path_cost = bfs_cost.find_path(start, goal)
    
    if path_cost:
        print(f"Path found with cost {bfs_cost.path_cost}")
        print(f"Nodes expanded: {bfs_cost.nodes_expanded}")
        print(f"Execution time: {bfs_cost.execution_time:.4f}s")


if __name__ == "__main__":
    run_bfs_test()
