"""
A* pathfinding algorithm implementation with admissible heuristics.
A* combines uniform cost search with heuristic guidance for optimal pathfinding.
"""

import sys
import os
import heapq
import math
from typing import List, Tuple, Optional, Dict, Callable
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import GridEnvironment, Position


class AStarNode:
    """Node for A* search with f, g, and h values."""
    
    def __init__(self, position: Position, g_cost: float, h_cost: float, parent: Optional['AStarNode'] = None):
        self.position = position
        self.g_cost = g_cost  # Cost from start to this node
        self.h_cost = h_cost  # Heuristic estimate to goal
        self.f_cost = g_cost + h_cost  # Total estimated cost
        self.parent = parent
    
    def __lt__(self, other):
        """Less than comparison for heap ordering."""
        if self.f_cost != other.f_cost:
            return self.f_cost < other.f_cost
        # Tie-breaking by h_cost (prefer nodes closer to goal)
        return self.h_cost < other.h_cost
    
    def __eq__(self, other):
        """Equality comparison."""
        return isinstance(other, AStarNode) and self.position == other.position
    
    def __hash__(self):
        """Hash for set operations."""
        return hash(self.position)


class AStarPlanner:
    """
    A* pathfinding planner with admissible heuristics.
    
    A* combines UCS (g-cost) with heuristic guidance (h-cost) to find optimal paths
    while exploring fewer nodes than UCS alone.
    
    Time Complexity: O(b^d) where b is branching factor and d is depth of optimal solution
    Space Complexity: O(b^d)
    """
    
    def __init__(self, environment: GridEnvironment, heuristic: Optional[Callable] = None):
        self.environment = environment
        self.nodes_expanded = 0
        self.path_cost = 0
        self.execution_time = 0.0
        
        # Default to Manhattan distance if no heuristic provided
        if heuristic is None:
            self.heuristic = self.manhattan_distance
        else:
            self.heuristic = heuristic
    
    def manhattan_distance(self, pos1: Position, pos2: Position) -> float:
        """Manhattan distance heuristic (admissible for grid with 4-connected movement)."""
        return abs(pos1.x - pos2.x) + abs(pos1.y - pos2.y)
    
    def euclidean_distance(self, pos1: Position, pos2: Position) -> float:
        """Euclidean distance heuristic (admissible but not consistent for grid)."""
        dx = pos1.x - pos2.x
        dy = pos1.y - pos2.y
        return math.sqrt(dx * dx + dy * dy)
    
    def diagonal_distance(self, pos1: Position, pos2: Position) -> float:
        """Diagonal distance heuristic (admissible for grid with 8-connected movement)."""
        dx = abs(pos1.x - pos2.x)
        dy = abs(pos1.y - pos2.y)
        return dx + dy + (math.sqrt(2) - 2) * min(dx, dy)
    
    def zero_heuristic(self, pos1: Position, pos2: Position) -> float:
        """Zero heuristic (reduces A* to UCS)."""
        return 0.0
    
    def find_path(self, start: Position, goal: Position, time_step: int = 0) -> Optional[List[Position]]:
        """
        Find path from start to goal using A*.
        
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
        open_set = []  # Priority queue (min-heap)
        open_set_dict = {}  # Dictionary for quick lookup
        closed_set = set()  # Set of explored positions
        
        # Create start node
        start_node = AStarNode(start, 0.0, self.heuristic(start, goal))
        heapq.heappush(open_set, start_node)
        open_set_dict[start] = start_node
        
        # Set environment time step for dynamic obstacles
        self.environment.set_time(time_step)
        
        while open_set:
            # Get node with minimum f-cost
            current_node = heapq.heappop(open_set)
            
            # Skip if already processed
            if current_node.position in closed_set:
                continue
            
            closed_set.add(current_node.position)
            self.nodes_expanded += 1
            
            # Check if goal reached
            if current_node.position == goal:
                # Reconstruct path
                path = []
                node = current_node
                while node is not None:
                    path.append(node.position)
                    node = node.parent
                path.reverse()
                
                self.path_cost = current_node.g_cost
                self.execution_time = time.time() - start_time
                return path
            
            # Explore neighbors
            neighbors = self.environment.get_neighbors(current_node.position, time_step)
            
            for neighbor_pos, movement_cost in neighbors:
                # Skip if already in closed set
                if neighbor_pos in closed_set:
                    continue
                
                # Calculate costs
                tentative_g_cost = current_node.g_cost + movement_cost
                h_cost = self.heuristic(neighbor_pos, goal)
                f_cost = tentative_g_cost + h_cost
                
                # Check if neighbor is already in open set
                if neighbor_pos in open_set_dict:
                    existing_node = open_set_dict[neighbor_pos]
                    
                    # If we found a better path to this neighbor
                    if tentative_g_cost < existing_node.g_cost:
                        # Update the node
                        existing_node.g_cost = tentative_g_cost
                        existing_node.f_cost = tentative_g_cost + h_cost
                        existing_node.parent = current_node
                        
                        # Re-heapify (not optimal, but simple)
                        heapq.heapify(open_set)
                else:
                    # Add new node to open set
                    new_node = AStarNode(neighbor_pos, tentative_g_cost, h_cost, current_node)
                    heapq.heappush(open_set, new_node)
                    open_set_dict[neighbor_pos] = new_node
        
        # No path found
        self.execution_time = time.time() - start_time
        return None
    
    def get_statistics(self) -> Dict[str, float]:
        """Get algorithm performance statistics."""
        return {
            'nodes_expanded': self.nodes_expanded,
            'path_cost': self.path_cost,
            'execution_time': self.execution_time
        }


class AStarWithTieBreaking(AStarPlanner):
    """
    A* with tie-breaking strategies for better performance.
    """
    
    def __init__(self, environment: GridEnvironment, heuristic: Optional[Callable] = None):
        super().__init__(environment, heuristic)
        self.tie_breaking_factor = 1.0 + 1e-6
    
    def manhattan_distance_with_tie_breaking(self, pos1: Position, pos2: Position) -> float:
        """Manhattan distance with small tie-breaking factor."""
        base_cost = self.manhattan_distance(pos1, pos2)
        return base_cost * self.tie_breaking_factor


class AStarWeighted(AStarPlanner):
    """
    Weighted A* for faster (but suboptimal) pathfinding.
    """
    
    def __init__(self, environment: GridEnvironment, weight: float = 1.5, heuristic: Optional[Callable] = None):
        super().__init__(environment, heuristic)
        self.weight = weight  # Weight for heuristic (w > 1 for suboptimal but faster search)
    
    def find_path(self, start: Position, goal: Position, time_step: int = 0) -> Optional[List[Position]]:
        """Weighted A* search with f = g + w * h."""
        start_time = time.time()
        self.nodes_expanded = 0
        self.path_cost = 0
        
        # Initialize data structures
        open_set = []
        open_set_dict = {}
        closed_set = set()
        
        # Create start node with weighted heuristic
        start_h = self.heuristic(start, goal)
        start_node = AStarNode(start, 0.0, start_h * self.weight)
        heapq.heappush(open_set, start_node)
        open_set_dict[start] = start_node
        
        self.environment.set_time(time_step)
        
        while open_set:
            current_node = heapq.heappop(open_set)
            
            if current_node.position in closed_set:
                continue
            
            closed_set.add(current_node.position)
            self.nodes_expanded += 1
            
            if current_node.position == goal:
                # Reconstruct path
                path = []
                node = current_node
                while node is not None:
                    path.append(node.position)
                    node = node.parent
                path.reverse()
                
                self.path_cost = current_node.g_cost
                self.execution_time = time.time() - start_time
                return path
            
            # Explore neighbors
            neighbors = self.environment.get_neighbors(current_node.position, time_step)
            
            for neighbor_pos, movement_cost in neighbors:
                if neighbor_pos in closed_set:
                    continue
                
                tentative_g_cost = current_node.g_cost + movement_cost
                h_cost = self.heuristic(neighbor_pos, goal)
                f_cost = tentative_g_cost + (h_cost * self.weight)
                
                if neighbor_pos in open_set_dict:
                    existing_node = open_set_dict[neighbor_pos]
                    
                    if tentative_g_cost < existing_node.g_cost:
                        existing_node.g_cost = tentative_g_cost
                        existing_node.f_cost = tentative_g_cost + (h_cost * self.weight)
                        existing_node.parent = current_node
                        heapq.heapify(open_set)
                else:
                    new_node = AStarNode(neighbor_pos, tentative_g_cost, h_cost * self.weight, current_node)
                    heapq.heappush(open_set, new_node)
                    open_set_dict[neighbor_pos] = new_node
        
        self.execution_time = time.time() - start_time
        return None


def run_astar_test():
    """Test function for A* implementation."""
    from environment import create_test_environment
    
    # Create test environment
    env = create_test_environment("small")
    
    # Test A* with Manhattan distance
    print("Testing A* with Manhattan distance...")
    astar_manhattan = AStarPlanner(env)
    
    start = Position(0, 0)
    goal = Position(9, 9)
    
    path = astar_manhattan.find_path(start, goal)
    
    if path:
        print(f"Path found with cost {astar_manhattan.path_cost}")
        print(f"Nodes expanded: {astar_manhattan.nodes_expanded}")
        print(f"Execution time: {astar_manhattan.execution_time:.4f}s")
        print("Path:", [str(p) for p in path])
        
        # Visualize
        env.visualize(agent_pos=start, path=path)
    else:
        print("No path found!")
    
    # Test A* with Euclidean distance
    print("\nTesting A* with Euclidean distance...")
    astar_euclidean = AStarPlanner(env)
    astar_euclidean.heuristic = astar_euclidean.euclidean_distance
    
    path_euclidean = astar_euclidean.find_path(start, goal)
    
    if path_euclidean:
        print(f"Path found with cost {astar_euclidean.path_cost}")
        print(f"Nodes expanded: {astar_euclidean.nodes_expanded}")
        print(f"Execution time: {astar_euclidean.execution_time:.4f}s")
    
    # Test Weighted A*
    print("\nTesting Weighted A* (weight=1.5)...")
    astar_weighted = AStarWeighted(env, weight=1.5)
    
    path_weighted = astar_weighted.find_path(start, goal)
    
    if path_weighted:
        print(f"Path found with cost {astar_weighted.path_cost}")
        print(f"Nodes expanded: {astar_weighted.nodes_expanded}")
        print(f"Execution time: {astar_weighted.execution_time:.4f}s")


if __name__ == "__main__":
    run_astar_test()
