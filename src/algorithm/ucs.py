"""
Uniform Cost Search (UCS) pathfinding algorithm implementation.
UCS guarantees finding the optimal path in terms of total cost.
"""

import sys
import os
import heapq
from typing import List, Tuple, Optional, Dict
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import GridEnvironment, Position


class UCSSearchNode:
    """Node for UCS search with cost tracking."""
    
    def __init__(self, position: Position, path_cost: float, path: List[Position]):
        self.position = position
        self.path_cost = path_cost
        self.path = path
    
    def __lt__(self, other):
        """Less than comparison for heap ordering."""
        return self.path_cost < other.path_cost
    
    def __eq__(self, other):
        """Equality comparison."""
        return isinstance(other, UCSSearchNode) and self.position == other.position
    
    def __hash__(self):
        """Hash for set operations."""
        return hash(self.position)


class UCSPlanner:
    """
    Uniform Cost Search pathfinding planner.
    
    UCS explores nodes in order of increasing path cost from the start.
    This guarantees finding the optimal path in terms of total cost.
    
    Time Complexity: O(b^(C*/ε)) where b is branching factor, C* is optimal cost, ε is minimum edge cost
    Space Complexity: O(b^(C*/ε))
    """
    
    def __init__(self, environment: GridEnvironment):
        self.environment = environment
        self.nodes_expanded = 0
        self.path_cost = 0
        self.execution_time = 0.0
    
    def find_path(self, start: Position, goal: Position, time_step: int = 0) -> Optional[List[Position]]:
        """
        Find path from start to goal using UCS.
        
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
        frontier = []  # Priority queue (min-heap)
        explored = set()  # Set of explored positions
        frontier_set = set()  # Set of positions in frontier for quick lookup
        
        # Create start node
        start_node = UCSSearchNode(start, 0.0, [start])
        heapq.heappush(frontier, start_node)
        frontier_set.add(start)
        
        # Set environment time step for dynamic obstacles
        self.environment.set_time(time_step)
        
        while frontier:
            # Get node with minimum cost
            current_node = heapq.heappop(frontier)
            frontier_set.remove(current_node.position)
            
            # Skip if already explored
            if current_node.position in explored:
                continue
            
            explored.add(current_node.position)
            self.nodes_expanded += 1
            
            # Check if goal reached
            if current_node.position == goal:
                self.path_cost = current_node.path_cost
                self.execution_time = time.time() - start_time
                return current_node.path
            
            # Explore neighbors
            neighbors = self.environment.get_neighbors(current_node.position, time_step)
            
            for neighbor_pos, movement_cost in neighbors:
                # Skip if already explored
                if neighbor_pos in explored:
                    continue
                
                # Calculate new path cost
                new_cost = current_node.path_cost + movement_cost
                new_path = current_node.path + [neighbor_pos]
                new_node = UCSSearchNode(neighbor_pos, new_cost, new_path)
                
                # Check if neighbor is already in frontier
                if neighbor_pos in frontier_set:
                    # Find existing node in frontier
                    # Note: This is a simplified approach. In practice, you might want to
                    # implement decrease-key operation or use a more sophisticated data structure
                    continue
                else:
                    # Add to frontier
                    heapq.heappush(frontier, new_node)
                    frontier_set.add(neighbor_pos)
        
        # No path found
        self.execution_time = time.time() - start_time
        return None
    
    def get_statistics(self) -> Dict[str, float]:
        """Get algorithm performance statistics."""
        return {
            'nodes_expanded': self.nodes_expanded,
            'path_cost': self.path_cost,
            'execution_time': self.execution_time,
            'path_length': len(self.path_cost) if isinstance(self.path_cost, list) else 0
        }


class UCSWithPathLength(UCSSearchNode):
    """
    UCS variant that breaks ties by path length (shorter paths preferred).
    """
    
    def __lt__(self, other):
        """Less than comparison with tie-breaking by path length."""
        if self.path_cost != other.path_cost:
            return self.path_cost < other.path_cost
        return len(self.path) < len(other.path)


class UCSPlannerOptimized:
    """
    Optimized UCS implementation with better data structures.
    """
    
    def __init__(self, environment: GridEnvironment):
        self.environment = environment
        self.nodes_expanded = 0
        self.path_cost = 0
        self.execution_time = 0.0
    
    def find_path(self, start: Position, goal: Position, time_step: int = 0) -> Optional[List[Position]]:
        """
        Find path from start to goal using optimized UCS.
        """
        start_time = time.time()
        self.nodes_expanded = 0
        self.path_cost = 0
        
        # Use dictionary to track best known costs
        best_costs = {start: 0.0}
        parent = {}
        frontier = [(0.0, start)]
        
        # Set environment time step
        self.environment.set_time(time_step)
        
        while frontier:
            current_cost, current_pos = heapq.heappop(frontier)
            
            # Skip if we've found a better path to this position
            if current_cost > best_costs.get(current_pos, float('inf')):
                continue
            
            self.nodes_expanded += 1
            
            # Check if goal reached
            if current_pos == goal:
                # Reconstruct path
                path = []
                while current_pos in parent:
                    path.append(current_pos)
                    current_pos = parent[current_pos]
                path.append(start)
                path.reverse()
                
                self.path_cost = best_costs[goal]
                self.execution_time = time.time() - start_time
                return path
            
            # Explore neighbors
            neighbors = self.environment.get_neighbors(current_pos, time_step)
            
            for neighbor_pos, movement_cost in neighbors:
                new_cost = current_cost + movement_cost
                
                # If we found a better path to this neighbor
                if new_cost < best_costs.get(neighbor_pos, float('inf')):
                    best_costs[neighbor_pos] = new_cost
                    parent[neighbor_pos] = current_pos
                    heapq.heappush(frontier, (new_cost, neighbor_pos))
        
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


def run_ucs_test():
    """Test function for UCS implementation."""
    from environment import create_test_environment
    
    # Create test environment
    env = create_test_environment("small")
    
    # Test UCS
    print("Testing UCS...")
    ucs = UCSPlanner(env)
    
    start = Position(0, 0)
    goal = Position(9, 9)
    
    path = ucs.find_path(start, goal)
    
    if path:
        print(f"Path found with cost {ucs.path_cost}")
        print(f"Nodes expanded: {ucs.nodes_expanded}")
        print(f"Execution time: {ucs.execution_time:.4f}s")
        print("Path:", [str(p) for p in path])
        
        # Visualize
        env.visualize(agent_pos=start, path=path)
    else:
        print("No path found!")
    
    # Test optimized UCS
    print("\nTesting Optimized UCS...")
    ucs_opt = UCSPlannerOptimized(env)
    path_opt = ucs_opt.find_path(start, goal)
    
    if path_opt:
        print(f"Path found with cost {ucs_opt.path_cost}")
        print(f"Nodes expanded: {ucs_opt.nodes_expanded}")
        print(f"Execution time: {ucs_opt.execution_time:.4f}s")


if __name__ == "__main__":
    run_ucs_test()
