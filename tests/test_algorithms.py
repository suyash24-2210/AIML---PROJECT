"""
Unit tests for pathfinding algorithms.
"""

import sys
import os
import unittest

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from environment import GridEnvironment, Position
from algorithms.bfs import BFSPlanner
from algorithms.ucs import UCSPlanner
from algorithms.astar import AStarPlanner
from algorithms.local_search import LocalSearchPlanner


class TestBFSPlanner(unittest.TestCase):
    """Test cases for BFS planner."""
    
    def setUp(self):
        """Set up test environment."""
        self.env = GridEnvironment(5, 5)
        self.planner = BFSPlanner(self.env)
    
    def test_simple_path(self):
        """Test finding a simple path."""
        start = Position(0, 0)
        goal = Position(4, 4)
        
        path = self.planner.find_path(start, goal)
        
        self.assertIsNotNone(path)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)
    
    def test_no_path(self):
        """Test when no path exists."""
        # Block the entire path
        for x in range(5):
            self.env.static_obstacles.add(Position(x, 2))
            self.env.grid[2, x] = -1
        
        start = Position(0, 0)
        goal = Position(0, 4)
        
        path = self.planner.find_path(start, goal)
        
        self.assertIsNone(path)


class TestUCSPlanner(unittest.TestCase):
    """Test cases for UCS planner."""
    
    def setUp(self):
        """Set up test environment."""
        self.env = GridEnvironment(5, 5)
        self.planner = UCSPlanner(self.env)
    
    def test_optimal_path(self):
        """Test finding optimal path."""
        # Add high-cost terrain
        self.env.grid[1, 0] = 5
        self.env.grid[2, 0] = 5
        
        start = Position(0, 0)
        goal = Position(0, 4)
        
        path = self.planner.find_path(start, goal)
        
        self.assertIsNotNone(path)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)


class TestAStarPlanner(unittest.TestCase):
    """Test cases for A* planner."""
    
    def setUp(self):
        """Set up test environment."""
        self.env = GridEnvironment(5, 5)
        self.planner = AStarPlanner(self.env)
    
    def test_heuristic_consistency(self):
        """Test heuristic function."""
        pos1 = Position(0, 0)
        pos2 = Position(3, 4)
        
        distance = self.planner.manhattan_distance(pos1, pos2)
        self.assertEqual(distance, 7)
    
    def test_pathfinding(self):
        """Test A* pathfinding."""
        start = Position(0, 0)
        goal = Position(4, 4)
        
        path = self.planner.find_path(start, goal)
        
        self.assertIsNotNone(path)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)


class TestLocalSearchPlanner(unittest.TestCase):
    """Test cases for local search planner."""
    
    def setUp(self):
        """Set up test environment."""
        self.env = GridEnvironment(5, 5)
        self.planner = LocalSearchPlanner(self.env, max_iterations=10)
    
    def test_path_calculation(self):
        """Test path cost calculation."""
        path = [Position(0, 0), Position(1, 0), Position(2, 0)]
        cost = self.planner.calculate_path_cost(path)
        
        self.assertEqual(cost, 2)  # Two moves, each cost 1
    
    def test_valid_path_checking(self):
        """Test path validation."""
        valid_path = [Position(0, 0), Position(1, 0), Position(1, 1)]
        invalid_path = [Position(0, 0), Position(2, 0)]  # Not adjacent
        
        self.assertTrue(self.planner.is_valid_path(valid_path, 0))
        self.assertFalse(self.planner.is_valid_path(invalid_path, 0))


if __name__ == '__main__':
    unittest.main()
