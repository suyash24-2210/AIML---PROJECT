"""
Unit tests for the environment module.
"""

import sys
import os
import unittest

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from environment import GridEnvironment, Position, DynamicObstacle


class TestGridEnvironment(unittest.TestCase):
    """Test cases for GridEnvironment class."""
    
    def setUp(self):
        """Set up test environment."""
        self.env = GridEnvironment(10, 10)
    
    def test_initialization(self):
        """Test environment initialization."""
        self.assertEqual(self.env.width, 10)
        self.assertEqual(self.env.height, 10)
        self.assertEqual(self.env.time_step, 0)
    
    def test_position_creation(self):
        """Test Position class."""
        pos = Position(5, 5)
        self.assertEqual(pos.x, 5)
        self.assertEqual(pos.y, 5)
        self.assertEqual(str(pos), "(5, 5)")
    
    def test_valid_position(self):
        """Test position validation."""
        self.assertTrue(self.env.is_valid_position(Position(0, 0)))
        self.assertTrue(self.env.is_valid_position(Position(9, 9)))
        self.assertFalse(self.env.is_valid_position(Position(-1, 0)))
        self.assertFalse(self.env.is_valid_position(Position(0, -1)))
        self.assertFalse(self.env.is_valid_position(Position(10, 0)))
        self.assertFalse(self.env.is_valid_position(Position(0, 10)))
    
    def test_manhattan_distance(self):
        """Test Manhattan distance calculation."""
        pos1 = Position(0, 0)
        pos2 = Position(3, 4)
        distance = self.env.manhattan_distance(pos1, pos2)
        self.assertEqual(distance, 7)
    
    def test_get_neighbors(self):
        """Test neighbor finding."""
        pos = Position(5, 5)
        neighbors = self.env.get_neighbors(pos)
        
        expected_neighbors = [
            Position(6, 5),  # right
            Position(5, 6),  # down
            Position(4, 5),  # left
            Position(5, 4)   # up
        ]
        
        neighbor_positions = [pos for pos, cost in neighbors]
        for expected in expected_neighbors:
            self.assertIn(expected, neighbor_positions)
    
    def test_obstacle_placement(self):
        """Test obstacle placement and checking."""
        obstacle_pos = Position(5, 5)
        self.env.static_obstacles.add(obstacle_pos)
        self.env.grid[5, 5] = -1
        
        self.assertTrue(self.env.is_obstacle(obstacle_pos))
        self.assertFalse(self.env.is_obstacle(Position(4, 4)))


class TestDynamicObstacle(unittest.TestCase):
    """Test cases for DynamicObstacle class."""
    
    def setUp(self):
        """Set up test dynamic obstacle."""
        schedule = [
            (0, Position(0, 0)),
            (1, Position(1, 0)),
            (2, Position(1, 1)),
            (3, Position(0, 1))
        ]
        self.obstacle = DynamicObstacle("A", Position(0, 0), schedule)
    
    def test_initialization(self):
        """Test dynamic obstacle initialization."""
        self.assertEqual(self.obstacle.id, "A")
        self.assertEqual(self.obstacle.current_pos, Position(0, 0))
        self.assertEqual(self.obstacle.current_time, 0)
    
    def test_get_position_at_time(self):
        """Test getting position at specific time."""
        self.assertEqual(self.obstacle.get_position_at_time(0), Position(0, 0))
        self.assertEqual(self.obstacle.get_position_at_time(1), Position(1, 0))
        self.assertEqual(self.obstacle.get_position_at_time(2), Position(1, 1))
        self.assertEqual(self.obstacle.get_position_at_time(3), Position(0, 1))
    
    def test_advance_time(self):
        """Test time advancement."""
        self.obstacle.advance_time()
        self.assertEqual(self.obstacle.current_time, 1)
        self.assertEqual(self.obstacle.current_pos, Position(1, 0))


if __name__ == '__main__':
    unittest.main()
