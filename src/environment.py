"""
Environment model for autonomous delivery agent.
Handles grid-based world with static obstacles, terrain costs, and dynamic moving obstacles.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum
import random


class CellType(Enum):
    """Types of cells in the grid."""
    EMPTY = "."
    OBSTACLE = "#"
    TERRAIN = "2-9"
    DYNAMIC_OBSTACLE = "A-Z"


@dataclass
class Position:
    """Represents a position in the grid."""
    x: int
    y: int
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        return isinstance(other, Position) and self.x == other.x and self.y == other.y
    
    def __str__(self):
        return f"({self.x}, {self.y})"


@dataclass
class DynamicObstacle:
    """Represents a moving obstacle with a deterministic schedule."""
    id: str
    current_pos: Position
    schedule: List[Tuple[int, Position]]  # (time_step, position)
    current_time: int = 0
    
    def get_position_at_time(self, time_step: int) -> Position:
        """Get obstacle position at a specific time step."""
        for i, (t, pos) in enumerate(self.schedule):
            if t >= time_step:
                return pos
        # If time exceeds schedule, stay at last position
        return self.schedule[-1][1]
    
    def advance_time(self):
        """Advance obstacle to next time step."""
        self.current_time += 1
        self.current_pos = self.get_position_at_time(self.current_time)


class GridEnvironment:
    """
    Grid-based environment for autonomous delivery agent.
    
    Features:
    - Static obstacles (impassable)
    - Terrain with varying movement costs (1-9)
    - Dynamic obstacles (moving vehicles with known schedules)
    - 4-connected movement (up, down, left, right)
    """
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = np.full((height, width), 1, dtype=int)  # Default cost 1
        self.static_obstacles: Set[Position] = set()
        self.dynamic_obstacles: Dict[str, DynamicObstacle] = {}
        self.time_step = 0
        
        # Movement directions (4-connected)
        self.directions = [
            (0, 1),   # right
            (1, 0),   # down
            (0, -1),  # left
            (-1, 0)   # up
        ]
    
    def load_from_file(self, filename: str):
        """Load environment from a text file."""
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        self.height = len(lines)
        self.width = len(lines[0].strip())
        self.grid = np.full((self.height, self.width), 1, dtype=int)
        self.static_obstacles.clear()
        self.dynamic_obstacles.clear()
        
        for y, line in enumerate(lines):
            line = line.strip()
            for x, char in enumerate(line):
                pos = Position(x, y)
                
                if char == '#':
                    # Static obstacle
                    self.static_obstacles.add(pos)
                    self.grid[y, x] = -1  # -1 indicates impassable
                elif char.isdigit() and char != '0':
                    # Terrain with cost
                    self.grid[y, x] = int(char)
                elif char.isalpha() and char.isupper():
                    # Dynamic obstacle (moving vehicle)
                    obstacle_id = char
                    if obstacle_id not in self.dynamic_obstacles:
                        # Initialize with a simple schedule - stay in place
                        schedule = [(0, pos)]
                        self.dynamic_obstacles[obstacle_id] = DynamicObstacle(
                            id=obstacle_id,
                            current_pos=pos,
                            schedule=schedule
                        )
                elif char == '.':
                    # Empty cell with cost 1
                    self.grid[y, x] = 1
    
    def save_to_file(self, filename: str):
        """Save environment to a text file."""
        with open(filename, 'w') as f:
            for y in range(self.height):
                for x in range(self.width):
                    pos = Position(x, y)
                    
                    if pos in self.static_obstacles:
                        f.write('#')
                    elif pos in [obs.current_pos for obs in self.dynamic_obstacles.values()]:
                        # Find which dynamic obstacle is at this position
                        for obs in self.dynamic_obstacles.values():
                            if obs.current_pos == pos:
                                f.write(obs.id)
                                break
                    else:
                        cost = self.grid[y, x]
                        if cost == 1:
                            f.write('.')
                        else:
                            f.write(str(cost))
                f.write('\n')
    
    def is_valid_position(self, pos: Position) -> bool:
        """Check if position is within grid bounds."""
        return 0 <= pos.x < self.width and 0 <= pos.y < self.height
    
    def is_obstacle(self, pos: Position, time_step: Optional[int] = None) -> bool:
        """Check if position is blocked by obstacle at given time step."""
        if not self.is_valid_position(pos):
            return True
        
        # Check static obstacles
        if pos in self.static_obstacles:
            return True
        
        # Check dynamic obstacles
        if time_step is None:
            time_step = self.time_step
        
        for obstacle in self.dynamic_obstacles.values():
            if obstacle.get_position_at_time(time_step) == pos:
                return True
        
        return False
    
    def get_movement_cost(self, pos: Position) -> int:
        """Get movement cost for a position."""
        if not self.is_valid_position(pos) or self.is_obstacle(pos):
            return -1  # Impassable
        return self.grid[pos.y, pos.x]
    
    def get_neighbors(self, pos: Position, time_step: Optional[int] = None) -> List[Tuple[Position, int]]:
        """
        Get valid neighboring positions with their movement costs.
        Returns list of (position, cost) tuples.
        """
        neighbors = []
        
        for dx, dy in self.directions:
            neighbor_pos = Position(pos.x + dx, pos.y + dy)
            
            if self.is_valid_position(neighbor_pos) and not self.is_obstacle(neighbor_pos, time_step):
                cost = self.get_movement_cost(neighbor_pos)
                if cost > 0:  # Valid passable cell
                    neighbors.append((neighbor_pos, cost))
        
        return neighbors
    
    def advance_time(self):
        """Advance environment to next time step."""
        self.time_step += 1
        for obstacle in self.dynamic_obstacles.values():
            obstacle.advance_time()
    
    def set_time(self, time_step: int):
        """Set environment to specific time step."""
        self.time_step = time_step
        for obstacle in self.dynamic_obstacles.values():
            obstacle.current_time = time_step
            obstacle.current_pos = obstacle.get_position_at_time(time_step)
    
    def add_dynamic_obstacle_schedule(self, obstacle_id: str, schedule: List[Tuple[int, Position]]):
        """Add or update schedule for a dynamic obstacle."""
        if obstacle_id in self.dynamic_obstacles:
            self.dynamic_obstacles[obstacle_id].schedule = schedule
        else:
            # Create new dynamic obstacle
            start_pos = schedule[0][1] if schedule else Position(0, 0)
            self.dynamic_obstacles[obstacle_id] = DynamicObstacle(
                id=obstacle_id,
                current_pos=start_pos,
                schedule=schedule
            )
    
    def manhattan_distance(self, pos1: Position, pos2: Position) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1.x - pos2.x) + abs(pos1.y - pos2.y)
    
    def get_all_dynamic_obstacle_positions(self, time_step: Optional[int] = None) -> Set[Position]:
        """Get all positions occupied by dynamic obstacles at given time step."""
        if time_step is None:
            time_step = self.time_step
        
        positions = set()
        for obstacle in self.dynamic_obstacles.values():
            positions.add(obstacle.get_position_at_time(time_step))
        return positions
    
    def visualize(self, agent_pos: Optional[Position] = None, path: Optional[List[Position]] = None):
        """Simple text-based visualization of the environment."""
        print(f"Environment at time step {self.time_step}")
        print("Legend: . = empty (cost 1), # = obstacle, 2-9 = terrain cost, A-Z = dynamic obstacle, @ = agent, * = path")
        print()
        
        for y in range(self.height):
            for x in range(self.width):
                pos = Position(x, y)
                
                if agent_pos and pos == agent_pos:
                    print("@", end="")
                elif path and pos in path:
                    print("*", end="")
                elif pos in self.static_obstacles:
                    print("#", end="")
                elif pos in self.get_all_dynamic_obstacle_positions():
                    # Find which dynamic obstacle
                    for obstacle in self.dynamic_obstacles.values():
                        if obstacle.get_position_at_time(self.time_step) == pos:
                            print(obstacle.id, end="")
                            break
                else:
                    cost = self.grid[y, x]
                    if cost == 1:
                        print(".", end="")
                    else:
                        print(str(cost), end="")
            print()
        print()


def create_test_environment(size: str = "small") -> GridEnvironment:
    """Create a test environment for experimentation."""
    if size == "small":
        env = GridEnvironment(10, 10)
        # Add some static obstacles
        for pos in [Position(3, 3), Position(4, 3), Position(5, 3), Position(6, 6)]:
            env.static_obstacles.add(pos)
            env.grid[pos.y, pos.x] = -1
        
        # Add terrain costs
        env.grid[1, 1] = 3
        env.grid[2, 2] = 5
        env.grid[7, 7] = 2
        
        # Add dynamic obstacle
        schedule = [
            (0, Position(2, 4)),
            (1, Position(3, 4)),
            (2, Position(4, 4)),
            (3, Position(5, 4)),
            (4, Position(5, 5)),
            (5, Position(5, 6)),
            (6, Position(4, 6)),
            (7, Position(3, 6)),
            (8, Position(2, 6)),
            (9, Position(2, 4))  # Loop back
        ]
        env.add_dynamic_obstacle_schedule("A", schedule)
        
    elif size == "medium":
        env = GridEnvironment(20, 20)
        # More complex environment
        for x in range(5, 15):
            env.static_obstacles.add(Position(x, 10))
            env.grid[10, x] = -1
        
        for y in range(5, 15):
            env.static_obstacles.add(Position(10, y))
            env.grid[y, 10] = -1
        
        # Add terrain costs
        for x in range(2, 8):
            for y in range(2, 8):
                env.grid[y, x] = 2
        
        for x in range(12, 18):
            for y in range(12, 18):
                env.grid[y, x] = 3
        
    elif size == "large":
        env = GridEnvironment(50, 50)
        # Large environment with complex obstacles
        for x in range(10, 40):
            for y in range(10, 15):
                env.static_obstacles.add(Position(x, y))
                env.grid[y, x] = -1
        
        for x in range(10, 15):
            for y in range(20, 40):
                env.static_obstacles.add(Position(x, y))
                env.grid[y, x] = -1
        
        # Add varied terrain
        for x in range(5, 45):
            for y in range(5, 45):
                if (x + y) % 3 == 0:
                    env.grid[y, x] = random.randint(2, 5)
    
    return env


if __name__ == "__main__":
    # Test the environment
    env = create_test_environment("small")
    env.visualize()
    
    # Test dynamic obstacle movement
    print("Testing dynamic obstacle movement:")
    for t in range(5):
        env.set_time(t)
        print(f"Time step {t}:")
        env.visualize()
