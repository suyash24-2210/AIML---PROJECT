# Autonomous Delivery Agent

An autonomous delivery agent that navigates a 2D grid city to deliver packages using various pathfinding algorithms including BFS, Uniform-cost search, A\*, and local search replanning strategies.

## Features

- **Multiple Pathfinding Algorithms**: BFS, Uniform-cost search, A\* with admissible heuristics
- **Dynamic Replanning**: Hill-climbing with random restarts for handling moving obstacles
- **Environment Modeling**: Static obstacles, varying terrain costs, dynamic moving obstacles
- **Experimental Comparison**: Performance analysis across different algorithms and map sizes
- **Visualization**: Real-time visualization of agent navigation and pathfinding

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd autonomous-delivery-agent
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Run the Demo

```bash
python demo.py
```

### Basic Usage

```bash
python main.py --algorithm astar --map maps/small.txt --start 0,0 --goal 9,9
```

### Available Algorithms

- `bfs`: Breadth-First Search (shortest path in steps)
- `bfs_cost`: BFS with cost tracking
- `ucs`: Uniform-Cost Search (optimal cost path)
- `ucs_opt`: Optimized UCS implementation
- `astar`: A\* with Manhattan distance heuristic (recommended)
- `astar_weighted`: Weighted A\* for faster suboptimal paths
- `local`: Hill-climbing with random restarts (dynamic environments)
- `sa`: Simulated annealing

### Map Format

Maps are text files where:

- `.` = empty cell (cost 1)
- `#` = obstacle (impassable)
- `2-9` = terrain with movement cost 2-9
- `A-Z` = dynamic obstacles (moving vehicles)

Example:

```
.2.#
..3.
#..A
```

### Command Line Options

```bash
python main.py [OPTIONS]

Required for single algorithm:
  --algorithm ALG     Algorithm to use
  --map MAP_FILE      Path to map file
  --start X,Y         Start coordinates
  --goal X,Y          Goal coordinates

Optional:
  --visualize         Enable visualization
  --time-step T       Time step for dynamic obstacles
  --weight W          Weight for weighted A\* (default: 1.5)
  --output FILE       Save results to file

Comparison and Analysis:
  --compare           Compare all algorithms
  --benchmark         Run comprehensive benchmark tests
  --dynamic-demo      Run dynamic replanning demonstration
```

### Usage Examples

```bash
# Find path with A* and visualize
python main.py --algorithm astar --map maps/small.txt --start 0,0 --goal 9,9 --visualize

# Compare all algorithms
python main.py --compare --map maps/medium.txt --start 0,0 --goal 19,19

# Run benchmark tests
python main.py --benchmark

# Dynamic replanning demo
python main.py --dynamic-demo --algorithm local

# Run experiments and save results
python main.py --benchmark --output results.json
```

## Project Structure

```
autonomous-delivery-agent/
├── main.py                 # Main entry point
├── src/
│   ├── environment.py      # Environment model
│   ├── algorithms/         # Pathfinding algorithms
│   │   ├── bfs.py
│   │   ├── ucs.py
│   │   ├── astar.py
│   │   └── local_search.py
│   └── visualization.py    # Visualization tools
├── maps/                   # Test maps
│   ├── small.txt
│   ├── medium.txt
│   ├── large.txt
│   └── dynamic.txt
├── tests/                  # Unit tests
├── results/                # Experimental results
└── report/                 # Analysis report
```

## Testing

Run unit tests:

```bash
python -m pytest tests/
```

Run benchmark comparisons:

```bash
python main.py --benchmark
```

## Results

Experimental results and analysis are available in the `results/` directory and the accompanying report.



