"""
Experimental framework for comparing pathfinding algorithms.
Provides systematic testing and analysis capabilities.
"""

import sys
import os
import json
import time
import random
import statistics
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import GridEnvironment, Position
from algorithms.bfs import BFSPlanner, BFSWithCosts
from algorithms.ucs import UCSPlanner, UCSPlannerOptimized
from algorithms.astar import AStarPlanner, AStarWeighted
from algorithms.local_search import LocalSearchPlanner, SimulatedAnnealingPlanner


class ExperimentRunner:
    """Systematic experiment runner for pathfinding algorithms."""
    
    def __init__(self):
        self.algorithms = {
            'bfs': BFSPlanner,
            'bfs_cost': BFSWithCosts,
            'ucs': UCSPlanner,
            'ucs_opt': UCSPlannerOptimized,
            'astar': AStarPlanner,
            'astar_weighted': AStarWeighted,
            'local': LocalSearchPlanner,
            'sa': SimulatedAnnealingPlanner
        }
        
        self.results = {}
    
    def create_test_scenarios(self) -> List[Dict[str, Any]]:
        """Create comprehensive test scenarios."""
        scenarios = []
        
        # Scenario 1: Small grid with simple obstacles
        scenarios.append({
            'name': 'small_simple',
            'description': 'Small grid (10x10) with simple obstacles',
            'map_file': 'maps/small.txt',
            'start': Position(0, 0),
            'goal': Position(9, 9),
            'time_steps': [0],
            'expected_difficulty': 'easy'
        })
        
        # Scenario 2: Medium grid with complex obstacles
        scenarios.append({
            'name': 'medium_complex',
            'description': 'Medium grid (20x20) with complex obstacle patterns',
            'map_file': 'maps/medium.txt',
            'start': Position(0, 0),
            'goal': Position(19, 19),
            'time_steps': [0],
            'expected_difficulty': 'medium'
        })
        
        # Scenario 3: Large grid with many obstacles
        scenarios.append({
            'name': 'large_many_obstacles',
            'description': 'Large grid (50x50) with many obstacles',
            'map_file': 'maps/large.txt',
            'start': Position(0, 0),
            'goal': Position(49, 49),
            'time_steps': [0],
            'expected_difficulty': 'hard'
        })
        
        # Scenario 4: Dynamic environment with moving obstacles
        scenarios.append({
            'name': 'dynamic_obstacles',
            'description': 'Dynamic environment with moving obstacles',
            'map_file': 'maps/dynamic.txt',
            'start': Position(0, 0),
            'goal': Position(19, 19),
            'time_steps': [0, 5, 10, 15],
            'expected_difficulty': 'dynamic'
        })
        
        # Scenario 5: Multiple start-goal pairs on same map
        for i in range(3):
            scenarios.append({
                'name': f'medium_multiple_{i}',
                'description': f'Medium grid with different start-goal pair {i}',
                'map_file': 'maps/medium.txt',
                'start': Position(i * 5, i * 5),
                'goal': Position(19 - i * 5, 19 - i * 5),
                'time_steps': [0],
                'expected_difficulty': 'medium'
            })
        
        return scenarios
    
    def run_single_experiment(self, environment: GridEnvironment, algorithm_name: str,
                            start: Position, goal: Position, time_step: int = 0,
                            **kwargs) -> Dict[str, Any]:
        """Run a single experiment with one algorithm."""
        if algorithm_name not in self.algorithms:
            return {'error': f'Unknown algorithm: {algorithm_name}'}
        
        try:
            # Create planner instance
            if algorithm_name == 'astar_weighted':
                planner = self.algorithms[algorithm_name](environment, weight=kwargs.get('weight', 1.5))
            elif algorithm_name in ['local', 'sa']:
                planner = self.algorithms[algorithm_name](environment, **kwargs)
            else:
                planner = self.algorithms[algorithm_name](environment)
            
            # Run algorithm
            start_time = time.time()
            path = planner.find_path(start, goal, time_step)
            end_time = time.time()
            
            # Get statistics
            stats = planner.get_statistics()
            stats['algorithm'] = algorithm_name
            stats['path_found'] = path is not None
            stats['path_length'] = len(path) if path else 0
            stats['total_time'] = end_time - start_time
            
            return stats
            
        except Exception as e:
            return {'error': str(e), 'algorithm': algorithm_name}
    
    def run_scenario(self, scenario: Dict[str, Any], algorithms: List[str] = None) -> Dict[str, Any]:
        """Run a complete scenario with all algorithms."""
        if algorithms is None:
            algorithms = ['bfs', 'ucs', 'astar', 'local']
        
        # Load environment
        environment = GridEnvironment(0, 0)
        environment.load_from_file(scenario['map_file'])
        
        # Set up dynamic obstacles if needed
        if 'dynamic' in scenario['map_file']:
            self._setup_dynamic_obstacles(environment)
        
        scenario_results = {
            'scenario': scenario,
            'algorithms': {}
        }
        
        print(f"Running scenario: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"Start: {scenario['start']}, Goal: {scenario['goal']}")
        
        for algorithm in algorithms:
            print(f"  Testing {algorithm}...")
            algorithm_results = {}
            
            for time_step in scenario['time_steps']:
                result = self.run_single_experiment(
                    environment, algorithm, scenario['start'], scenario['goal'], time_step
                )
                algorithm_results[f'time_{time_step}'] = result
            
            scenario_results['algorithms'][algorithm] = algorithm_results
        
        return scenario_results
    
    def _setup_dynamic_obstacles(self, environment: GridEnvironment):
        """Set up dynamic obstacles for testing."""
        schedules = {
            'A': [(i, Position(5 + i % 10, 10)) for i in range(20)],
            'B': [(i, Position(15, 5 + i % 10)) for i in range(20)],
            'C': [(i, Position(10 - i % 10, 15)) for i in range(20)],
        }
        
        for vehicle_id, schedule in schedules.items():
            environment.add_dynamic_obstacle_schedule(vehicle_id, schedule)
    
    def run_comprehensive_experiments(self, algorithms: List[str] = None, 
                                    scenarios: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run comprehensive experiments across all scenarios."""
        if algorithms is None:
            algorithms = ['bfs', 'ucs', 'astar', 'local']
        
        if scenarios is None:
            scenarios = self.create_test_scenarios()
        
        print("Starting comprehensive experiments...")
        print(f"Algorithms: {algorithms}")
        print(f"Scenarios: {len(scenarios)}")
        print("=" * 60)
        
        all_results = {
            'experiment_info': {
                'algorithms': algorithms,
                'num_scenarios': len(scenarios),
                'timestamp': time.time()
            },
            'scenarios': {}
        }
        
        for scenario in scenarios:
            scenario_result = self.run_scenario(scenario, algorithms)
            all_results['scenarios'][scenario['name']] = scenario_result
            print()
        
        self.results = all_results
        return all_results
    
    def analyze_results(self, results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze experimental results and generate statistics."""
        if results is None:
            results = self.results
        
        if not results:
            return {'error': 'No results to analyze'}
        
        analysis = {
            'summary': {},
            'algorithm_comparison': {},
            'scenario_analysis': {}
        }
        
        # Collect all results by algorithm
        algorithm_stats = {}
        for scenario_name, scenario_data in results['scenarios'].items():
            for algorithm, algorithm_data in scenario_data['algorithms'].items():
                if algorithm not in algorithm_stats:
                    algorithm_stats[algorithm] = {
                        'successful_runs': 0,
                        'total_runs': 0,
                        'path_costs': [],
                        'nodes_expanded': [],
                        'execution_times': [],
                        'path_lengths': []
                    }
                
                for time_key, result in algorithm_data.items():
                    algorithm_stats[algorithm]['total_runs'] += 1
                    
                    if result.get('path_found', False):
                        algorithm_stats[algorithm]['successful_runs'] += 1
                        algorithm_stats[algorithm]['path_costs'].append(result.get('path_cost', 0))
                        algorithm_stats[algorithm]['nodes_expanded'].append(result.get('nodes_expanded', 0))
                        algorithm_stats[algorithm]['execution_times'].append(result.get('execution_time', 0))
                        algorithm_stats[algorithm]['path_lengths'].append(result.get('path_length', 0))
        
        # Calculate statistics for each algorithm
        for algorithm, stats in algorithm_stats.items():
            if stats['total_runs'] > 0:
                success_rate = stats['successful_runs'] / stats['total_runs']
                
                analysis['algorithm_comparison'][algorithm] = {
                    'success_rate': success_rate,
                    'avg_path_cost': statistics.mean(stats['path_costs']) if stats['path_costs'] else 0,
                    'avg_nodes_expanded': statistics.mean(stats['nodes_expanded']) if stats['nodes_expanded'] else 0,
                    'avg_execution_time': statistics.mean(stats['execution_times']) if stats['execution_times'] else 0,
                    'avg_path_length': statistics.mean(stats['path_lengths']) if stats['path_lengths'] else 0,
                    'std_path_cost': statistics.stdev(stats['path_costs']) if len(stats['path_costs']) > 1 else 0,
                    'std_nodes_expanded': statistics.stdev(stats['nodes_expanded']) if len(stats['nodes_expanded']) > 1 else 0,
                    'std_execution_time': statistics.stdev(stats['execution_times']) if len(stats['execution_times']) > 1 else 0
                }
        
        return analysis
    
    def generate_report(self, results: Dict[str, Any] = None, 
                       analysis: Dict[str, Any] = None) -> str:
        """Generate a text report of the experimental results."""
        if results is None:
            results = self.results
        
        if analysis is None:
            analysis = self.analyze_results(results)
        
        report = []
        report.append("AUTONOMOUS DELIVERY AGENT - EXPERIMENTAL RESULTS")
        report.append("=" * 60)
        report.append("")
        
        # Experiment summary
        if 'experiment_info' in results:
            info = results['experiment_info']
            report.append("EXPERIMENT SUMMARY:")
            report.append(f"  Algorithms tested: {', '.join(info['algorithms'])}")
            report.append(f"  Number of scenarios: {info['num_scenarios']}")
            report.append(f"  Timestamp: {time.ctime(info['timestamp'])}")
            report.append("")
        
        # Algorithm comparison
        if 'algorithm_comparison' in analysis:
            report.append("ALGORITHM COMPARISON:")
            report.append("-" * 40)
            
            for algorithm, stats in analysis['algorithm_comparison'].items():
                report.append(f"\n{algorithm.upper()}:")
                report.append(f"  Success rate: {stats['success_rate']:.2%}")
                report.append(f"  Average path cost: {stats['avg_path_cost']:.2f} ± {stats['std_path_cost']:.2f}")
                report.append(f"  Average nodes expanded: {stats['avg_nodes_expanded']:.0f} ± {stats['std_nodes_expanded']:.0f}")
                report.append(f"  Average execution time: {stats['avg_execution_time']:.4f}s ± {stats['std_execution_time']:.4f}s")
                report.append(f"  Average path length: {stats['avg_path_length']:.0f}")
            
            report.append("")
        
        # Scenario analysis
        report.append("SCENARIO RESULTS:")
        report.append("-" * 40)
        
        for scenario_name, scenario_data in results['scenarios'].items():
            scenario = scenario_data['scenario']
            report.append(f"\n{scenario_name.upper()}:")
            report.append(f"  Description: {scenario['description']}")
            report.append(f"  Difficulty: {scenario.get('expected_difficulty', 'unknown')}")
            
            for algorithm, algorithm_data in scenario_data['algorithms'].items():
                # Get the first time step result
                first_result = list(algorithm_data.values())[0]
                if first_result.get('path_found', False):
                    report.append(f"  {algorithm}: Cost={first_result.get('path_cost', 0):.2f}, "
                                f"Nodes={first_result.get('nodes_expanded', 0)}, "
                                f"Time={first_result.get('execution_time', 0):.4f}s")
                else:
                    report.append(f"  {algorithm}: No path found")
        
        return "\n".join(report)
    
    def save_results(self, filename: str, results: Dict[str, Any] = None):
        """Save results to JSON file."""
        if results is None:
            results = self.results
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to {filename}")
    
    def create_performance_plots(self, analysis: Dict[str, Any] = None, 
                               save_plots: bool = True):
        """Create performance comparison plots."""
        if analysis is None:
            analysis = self.analyze_results()
        
        if 'algorithm_comparison' not in analysis:
            print("No algorithm comparison data available")
            return
        
        algorithms = list(analysis['algorithm_comparison'].keys())
        
        # Prepare data
        success_rates = [analysis['algorithm_comparison'][alg]['success_rate'] for alg in algorithms]
        avg_costs = [analysis['algorithm_comparison'][alg]['avg_path_cost'] for alg in algorithms]
        avg_nodes = [analysis['algorithm_comparison'][alg]['avg_nodes_expanded'] for alg in algorithms]
        avg_times = [analysis['algorithm_comparison'][alg]['avg_execution_time'] for alg in algorithms]
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Success rates
        ax1.bar(algorithms, success_rates, color='green', alpha=0.7)
        ax1.set_title('Success Rate Comparison')
        ax1.set_ylabel('Success Rate')
        ax1.set_ylim(0, 1)
        
        # Path costs
        ax2.bar(algorithms, avg_costs, color='blue', alpha=0.7)
        ax2.set_title('Average Path Cost Comparison')
        ax2.set_ylabel('Path Cost')
        
        # Nodes expanded
        ax3.bar(algorithms, avg_nodes, color='orange', alpha=0.7)
        ax3.set_title('Average Nodes Expanded Comparison')
        ax3.set_ylabel('Nodes Expanded')
        
        # Execution times
        ax4.bar(algorithms, avg_times, color='red', alpha=0.7)
        ax4.set_title('Average Execution Time Comparison')
        ax4.set_ylabel('Execution Time (s)')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('results/performance_comparison.png', dpi=300, bbox_inches='tight')
            print("Performance plots saved to results/performance_comparison.png")
        
        plt.show()


def main():
    """Run comprehensive experiments."""
    runner = ExperimentRunner()
    
    # Run experiments
    print("Running comprehensive experiments...")
    results = runner.run_comprehensive_experiments()
    
    # Analyze results
    print("\nAnalyzing results...")
    analysis = runner.analyze_results(results)
    
    # Generate and save report
    report = runner.generate_report(results, analysis)
    print("\n" + report)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    runner.save_results('results/experimental_results.json', results)
    
    with open('results/analysis_report.txt', 'w') as f:
        f.write(report)
    
    # Create plots
    runner.create_performance_plots(analysis)
    
    print("\nExperiments completed!")


if __name__ == "__main__":
    main()
