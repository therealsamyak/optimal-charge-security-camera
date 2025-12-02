#!/usr/bin/env python3
"""
Results Parser - Parses and displays CSV data from simulation results in human-readable format
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import statistics

def read_csv_full(filepath: str) -> List[Dict[str, Any]]:
    """Read entire CSV file"""
    data = []
    try:
        # Increase field size limit to handle large battery_levels data
        csv.field_size_limit(1000000)  # 1MB limit
        with open(filepath, 'r', newline='') as file:
            reader = csv.DictReader(file)
            data = list(reader)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []
    return data

def parse_json_field(field_str: str) -> Any:
    """Parse JSON field if possible, otherwise return as string"""
    if not field_str or field_str.strip() == '':
        return None
    try:
        return json.loads(field_str)
    except (json.JSONDecodeError, TypeError):
        return field_str

def analyze_battery_levels(battery_levels_str: str) -> Dict[str, float]:
    """Analyze battery levels data"""
    if not battery_levels_str:
        return {}
    
    try:
        levels = json.loads(battery_levels_str)
        if not levels:
            return {}
        
        battery_values = [entry['level'] for entry in levels]
        return {
            'initial': battery_values[0] if battery_values else 0,
            'final': battery_values[-1] if battery_values else 0,
            'min': min(battery_values) if battery_values else 0,
            'max': max(battery_values) if battery_values else 0,
            'avg': statistics.mean(battery_values) if battery_values else 0,
            'data_points': len(battery_values)
        }
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}

def print_section(title: str):
    """Print formatted section header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)

def print_subsection(title: str):
    """Print formatted subsection header"""
    print(f"\n--- {title} ---")

def display_metadata(metadata_path: str):
    """Display simulation metadata"""
    print_section("SIMULATION METADATA")
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"Total Simulations: {metadata.get('total_simulations', 'N/A')}")
        print(f"Total Variations: {metadata.get('total_variations', 'N/A')}")
        print(f"Locations: {', '.join(metadata.get('locations', []))}")
        print(f"Seasons: {', '.join(metadata.get('seasons', []))}")
        print(f"Controllers: {', '.join(metadata.get('controllers', []))}")
        print(f"Weeks Simulated: {', '.join(map(str, metadata.get('weeks_simulated', [])))}")
        print(f"Export Timestamp: {metadata.get('export_timestamp', 'N/A')}")
        print(f"Config File: {metadata.get('config_file', 'N/A')}")
        
        batch_config = metadata.get('batch_config', {})
        if batch_config:
            print_subsection("Batch Configuration Ranges")
            for key, range_data in batch_config.items():
                if isinstance(range_data, dict) and 'min' in range_data and 'max' in range_data:
                    print(f"  {key.replace('_', ' ').title()}: {range_data['min']} - {range_data['max']}")
                else:
                    print(f"  {key.replace('_', ' ').title()}: {range_data}")
    
    except Exception as e:
        print(f"Error reading metadata: {e}")

def display_aggregated_results(data: List[Dict[str, Any]]):
    """Display aggregated results summary"""
    print_section("AGGREGATED RESULTS SUMMARY")
    
    if not data:
        print("No data available")
        return
    
    # Basic statistics
    controllers = list(set(row.get('controller', 'Unknown') for row in data))
    locations = list(set(row.get('location', 'Unknown') for row in data))
    
    print(f"Total Records Analyzed: {len(data)}")
    print(f"Controllers: {', '.join(controllers)}")
    print(f"Locations: {', '.join(locations)}")
    
    # Success rates by controller
    print_subsection("Success Rates by Controller")
    controller_stats = {}
    for controller in controllers:
        controller_data = [row for row in data if row.get('controller') == controller]
        successful = sum(1 for row in controller_data if row.get('success') == 'True')
        total = len(controller_data)
        success_rate = (successful / total * 100) if total > 0 else 0
        controller_stats[controller] = {'success_rate': success_rate, 'total': total, 'successful': successful}
        print(f"  {controller}: {success_rate:.1f}% ({successful}/{total} successful)")
    
    # Performance metrics
    print_subsection("Performance Metrics")
    
    # Task completion rates
    completion_rates = []
    for row in data:
        if row.get('task_completion_rate'):
            try:
                completion_rates.append(float(row['task_completion_rate']))
            except ValueError:
                pass
    
    if completion_rates:
        print(f"  Task Completion Rate - Avg: {statistics.mean(completion_rates):.1f}%, "
              f"Min: {min(completion_rates):.1f}%, Max: {max(completion_rates):.1f}%")
    
    # Energy statistics
    clean_energy_percentages = []
    for row in data:
        if row.get('clean_energy_percentage'):
            try:
                clean_energy_percentages.append(float(row['clean_energy_percentage']))
            except ValueError:
                pass
    
    if clean_energy_percentages:
        print(f"  Clean Energy Usage - Avg: {statistics.mean(clean_energy_percentages):.1f}%, "
              f"Min: {min(clean_energy_percentages):.1f}%, Max: {max(clean_energy_percentages):.1f}%")
    
    # Battery analysis
    print_subsection("Battery Analysis")
    battery_stats = []
    for row in data:
        if row.get('battery_levels'):
            stats = analyze_battery_levels(row['battery_levels'])
            if stats:
                battery_stats.append(stats)
    
    if battery_stats:
        avg_initial = statistics.mean([s['initial'] for s in battery_stats])
        avg_final = statistics.mean([s['final'] for s in battery_stats])
        avg_depletion = avg_initial - avg_final
        print(f"  Average Initial Battery: {avg_initial:.1f}%")
        print(f"  Average Final Battery: {avg_final:.1f}%")
        print(f"  Average Battery Depletion: {avg_depletion:.1f}%")

def display_detailed_results(data: List[Dict[str, Any]], limit: int = 10):
    """Display detailed results for first few records"""
    print_section(f"DETAILED RESULTS (First {limit} Records)")
    
    if not data:
        print("No data available")
        return
    
    for i, row in enumerate(data[:limit]):
        print_subsection(f"Record {i+1}")
        
        # Basic info
        print(f"  Controller: {row.get('controller', 'N/A')}")
        print(f"  Location: {row.get('location', 'N/A')}")
        print(f"  Season: {row.get('season', 'N/A')}")
        print(f"  Week: {row.get('week', 'N/A')}")
        print(f"  Success: {row.get('success', 'N/A')}")
        
        # Performance metrics
        print(f"  Task Completion Rate: {row.get('task_completion_rate', 'N/A')}%")
        print(f"  Completed Tasks: {row.get('completed_tasks', 'N/A')}")
        print(f"  Total Tasks: {row.get('total_tasks', 'N/A')}")
        print(f"  Missed Deadlines: {row.get('missed_deadlines', 'N/A')}")
        
        # Energy metrics
        print(f"  Clean Energy Percentage: {row.get('clean_energy_percentage', 'N/A')}%")
        print(f"  Total Energy (Wh): {row.get('total_energy_wh', 'N/A')}")
        print(f"  Battery Capacity (Wh): {row.get('battery_capacity_wh', 'N/A')}")
        
        # Model performance
        print(f"  Large Model Miss Rate: {row.get('large_model_miss_rate', 'N/A')}%")
        print(f"  Small Model Miss Rate: {row.get('small_model_miss_rate', 'N/A')}%")
        
        # Requirements
        print(f"  Accuracy Requirement: {row.get('accuracy_requirement', 'N/A')}%")
        print(f"  Latency Requirement: {row.get('latency_requirement', 'N/A')}s")
        
        # Battery analysis
        if row.get('battery_levels'):
            battery_stats = analyze_battery_levels(row['battery_levels'])
            if battery_stats:
                print(f"  Battery Analysis:")
                print(f"    Initial: {battery_stats['initial']:.1f}%")
                print(f"    Final: {battery_stats['final']:.1f}%")
                print(f"    Depletion: {battery_stats['initial'] - battery_stats['final']:.1f}%")
                print(f"    Average: {battery_stats['avg']:.1f}%")

def display_key_insights(data: List[Dict[str, Any]]):
    """Display key insights and conclusions"""
    print_section("KEY INSIGHTS & CONCLUSIONS")
    
    if not data:
        print("No data available for analysis")
        return
    
    insights = []
    
    # Controller performance comparison
    controllers = list(set(row.get('controller', 'Unknown') for row in data))
    controller_performance = {}
    
    for controller in controllers:
        controller_data = [row for row in data if row.get('controller') == controller]
        success_rate = sum(1 for row in controller_data if row.get('success') == 'True') / len(controller_data) * 100
        avg_completion = statistics.mean([float(row.get('task_completion_rate', 0)) for row in controller_data if row.get('task_completion_rate')])
        controller_performance[controller] = {'success_rate': success_rate, 'avg_completion': avg_completion}
    
    best_controller = max(controller_performance.items(), key=lambda x: x[1]['success_rate'])
    insights.append(f"Best performing controller: {best_controller[0]} with {best_controller[1]['success_rate']:.1f}% success rate")
    
    # Location analysis
    locations = list(set(row.get('location', 'Unknown') for row in data))
    location_performance = {}
    
    for location in locations:
        location_data = [row for row in data if row.get('location') == location]
        avg_clean_energy = statistics.mean([float(row.get('clean_energy_percentage', 0)) for row in location_data if row.get('clean_energy_percentage')])
        location_performance[location] = avg_clean_energy
    
    best_location = max(location_performance.items(), key=lambda x: x[1])
    insights.append(f"Best location for clean energy: {best_location[0]} with {best_location[1]:.1f}% clean energy usage")
    
    # Battery efficiency
    battery_efficiencies = []
    for row in data:
        if row.get('battery_levels') and row.get('total_tasks'):
            battery_stats = analyze_battery_levels(row['battery_levels'])
            if battery_stats and battery_stats['initial'] > 0:
                depletion = battery_stats['initial'] - battery_stats['final']
                tasks_per_percent = float(row['total_tasks']) / depletion if depletion > 0 else float('inf')
                battery_efficiencies.append(tasks_per_percent)
    
    if battery_efficiencies:
        avg_efficiency = statistics.mean([e for e in battery_efficiencies if e != float('inf')])
        insights.append(f"Average battery efficiency: {avg_efficiency:.2f} tasks per percent battery depletion")
    
    # Display insights
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    # Recommendations
    print_subsection("Recommendations")
    print("• Consider using the best performing controller for critical applications")
    print("• Focus on locations with higher clean energy availability for sustainability")
    print("• Optimize battery management strategies based on efficiency patterns")
    print("• Further analysis needed on seasonal variations and their impact")

def main():
    """Main function to parse and display results"""
    results_dir = Path("results")
    
    if not results_dir.exists():
        print(f"Results directory '{results_dir}' not found!")
        return
    
    # File paths
    aggregated_file = results_dir / "batch-run-20251201_015744-aggregated-results.csv"
    detailed_file = results_dir / "batch-run-20251201_015744-detailed-results.csv"
    metadata_file = results_dir / "batch-run-20251201_015744-metadata.json"
    
    print("SECURITY CAMERA SIMULATION RESULTS ANALYZER")
    print("=" * 60)
    print(f"Analyzing results from: {results_dir}")
    
    # Display metadata
    if metadata_file.exists():
        display_metadata(str(metadata_file))
    else:
        print(f"Metadata file not found: {metadata_file}")
    
    # Read and display aggregated results
    if aggregated_file.exists():
        aggregated_data = read_csv_full(str(aggregated_file))
        display_aggregated_results(aggregated_data)
    else:
        print(f"Aggregated results file not found: {aggregated_file}")
    
    # Read and display detailed results
    if detailed_file.exists():
        detailed_data = read_csv_full(str(detailed_file))
        display_detailed_results(detailed_data, limit=20)
    else:
        print(f"Detailed results file not found: {detailed_file}")
    
    # Display key insights
    aggregated_data = locals().get('aggregated_data')
    if aggregated_data:
        display_key_insights(aggregated_data)
    
    print_section("ANALYSIS COMPLETE")
    print("Note: This analysis processes the complete datasets")

if __name__ == "__main__":
    main()