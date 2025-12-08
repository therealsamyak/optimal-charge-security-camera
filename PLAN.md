# Complete Plan: Enhanced Oracle Controller & Training Data Generation

## Critical Analysis of Current Implementation

**Current Oracle Controller Issues:**
- Gets empty `future_energy_data: {}` and `future_tasks: 0` 
- Solves single-timestep MILP instead of full-horizon optimization
- No DP matrix for pre-computed optimal decisions

**Current Simulation Engine Issues:**
- Calls `controller.select_model()` at each timestep (line 255)
- Oracle solves MILP from scratch each time instead of using pre-computed DP matrix
- No integration with full-day future knowledge

## Enhanced Solution Architecture

### Phase 1: Oracle Controller with DP Matrix System

**Enhanced OracleController Design:**
```python
class OracleController(Controller):
    def __init__(self, clean_energy_series: List[float], task_requirements: List[Dict], config):
        self.config = config
        self.clean_energy_series = clean_energy_series  # Full day data
        self.task_requirements = task_requirements      # Full day requirements
        
        # Pre-compute optimal schedule for entire day
        self.optimal_schedule = self.solve_full_horizon_milp()
        
        # Create DP matrix for fast lookup
        self.dp_matrix = self.create_dp_matrix()
    
    def solve_full_horizon_milp(self):
        """Solve MILP for entire day (288 timesteps at 5-min intervals)"""
        # Decision variables: model_t[m], charge_t for all t
        # Objective: maximize Σ(clean_energy_t * charge_t)
        # Constraints: battery dynamics, task requirements, bounds
        return optimal_schedule  # List of (model_t, charge_t) for each timestep
    
    def create_dp_matrix(self):
        """Create lookup table: timestep → (optimal_model, should_charge)"""
        dp_matrix = {}
        for t, (model, charge) in enumerate(self.optimal_schedule):
            dp_matrix[t] = {"model": model, "charge": charge}
        return dp_matrix
    
    def select_model(self, battery_level, clean_energy_percentage, 
                    user_accuracy_requirement, user_latency_requirement, 
                    available_models):
        """Fast DP matrix lookup instead of solving MILP"""
        current_timestep = self.get_current_timestep()
        optimal_decision = self.dp_matrix[current_timestep]
        
        return ModelChoice(
            model_name=optimal_decision["model"],
            should_charge=optimal_decision["charge"]
        )
    
    def should_charge(self):
        """Return pre-computed charging decision"""
        current_timestep = self.get_current_timestep()
        return self.dp_matrix[current_timestep]["charge"]
```

### Phase 2: Simulation Engine Integration

**Enhanced Simulation Engine Flow:**
```python
class SimulationEngine:
    def __init__(self, config, controller, location, season, week, power_profiles, energy_data):
        # ... existing initialization ...
        
        # For oracle controller: prepare full-day data
        if isinstance(controller, OracleController):
            self.prepare_oracle_future_data()
    
    def prepare_oracle_future_data(self):
        """Extract full day's clean energy and task requirements for oracle"""
        # Get clean energy for entire day
        clean_energy_series = self.get_full_day_clean_energy()
        
        # Generate task requirements for entire day
        task_requirements = self.generate_full_day_task_requirements()
        
        # Re-initialize oracle with full data
        self.controller = OracleController(
            clean_energy_series=clean_energy_series,
            task_requirements=task_requirements,
            config=self.config
        )
    
    def _execute_task(self, task):
        """Enhanced task execution with oracle DP lookup"""
        # ... existing code until controller decision ...
        
        # Oracle uses fast DP matrix lookup
        choice = self.controller.select_model(
            battery_level=battery_level,
            clean_energy_percentage=clean_energy_pct,
            user_accuracy_requirement=task.accuracy_requirement,
            user_latency_requirement=task.latency_requirement,
            available_models=available_models,
        )
        
        # ... rest of existing execution logic ...
```

### Phase 3: Training Data Generation with Multiprocessing

**Parallel Training Data Pipeline:**
```python
def generate_full_horizon_training_data():
    config = ConfigLoader().get_simulation_config()
    max_workers = config.workers.max_workers  # From config.jsonc
    
    # Create work items: (location, season, day) combinations
    work_items = []
    for location in config.locations:           # ["CA", "FL", "NW", "NY"]
        for season in config.seasons:           # ["winter", "spring", "summer", "fall"]
            for day in get_representative_day(season):  # 1 day per season
                work_items.append((location, season, day))
    
    # Parallel processing using config workers
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(generate_day_training_data, item, config)
            for item in work_items
        ]
        
        # Collect results
        all_training_data = []
        for future in as_completed(futures):
            day_data = future.result()
            all_training_data.extend(day_data)
    
    return all_training_data

def generate_day_training_data(work_item, config):
    """Generate training data for one day using full-horizon MILP"""
    location, season, day = work_item
    
    # Extract full day's clean energy series
    clean_energy_series = get_clean_energy_series(location, day)
    
    # Generate task requirements for the day
    task_requirements = generate_task_requirements(config, day)
    
    # Solve full-horizon MILP (same as oracle)
    optimal_schedule = solve_full_horizon_milp(
        clean_energy_series, task_requirements, config
    )
    
    # Extract training examples from optimal schedule
    training_examples = []
    for t, (model, charge) in enumerate(optimal_schedule):
        example = {
            "battery_level": calculate_battery_at_timestep(t, optimal_schedule, config),
            "clean_energy_percentage": clean_energy_series[t],
            "accuracy_requirement": task_requirements[t]["accuracy"],
            "latency_requirement": task_requirements[t]["latency"],
            "optimal_model": model,
            "should_charge": charge
        }
        training_examples.append(example)
    
    return training_examples
```

### Phase 4: Runtime Configuration Integration

**Config-Driven Parameters:**
```python
def solve_full_horizon_milp(clean_energy_series, task_requirements, config):
    """Shared MILP solver for both oracle and training data"""
    
    # Use runtime config parameters
    task_interval = config.simulation.task_interval_seconds      # 5s
    battery_capacity = config.battery.capacity_wh               # 5.0Wh
    charge_rate = config.battery.charge_rate_watts              # 100W
    accuracy_req = config.simulation.user_accuracy_requirement   # 45.0%
    latency_req = config.simulation.user_latency_requirement     # 8.0s
    
    # MILP formulation using config parameters
    prob = pulp.LpProblem("Full_Horizon_Optimization", pulp.LpMaximize)
    
    # Decision variables for all timesteps
    model_vars = {}
    charge_vars = {}
    
    for t in range(len(clean_energy_series)):
        model_vars[t] = {
            name: pulp.LpVariable(f"model_{t}_{name}", cat="Binary")
            for name in available_models.keys()
        }
        charge_vars[t] = pulp.LpVariable(f"charge_{t}", cat="Binary")
    
    # Objective: maximize clean energy usage
    prob += pulp.lpSum([
        clean_energy_series[t] * charge_vars[t] 
        for t in range(len(clean_energy_series))
    ])
    
    # Constraints using config parameters
    # Battery dynamics, task requirements, etc.
    
    return optimal_schedule
```

## Implementation Tasks

### Phase 1: Oracle Controller Enhancement
- [x] Design full-horizon MILP formulation for oracle controller
- [x] Implement enhanced OracleController with DP matrix system
- [x] Add pre-computation of optimal schedule for entire day
- [x] Implement fast DP matrix lookup for runtime decisions
- [x] Add should_charge() method to return pre-computed charging decisions

### Phase 2: Simulation Engine Integration
- [x] Update SimulationEngine to prepare oracle future data during initialization
- [x] Implement get_full_day_clean_energy() method
- [x] Implement generate_full_day_task_requirements() method
- [x] Modify _execute_task() to work with oracle DP matrix lookup
- [x] Add get_current_timestep() method for oracle controller

### Phase 3: Training Data Generation
- [x] Design full-horizon training data generation pipeline
- [x] Implement shared MILP solver for both oracle and training data
- [x] Create generate_full_horizon_training_data() function
- [x] Implement generate_day_training_data() for single day processing
- [x] Add parallel processing using config.workers.max_workers
- [x] Implement training data extraction from optimal schedule

### Phase 4: Runtime Configuration
- [x] Integrate ConfigLoader for runtime parameter reading
- [x] Update MILP solver to use config.jsonc parameters
- [x] Ensure both oracle and training use same runtime config
- [x] Add config parameter validation

### Phase 5: Integration and Testing
- [x] Update simulation runner to pass future data to oracle
- [x] Test oracle controller with full-horizon optimization
- [ ] Validate training data generation produces optimal examples
- [ ] Test parallel processing with config workers
- [ ] Compare improved controllers against baseline

### Phase 6: Data Structure and Utilities
- [x] Define representative days per season (Jan 15, Apr 15, Jul 15, Oct 15)
- [x] Implement clean energy series extraction for full day
- [x] Create task requirement generation using config parameters
- [x] Add battery state calculation for training data extraction

## Expected Outcomes

### Oracle Controller Performance
- Pre-computed optimal decisions (no runtime MILP solving)
- Fast DP matrix lookup during simulation
- True optimal performance with full future knowledge
- Runtime config parameter integration

### Training Data Quality
- Truly optimal decisions from full-horizon optimization
- 4,608 training examples (16 days × 288 timesteps)
- Parallel generation using config workers
- Runtime config parameter consistency

### System Integration
- Seamless integration with existing simulation engine
- Config-driven behavior adapts to parameter changes
- Efficient multiprocessing for training data generation
- Shared MILP formulation ensures consistency

## Data Volume Estimates
- 1 day × 288 intervals = 288 examples per location
- 4 locations × 4 seasons × 1 day = 16 days total
- 16 × 288 = 4,608 training examples total

## Configuration Parameters Used
- `simulation.task_interval_seconds`: 5s (288 timesteps per day)
- `simulation.user_accuracy_requirement`: 45.0%
- `simulation.user_latency_requirement`: 8.0s
- `battery.capacity_wh`: 5.0Wh
- `battery.charge_rate_watts`: 100W
- `locations`: ["CA", "FL", "NW", "NY"]
- `seasons`: ["winter", "spring", "summer", "fall"]
- `workers.max_workers`: 100 (for parallel processing)