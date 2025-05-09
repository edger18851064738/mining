#!/usr/bin/env python3
"""
Main entry point for the mining dispatch system.

This script initializes and runs the mining vehicle coordination system with 
either simulation mode or interactive mode for manual control.
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime, timedelta
import random

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Import utility modules
from utils.config import get_config, load_config
from utils.logger import get_logger
from utils.geo.coordinates import Point2D, BoundingBox

# Import domain modules
from domain.environment.mining_environment import MiningEnvironment, ZoneType
from domain.vehicles.mining_vehicle import MiningVehicle, DumpTruck, Excavator, SupportVehicle
from domain.tasks.transport_task import TransportTask, LoadingTask, UnloadingTask
from domain.tasks.task_status import TaskPriority, TaskStatus

# Import coordination modules
from coordination.dispatcher.mining_dispatcher import MiningDispatcher
from coordination.allocation.priority_allocator import PriorityAllocator
from coordination.simulation.simulator import Simulator, SimulationConfig ,SimulationStatus

# Configure logger
logger = get_logger("main")

def setup_environment(config):
    """Create and initialize the mining environment."""
    logger.info("Setting up mining environment")
    
    # Get environment size from config
    size = config.map.grid_size
    
    # Create environment
    env = MiningEnvironment(
        name="Mining Site Alpha",
        bounds=(0, 0, size, size),
        resolution=5.0,  # 5 meters per grid cell
        config=vars(config.map)
    )
    
    # Generate random terrain features
    env.create_terrain_features()
    
    logger.info(f"Created mining environment of size {size}x{size}")
    return env

def create_vehicles(environment, config, count=None):
    """Create vehicles for the mining operation."""
    logger.info("Creating mining vehicles")
    
    # Determine number of vehicles
    if count is None:
        count = getattr(config, 'vehicle_count', 5)
    
    # Get base locations
    base_location = None
    for location_name, location in environment.key_locations.items():
        if 'parking' in location_name.lower():
            base_location = location
            break
    
    if base_location is None:
        base_location = environment.center
    
    # Create vehicles
    vehicles = {}
    
    # Create dump trucks (60% of fleet)
    num_trucks = max(1, int(count * 0.6))
    for i in range(num_trucks):
        truck = DumpTruck(
            vehicle_id=f"TRUCK-{i+1:03d}",
            environment=environment
        )
        # Set initial position
        truck.current_location = Point2D(
            base_location.x + random.uniform(-50, 50),
            base_location.y + random.uniform(-50, 50)
        )
        vehicles[truck.vehicle_id] = truck
    
    # Create excavators (30% of fleet)
    num_excavators = max(1, int(count * 0.3))
    for i in range(num_excavators):
        excavator = Excavator(
            vehicle_id=f"EXCV-{i+1:03d}",
            environment=environment
        )
        # Set initial position near loading zones if possible
        load_zones = environment.get_zones_by_type(ZoneType.LOADING)
        if load_zones and i < len(load_zones):
            zone = load_zones[i]
            # Place within zone
            excavator.current_location = Point2D(
                base_location.x + random.uniform(-200, 200),
                base_location.y + random.uniform(-200, 200)
            )
        else:
            # Fallback placement
            excavator.current_location = Point2D(
                base_location.x + random.uniform(-200, 200),
                base_location.y + random.uniform(-200, 200)
            )
        vehicles[excavator.vehicle_id] = excavator
    
    # Create support vehicles (10% of fleet)
    num_support = max(1, count - num_trucks - num_excavators)
    for i in range(num_support):
        support = SupportVehicle(
            vehicle_id=f"SUPP-{i+1:03d}",
            environment=environment
        )
        # Set initial position
        support.current_location = Point2D(
            base_location.x + random.uniform(-100, 100),
            base_location.y + random.uniform(-100, 100)
        )
        vehicles[support.vehicle_id] = support
    
    logger.info(f"Created {len(vehicles)} vehicles: {num_trucks} trucks, {num_excavators} excavators, {num_support} support")
    return vehicles

def create_initial_tasks(environment, config, count=5):
    """Create initial tasks for the mining operation."""
    logger.info("Creating initial tasks")
    
    tasks = {}
    
    # Find loading and unloading zones
    loading_zones = environment.get_zones_by_type(ZoneType.LOADING)
    unloading_zones = environment.get_zones_by_type(ZoneType.UNLOADING)
    
    # Default coordinates in case zones are not available
    default_load_point = Point2D(100, 100)
    default_unload_point = Point2D(900, 900)
    
    # Create transport tasks
    for i in range(count):
        # Randomly select loading and unloading points
        if loading_zones:
            load_zone = random.choice(loading_zones)
            # Get a point within the zone
            load_point = Point2D(
                load_zone.bounds.min_point.x + random.uniform(0, load_zone.bounds.width),
                load_zone.bounds.min_point.y + random.uniform(0, load_zone.bounds.height)
            )
        else:
            load_point = default_load_point
        
        if unloading_zones:
            unload_zone = random.choice(unloading_zones)
            unload_point = Point2D(
                unload_zone.bounds.min_point.x + random.uniform(0, unload_zone.bounds.width),
                unload_zone.bounds.min_point.y + random.uniform(0, unload_zone.bounds.height)
            )
        else:
            unload_point = default_unload_point
        
        # Create task with random priority
        priority = random.choice(list(TaskPriority))
        amount = random.uniform(10000, 50000)  # Random amount between 10-50 tons
        
        task = TransportTask(
            task_id=f"TASK-{i+1:03d}",
            start_point=load_point,
            end_point=unload_point,
            material_type="ore",
            amount=amount,
            priority=priority
        )
        
        tasks[task.task_id] = task
    
    logger.info(f"Created {len(tasks)} initial tasks")
    return tasks

def setup_dispatcher(environment, vehicles=None, tasks=None):
    """Set up the mining dispatcher."""
    logger.info("Setting up mining dispatcher")
    
    # Create dispatcher
    dispatcher = MiningDispatcher(environment=environment)
    
    # Set up allocator
    from coordination.allocation.base import AllocationConfig, AllocationStrategy, AllocationObjective
    allocator_config = AllocationConfig(
        strategy=AllocationStrategy.PRIORITY,
        objective=AllocationObjective.MINIMIZE_MAKESPAN,
        max_tasks_per_vehicle=3
    )
    dispatcher._init_allocator()
    
    # Add vehicles if provided
    if vehicles:
        for vehicle_id, vehicle in vehicles.items():
            dispatcher.add_vehicle(vehicle)
    
    # Add tasks if provided
    if tasks:
        for task_id, task in tasks.items():
            dispatcher.add_task(task)
    
    logger.info("Mining dispatcher initialized")
    return dispatcher

def setup_simulator(environment, dispatcher, config):
    """Set up the simulation system."""
    logger.info("Setting up simulator")
    
    # Create simulation config
    sim_config = SimulationConfig(
        real_time_factor=getattr(config.ui, 'default_simulation_speed', 10.0),
        task_generation_rate=1.0,  # 1 task per minute
        vehicle_count=len(dispatcher.get_all_vehicles()) if dispatcher else 0
    )
    
    # Create simulator
    simulator = Simulator(
        environment=environment,
        dispatcher=dispatcher,
        config=sim_config
    )
    
    logger.info("Simulator initialized")
    return simulator

def print_simulation_status(simulator):
    """Print current simulation status and metrics."""
    if not simulator:
        print("Simulator not initialized")
        return
    
    # Get current time and status
    current_time = simulator.clock.current_time
    status = simulator.status.name
    
    # Get metrics
    metrics = simulator.get_metrics()
    
    # Print header
    print("\n" + "="*60)
    print(f"SIMULATION STATUS: {status}")
    print(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Elapsed time: {simulator.clock.elapsed_time}")
    print("-"*60)
    
    # Print key metrics
    print("METRICS:")
    print(f"  Vehicles: {metrics.total_vehicles}")
    print(f"  Tasks: {metrics.total_tasks} total, {metrics.completed_tasks} completed, {metrics.failed_tasks} failed")
    print(f"  Distance traveled: {metrics.total_distance_traveled:.2f} meters")
    print(f"  Throughput: {metrics.throughput:.2f} tasks/hour")
    print(f"  Vehicle utilization: {metrics.average_vehicle_utilization:.2f}%")
    print("="*60)

def print_vehicles_status(dispatcher):
    """Print status of all vehicles."""
    if not dispatcher:
        print("Dispatcher not initialized")
        return
    
    # Get all vehicles
    vehicles = dispatcher.get_all_vehicles()
    
    # Print header
    print("\n" + "="*60)
    print(f"VEHICLES STATUS ({len(vehicles)} total)")
    print("-"*60)
    
    # Print each vehicle
    for vehicle_id, vehicle in sorted(vehicles.items()):
        state = vehicle.state.name if hasattr(vehicle.state, 'name') else str(vehicle.state)
        pos = vehicle.current_location
        load = f"{vehicle.load_ratio*100:.1f}%" if hasattr(vehicle, 'load_ratio') else "N/A"
        tasks = dispatcher.get_assignments().get(vehicle_id, [])
        
        print(f"{vehicle_id} | Type: {vehicle.__class__.__name__} | State: {state}")
        print(f"  Position: ({pos.x:.1f}, {pos.y:.1f}) | Load: {load} | Tasks: {len(tasks)}")
    
    print("="*60)

def print_tasks_status(dispatcher):
    """Print status of all tasks."""
    if not dispatcher:
        print("Dispatcher not initialized")
        return
    
    # Get all tasks
    tasks = dispatcher.get_all_tasks()
    assignments = dispatcher.get_assignments()
    
    # Create reverse mapping (task -> vehicle)
    task_to_vehicle = {}
    for vehicle_id, task_ids in assignments.items():
        for task_id in task_ids:
            task_to_vehicle[task_id] = vehicle_id
    
    # Print header
    print("\n" + "="*60)
    print(f"TASKS STATUS ({len(tasks)} total)")
    print("-"*60)
    
    # Group tasks by status
    grouped_tasks = {}
    for task_id, task in tasks.items():
        status = task.status
        if status not in grouped_tasks:
            grouped_tasks[status] = []
        grouped_tasks[status].append((task_id, task))
    
    # Print tasks by status group
    for status, task_list in sorted(grouped_tasks.items()):
        print(f"\n{status} Tasks ({len(task_list)}):")
        
        for task_id, task in sorted(task_list):
            priority = task.priority.name if hasattr(task.priority, 'name') else str(task.priority)
            progress = f"{task.progress*100:.1f}%" if hasattr(task, 'progress') else "N/A"
            assigned_to = task_to_vehicle.get(task_id, "None")
            
            print(f"{task_id} | Type: {task.__class__.__name__} | Priority: {priority}")
            print(f"  Progress: {progress} | Assigned to: {assigned_to}")
    
    print("="*60)

def run_simulation_menu(environment, dispatcher, simulator):
    """Run the simulation with an interactive menu."""
    running = True
    
    # Start with a status display
    print_simulation_status(simulator)
    
    while running:
        print("\nMINING DISPATCH SYSTEM - SIMULATION MENU")
        print("1. Start/Resume simulation")
        print("2. Pause simulation")
        print("3. Step simulation (1 minute)")
        print("4. Run for specific duration")
        print("5. Show simulation status")
        print("6. Show vehicles status")
        print("7. Show tasks status")
        print("8. Create new task")
        print("9. Create new vehicle")
        print("0. Exit")
        
        choice = input("\nEnter choice: ")
        
        if choice == "1":
            # Start/Resume simulation
            if simulator.status == SimulationStatus.STOPPED:
                simulator.start()
            elif simulator.status == SimulationStatus.PAUSED:
                simulator.resume()
            print("Simulation running...")
            
        elif choice == "2":
            # Pause simulation
            simulator.pause()
            print("Simulation paused")
            
        elif choice == "3":
            # Step simulation
            simulator.step(60)  # Step 1 minute
            print("Simulation stepped by 1 minute")
            print_simulation_status(simulator)
            
        elif choice == "4":
            # Run for duration
            try:
                minutes = float(input("Enter duration in minutes: "))
                simulator.run_for(timedelta(minutes=minutes))
                print(f"Simulation run for {minutes} minutes")
                print_simulation_status(simulator)
            except ValueError:
                print("Invalid input. Please enter a number.")
            
        elif choice == "5":
            # Show simulation status
            print_simulation_status(simulator)
            
        elif choice == "6":
            # Show vehicles status
            print_vehicles_status(dispatcher)
            
        elif choice == "7":
            # Show tasks status
            print_tasks_status(dispatcher)
            
        elif choice == "8":
            # Create new task
            try:
                # Get task parameters
                start_x = float(input("Enter start X coordinate: "))
                start_y = float(input("Enter start Y coordinate: "))
                end_x = float(input("Enter end X coordinate: "))
                end_y = float(input("Enter end Y coordinate: "))
                
                # Create task
                task = TransportTask(
                    start_point=Point2D(start_x, start_y),
                    end_point=Point2D(end_x, end_y),
                    material_type="ore",
                    amount=40000.0,
                    priority=TaskPriority.NORMAL
                )
                
                # Add to dispatcher
                dispatcher.add_task(task)
                print(f"Task {task.task_id} created and added to dispatcher")
                
            except ValueError:
                print("Invalid input. Please enter valid coordinates.")
            
        elif choice == "9":
            # Create new vehicle
            try:
                # Get vehicle type
                print("Vehicle types:")
                print("1. Dump Truck")
                print("2. Excavator")
                print("3. Support Vehicle")
                vehicle_type = input("Enter vehicle type (1-3): ")
                
                # Get position
                pos_x = float(input("Enter X coordinate: "))
                pos_y = float(input("Enter Y coordinate: "))
                
                # Create vehicle based on type
                vehicle = None
                if vehicle_type == "1":
                    vehicle = DumpTruck(environment=environment)
                elif vehicle_type == "2":
                    vehicle = Excavator(environment=environment)
                elif vehicle_type == "3":
                    vehicle = SupportVehicle(environment=environment)
                else:
                    print("Invalid vehicle type. Creating dump truck.")
                    vehicle = DumpTruck(environment=environment)
                
                # Set position
                vehicle.current_location = Point2D(pos_x, pos_y)
                
                # Add to dispatcher
                dispatcher.add_vehicle(vehicle)
                print(f"Vehicle {vehicle.vehicle_id} created and added to dispatcher")
                
            except ValueError:
                print("Invalid input. Please enter valid coordinates.")
            
        elif choice == "0":
            # Exit
            running = False
            print("Exiting simulation")
            
        else:
            print("Invalid choice, please try again.")
    
    # Stop simulation before exiting
    simulator.stop()

def main():
    """Main entry point for the mining dispatch system."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Mining Dispatch System")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--vehicle-count", type=int, help="Number of vehicles to create")
    parser.add_argument("--task-count", type=int, default=5, help="Number of initial tasks to create")
    parser.add_argument("--simulation", action="store_true", help="Run in simulation mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        load_config(args.config)
    
    # Set log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get system configuration
    config = get_config()
    
    try:
        # Set up environment
        environment = setup_environment(config)
        
        # Create vehicles
        vehicles = create_vehicles(environment, config, count=args.vehicle_count)
        
        # Create initial tasks
        tasks = create_initial_tasks(environment, config, count=args.task_count)
        
        # Set up dispatcher
        dispatcher = setup_dispatcher(environment, vehicles, tasks)
        
        # Set up simulator
        simulator = setup_simulator(environment, dispatcher, config)
        
        # Run simulation interactive menu
        run_simulation_menu(environment, dispatcher, simulator)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        print(f"An error occurred: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())