# run_simulation.py
import torch
import os
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

# --- Configuration (move to a JSON config file later) ---
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    gpu_properties = torch.cuda.get_device_properties(0)
    print(f"CUDA device: {gpu_properties.name} selected.")
else:
    DEVICE = torch.device('cpu')
    print("CUDA not available. Using CPU.")
# DEVICE = torch.device('cpu')
DTYPE = torch.float32

# Paths
BASIN_JSON_PATH = r'data/watershed_processed/FromGIS_d2k2_rdp_to6.json' 
HDF5_OUTPUT_PATH = r'data/output/rdp_d2k2_to6.hdf5' 

# Global Parameter Overrides (values are Python floats/ints, basin_loader handles conversion)
GLOBAL_PARAM_OVERRIDES = {
    'plane':   {'MAN': 0.10, 'Ks': 12, 'Hf_max': 20, 'theta_init_condition': 0.3, 'm_exponent': 0.6, 'k_drain':2e-07}, 
    'channel': {'MAN': 0.05, 'Ks': 12, 'Hf_max': 10, 'theta_init_condition': 0.8, 'm_exponent': 0.8, 'k_drain':1e-07}
}
# List of parameters to make learnable (case-insensitive matching in loader)
LEARNABLE_PARAMS_LIST = ['man', 'ks', 'hf_max', 'm_exponent'] 

# Simulation Settings
SIMULATION_SETTINGS = {
    'sim_duration_min': 30.0,    # Total simulation time
    'cfl_number': 0.95, # Courant number
    'max_dt_min': 1.0, # Optional Max allowed dt in minutes
    'min_dt_min': 1e-2, # Optional Min allowed dt in minutes
    'rain_event_dur_min': 20.0, # Duration of the rainfall event
    'rain_peak_t_min': 10.0,    # Time when rainfall intensity peaks
    'rain_peak_mmhr': 60.0,     # Peak rainfall intensity
    'save_interval_min': 1.0     # How often to save results to HDF5
}

# --- End Configuration ---

# Import your new modules
# from src.data_structures import ElementProperties, SoilPropertiesIntermediate, OverlandFlowState, InfiltrationState
from src.watershed import Watershed
from src.simulation_engine import SimulationEngine
from src.io.results_handler import ResultSaver
from src.utils.rainfall_generator import generate_triangular_rainfall # For ResultSaver init

from torch.profiler import profile, ProfilerActivity

def main_simulation_run():
    print(f"--- Main Simulation Script ---")
    print(f"Using device: {DEVICE}, dtype: {DTYPE}")

    # 1. Prepare rainfall data for ResultSaver's metadata (needs mm/hr, CPU tensor)
    #    SimulationEngine generates its own rain series for interpolation.
    _, _, rain_rates_mmhr_for_hdf5_metadata = generate_triangular_rainfall(
        SIMULATION_SETTINGS['sim_duration_min'], SIMULATION_SETTINGS['rain_event_dur_min'],
        SIMULATION_SETTINGS['rain_peak_t_min'], SIMULATION_SETTINGS['rain_peak_mmhr'],
        dt_min_for_gen=max(0.001, SIMULATION_SETTINGS['min_dt_min'] / 5.0 if SIMULATION_SETTINGS['min_dt_min'] > 0 else 0.01),
        save_interval_min=SIMULATION_SETTINGS['save_interval_min'] 
    )
    rain_rates_mmhr_for_hdf5_metadata = rain_rates_mmhr_for_hdf5_metadata.cpu()


    # 2. Initialize Watershed (loads JSON, creates element nn.Modules)
    print("\nInitializing Watershed...")
    watershed_manager = Watershed(
        basin_json_path=BASIN_JSON_PATH,
        global_param_overrides=GLOBAL_PARAM_OVERRIDES,
        learnable_params_list=LEARNABLE_PARAMS_LIST,
        device=DEVICE,
        dtype=DTYPE
    )
    print(f"Watershed loaded: {len(watershed_manager.element_modules)} element modules created.")

    # 3. Initialize Result Saver
    output_dir = os.path.dirname(HDF5_OUTPUT_PATH)
    if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
    
    result_saver_instance = ResultSaver(
        filepath=HDF5_OUTPUT_PATH,
        element_props_map=watershed_manager.element_properties, # Pass the dict of ElementProperties
        sim_duration_s=SIMULATION_SETTINGS['sim_duration_min'] * 60.0,
        save_interval_s=SIMULATION_SETTINGS['save_interval_min'] * 60.0,
        rainfall_mmhr_cpu=rain_rates_mmhr_for_hdf5_metadata,
        batch_save_size=10, # Example, make configurable
        enable_batching=True 
    )

    # 4. Initialize Simulation Engine
    print("\nInitializing Simulation Engine...")
    engine = SimulationEngine(
        watershed_obj=watershed_manager,
        result_saver=result_saver_instance,
        simulation_settings=SIMULATION_SETTINGS,
        device=DEVICE,
        dtype=DTYPE
    )

    # 5. Run Simulation
    # --- Profiler Setup (Optional) ---
    run_with_profiler = False # Set to True to enable profiler # False
    prof_ctx = None
    # ----------------------------------

    if run_with_profiler:
        profiler_activities = [ProfilerActivity.CPU]
        if DEVICE.type == 'cuda' and torch.cuda.is_available():
            profiler_activities.append(ProfilerActivity.CUDA)
        
        print(f"\nRunning simulation WITH PROFILER (Activities: {profiler_activities})...")
        with profile(activities=profiler_activities, 
                    #  record_shapes=True, 
                    #  profile_memory=True, 
                    #  with_stack=True,
                     # on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs") # For TensorBoard
                     ) as prof:
            engine.run()
        prof_ctx = prof
    else:
        print("\nRunning simulation WITHOUT profiler...")
        engine.run()

    # --- Profiler Reporting (if run) ---
    if prof_ctx is not None:
        print("\n--- Profiler Results ---")
        sort_key = "cuda_time_total" if ProfilerActivity.CUDA in profiler_activities and DEVICE.type == 'cuda' else "cpu_time_total"
        try:
            print(prof_ctx.key_averages(group_by_input_shape=True).table(sort_by=sort_key, row_limit=15))
            print(f"\nAggregated by operator name (top 15 by {sort_key}):")
            print(prof_ctx.key_averages().table(sort_by=sort_key, row_limit=15))
            trace_file = "sim_trace_nn_module.json"
            prof_ctx.export_chrome_trace(trace_file)
            print(f"Profiler trace exported to {trace_file}")
        except Exception as e:
            print(f"Error displaying/exporting profiler results: {e}")
            print("Raw profiler averages:")
            print(prof_ctx.key_averages().table(row_limit=15))
    
    print("\n--- Main script finished. ---")

if __name__ == '__main__':
    main_simulation_run()