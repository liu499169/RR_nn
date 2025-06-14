# run_simulation.py
import os
import torch

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
DTYPE = torch.float32 # float16

# Paths
BASIN_JSON_PATH = r'data/watershed_processed/FromGIS_d2k2_eqdx_20m.json' 
HDF5_OUTPUT_PATH_phase1 = r'data/output/eqdx_d2k2_phase1_20m.hdf5'
HDF5_OUTPUT_PATH_phase2 = r'data/output/eqdx_d2k2_phase2_20m.hdf5' 


## Tao: Q_peak of channel 64 with dx=20m is 5% larger than that of dx=10m
# BASIN_JSON_PATH = r'data/watershed_processed/FromGIS_d2k2_eqdx_to6_10m.json' 
# HDF5_OUTPUT_PATH_phase1 = r'data/output/eqdx_d2k2_to6_phase1_10m.hdf5'
# HDF5_OUTPUT_PATH_phase2 = r'data/output/eqdx_d2k2_to6_phase2_10m.hdf5' 
# BASIN_JSON_PATH = r'data/watershed_processed/FromGIS_d2k2_eqdx_to6_20m.json' 
# HDF5_OUTPUT_PATH_phase1 = r'data/output/eqdx_d2k2_to6_phase1_20m.hdf5'
# HDF5_OUTPUT_PATH_phase2 = r'data/output/eqdx_d2k2_to6_phase2_20m.hdf5' 


# Global Parameter Overrides (values are Python floats/ints, basin_loader handles conversion)
GLOBAL_PARAM_OVERRIDES = {
    'plane':   {'MAN': 0.10, 'Ks': 0, 'Hf_max': 0, 'theta_init_condition': 0.1, 'm_exponent': 0, 'k_drain':0e-07}, 
    'channel': {'MAN': 0.06, 'Ks': 0, 'Hf_max': 0, 'theta_init_condition': 0.1, 'm_exponent': 0, 'k_drain':0e-07}
}
# List of parameters to make learnable (case-insensitive matching in loader)
LEARNABLE_PARAMS_LIST = ['man', 'ks', 'hf_max', 'm_exponent'] 

# Simulation Settings
SIMULATION_SETTINGS = {
    'sim_duration_min': 30.0,    # Total simulation time
    'cfl_number': 0.9, # Courant number
    'max_dt_min': 1.0, # Optional Max allowed dt in minutes
    'min_dt_min': 9e-3, # Optional Min allowed dt in minutes
    'rain_event_dur_min': 15.0, # Duration of the rainfall event
    'rain_peak_t_min': 10.0,    # Time when rainfall intensity peaks
    'rain_peak_mmhr': 15,     # Peak rainfall intensity
    'save_interval_min': 1.0     # How often to save results to HDF5
}

# --- End Configuration ---

# Import modules
from src.data_structures import ElementProperties, SoilPropertiesIntermediate, OverlandFlowState, InfiltrationState
from src.watershed import Watershed
from src.phased_engine import PhasedSimulationEngine #, PhasedSimulationEngine_1
from src.io.results_handler import ResultSaver
from src.utils.rainfall_generator import generate_triangular_rainfall # For ResultSaver init
from src.io.config_saver import save_simulation_configuration
from src.core.physics_formulas import (get_h_from_trapezoid_area, get_plane_h_from_area,
                                        get_trapezoid_topwidth_from_h, get_trapezoid_wp_from_h,
                                        calculate_q_manning, calculate_froude_number, calculate_cfl_number)

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
    # --- Save Resolved Configuration ---
    config_output_filename = os.path.splitext(HDF5_OUTPUT_PATH_phase2)[0] + "_resolved_config.json"
    save_simulation_configuration(
        filepath=config_output_filename,
        simulation_settings=SIMULATION_SETTINGS,
        global_param_overrides=GLOBAL_PARAM_OVERRIDES, # 
        learnable_params_list=LEARNABLE_PARAMS_LIST,
        watershed_obj=watershed_manager, # Pass the Watershed object
        element_properties_map=watershed_manager.element_properties,
        soil_parameters_map=watershed_manager.soil_parameters,
        simulation_order=watershed_manager.simulation_order,
        connectivity_map=watershed_manager.connectivity
    )

    # 3. Initialize Result Saver
    output_dir = os.path.dirname(HDF5_OUTPUT_PATH_phase1)
    if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
    
    result_saver_phase1 = ResultSaver(
        filepath=HDF5_OUTPUT_PATH_phase1, # Phase 1 output path
        element_props_map=watershed_manager.element_properties, # Pass the dict of ElementProperties
        sim_duration_s=SIMULATION_SETTINGS['sim_duration_min'] * 60.0,
        save_interval_s=SIMULATION_SETTINGS['save_interval_min'] * 60.0,
        rainfall_mmhr_cpu=rain_rates_mmhr_for_hdf5_metadata,
        batch_save_size=10, # Example, make configurable
        enable_batching=True 
    )

    result_saver_phase2 = ResultSaver(
        filepath=HDF5_OUTPUT_PATH_phase2, # Phase 1 output path
        element_props_map=watershed_manager.element_properties, # Pass the dict of ElementProperties
        sim_duration_s=SIMULATION_SETTINGS['sim_duration_min'] * 60.0,
        save_interval_s=SIMULATION_SETTINGS['save_interval_min'] * 60.0,
        rainfall_mmhr_cpu=rain_rates_mmhr_for_hdf5_metadata,
        batch_save_size=10, # Example, make configurable
        enable_batching=True 
    )

    # 4. Initialize Simulation Engine
    print("\nInitializing Simulation Engine...")
    engine = PhasedSimulationEngine(
        watershed_obj=watershed_manager,
        result_saver_planes=result_saver_phase1, # Tao: 
        result_saver_channels=result_saver_phase2,
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
        # engine._run_phase1_planes()
        engine.run_decoupled_simulation()

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