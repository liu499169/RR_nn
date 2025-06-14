# run_simulation.py
import os
import torch
import gc

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
    'sim_duration_min': 12.0,    # Total simulation time
    'cfl_number': 0.9, # Courant number
    'max_dt_min': 1.0, # Optional Max allowed dt in minutes
    'min_dt_min': 9e-3, # Optional Min allowed dt in minutes
    'rain_event_dur_min': 10.0, # Duration of the rainfall event
    'rain_peak_t_min': 3.0,    # Time when rainfall intensity peaks
    'rain_peak_mmhr': 6,     # Peak rainfall intensity
    'save_interval_min': 1.0     # How often to save results to HDF5
}

# --- End Configuration ---

# Import modules
from src.data_structures import ElementProperties, SoilPropertiesIntermediate, OverlandFlowState, InfiltrationState
from src.watershed import Watershed
from src.components.plane_element import PlaneElement # For isinstance check
from src.components.channel_element import ChannelElement # For isinstance check
from src.PhasedSimulationEngine import PhasedSimulationEngine #, PhasedSimulationEngine_1
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
    rain_times_s_cpu, rain_rates_ms_device, rain_rates_mmhr_for_hdf5_metadata = generate_triangular_rainfall(
        SIMULATION_SETTINGS['sim_duration_min'], SIMULATION_SETTINGS['rain_event_dur_min'],
        SIMULATION_SETTINGS['rain_peak_t_min'], SIMULATION_SETTINGS['rain_peak_mmhr'],
        dt_min_for_gen=max(0.001, SIMULATION_SETTINGS['min_dt_min'] / 5.0 if SIMULATION_SETTINGS['min_dt_min'] > 0 else 0.01),
        save_interval_min=SIMULATION_SETTINGS['save_interval_min'] 
    )
    rain_rates_mmhr_for_hdf5_metadata = rain_rates_mmhr_for_hdf5_metadata.cpu()
    rain_rates_ms_device = rain_rates_ms_device.to(device=DEVICE, dtype=DTYPE)

    # 2. Initialize Watershed (loads JSON, creates element nn.Modules)
    print("\n--- Initializing Watershed and Engine for Phase 1 ---")
    watershed_manager_p1 = Watershed(
        basin_json_path=BASIN_JSON_PATH,
        global_param_overrides=GLOBAL_PARAM_OVERRIDES,
        learnable_params_list=LEARNABLE_PARAMS_LIST,
        device=DEVICE,
        dtype=DTYPE
    )

    output_dir = os.path.dirname(HDF5_OUTPUT_PATH_phase1)
    if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
    
    result_saver_phase1 = ResultSaver(
        filepath=HDF5_OUTPUT_PATH_phase1, # Phase 1 output path
        element_props_map=watershed_manager_p1.element_properties, # Pass the dict of ElementProperties
        sim_duration_s=SIMULATION_SETTINGS['sim_duration_min'] * 60.0,
        save_interval_s=SIMULATION_SETTINGS['save_interval_min'] * 60.0,
        rainfall_mmhr_cpu=rain_rates_mmhr_for_hdf5_metadata,
        batch_save_size=10, # Example, make configurable
        enable_batching=True 
    )

    engine_p1 = PhasedSimulationEngine(
        watershed_obj=watershed_manager_p1,
        result_saver_planes=result_saver_phase1,
        result_saver_channels=None, # Not needed
        simulation_settings=SIMULATION_SETTINGS,
        device=DEVICE,
        dtype=DTYPE
    )

    print("\n--- Running Phase 1 Simulation ---")
    plane_modules_p1 = [m for m in watershed_manager_p1.element_modules.values() if isinstance(m, PlaneElement)]
    hydrograph_data, mb_results_p1 = engine_p1._run_phase1_planes(
                        plane_modules = plane_modules_p1, 
                        rainfall_times_s_cpu = rain_times_s_cpu, 
                        rainfall_rates_ms_device = rain_rates_ms_device,
                        output_save_interval_s = SIMULATION_SETTINGS['save_interval_min'] * 60.0,
        ) 
    
    # Finalize the timeseries data HDF5 file
    if engine_p1.result_saver_planes:
        engine_p1.result_saver_planes.finalize()

    # Save the returned hydrographs to the now-closed file
    engine_p1._save_hydrographs_to_file(hydrograph_data, HDF5_OUTPUT_PATH_phase1)

    print("\n--- Cleaning Up Phase 1 Resources ---")
    del hydrograph_data
    del engine_p1
    del watershed_manager_p1
    del result_saver_phase1
    del plane_modules_p1
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n" + "="*50)
    print("CUDA MEMORY SUMMARY AFTER PHASE 1 CLEANUP")
    print(torch.cuda.memory_summary())
    print("="*50 + "\n")
    

    print("--- Re-initializing Watershed and Engine for Phase 2 ---")
    watershed_manager_p2 = Watershed(
        basin_json_path=BASIN_JSON_PATH,
        global_param_overrides=GLOBAL_PARAM_OVERRIDES,
        learnable_params_list=LEARNABLE_PARAMS_LIST,
        device=DEVICE,
        dtype=DTYPE) # Re-init as before
    result_saver_phase2 = ResultSaver(filepath=HDF5_OUTPUT_PATH_phase2, # Phase 1 output path
        element_props_map=watershed_manager_p2.element_properties, # Pass the dict of ElementProperties
        sim_duration_s=SIMULATION_SETTINGS['sim_duration_min'] * 60.0,
        save_interval_s=SIMULATION_SETTINGS['save_interval_min'] * 60.0,
        rainfall_mmhr_cpu=rain_rates_mmhr_for_hdf5_metadata,
        batch_save_size=10, # Example, make configurable
        enable_batching=True) # Re-init as before

    engine_p2 = PhasedSimulationEngine(
        watershed_obj=watershed_manager_p2,
        result_saver_planes=None,
        result_saver_channels=result_saver_phase2,
        simulation_settings=SIMULATION_SETTINGS,
        device=DEVICE,
        dtype=DTYPE,
        phase1_results=mb_results_p1 # <-- Pass the mass balance results here
    )

    engine_p2.load_and_resample_plane_hydrographs(HDF5_OUTPUT_PATH_phase1)

    print("\n--- Running Phase 2 Simulation ---")
    channel_modules_p2 = [m for m in watershed_manager_p2.element_modules.values() if isinstance(m, ChannelElement)]
    engine_p2._run_phase2_channels(channel_modules_p2, rain_times_s_cpu, rain_rates_ms_device) # Pass required args

    print("\n--- Finalizing Phase 2 ---")
    if engine_p2.result_saver_channels:
        engine_p2.result_saver_channels.finalize()
    
    engine_p2.report_mass_balance() # This will now work correctly
    
    print("\n--- Main script finished. ---")


if __name__ == '__main__':
    main_simulation_run()