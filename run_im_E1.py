import os
import torch
import gc
import json

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
# Load configuration from external JSON file
CONFIG_PATH = "configs/E1_config_im.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

EVENT_ID = config['EVENT_ID']
EVENT_DATE = config['EVENT_DATE']
EVENT_NAME_HDF = f"{EVENT_ID}_{EVENT_DATE}"
PREPARED_RAINFALL_PATH = config['PREPARED_RAINFALL_PATH'].format(EVENT_NAME_HDF=EVENT_NAME_HDF)

BASIN_JSON_PATH = config['BASIN_JSON_PATH']
HDF5_OUTPUT_PATH_phase1 = config['HDF5_OUTPUT_PATH_phase1'].format(EVENT_NAME_HDF=EVENT_NAME_HDF)
HDF5_OUTPUT_PATH_phase2 = config['HDF5_OUTPUT_PATH_phase2'].format(EVENT_NAME_HDF=EVENT_NAME_HDF)

GLOBAL_PARAM_OVERRIDES = config['GLOBAL_PARAM_OVERRIDES']
LEARNABLE_PARAMS_LIST = config['LEARNABLE_PARAMS_LIST']
SIMULATION_SETTINGS = config['SIMULATION_SETTINGS']

# --- End Configuration ---

# Import modules
from src.data_structures import ElementProperties, SoilPropertiesIntermediate, OverlandFlowState, InfiltrationState
from src.watershed import Watershed_im
from src.phased_engine import PhasedSimulationEngine_v3
from src.io.results_handler import ResultSaver_v2
from src.io.config_saver import save_simulation_configuration
from src.io.event_loader import PreloadedRainfallManager
from src.core.physics_formulas import (get_h_from_trapezoid_area, get_plane_h_from_area,
                                        get_trapezoid_topwidth_from_h, get_trapezoid_wp_from_h,
                                        calculate_q_manning, calculate_froude_number, calculate_cfl_number)

from torch.profiler import profile, ProfilerActivity
import json

def main_simulation_run():
    print(f"--- Main Simulation Script ---")
    print(f"Using device: {DEVICE}, dtype: {DTYPE}")

    # ========================================================================
    # --- PART 1: RUN AND SAVE PHASE 1 ---
    # ========================================================================

    # 1. Prepare rainfall data 
    rainfall_manager = PreloadedRainfallManager(
    element_rainfall_path=PREPARED_RAINFALL_PATH,
    device=DEVICE,
    dtype=DTYPE
)

    # 2. Initialize Watershed for phase 1 (loads JSON, creates element nn.Modules)
    watershed_manager_p1 = Watershed_im(
        basin_json_path=BASIN_JSON_PATH,
        global_param_overrides=GLOBAL_PARAM_OVERRIDES,
        learnable_params_list=LEARNABLE_PARAMS_LIST,
        device=DEVICE,
        dtype=DTYPE
    )
    print(f"Watershed loaded: {len(watershed_manager_p1.element_modules)} element modules created.")

    # 3. Save the master configuration file
    output_dir = os.path.dirname(HDF5_OUTPUT_PATH_phase1)
    if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)

    config_output_filename = os.path.splitext(HDF5_OUTPUT_PATH_phase2)[0] + "_resolved_config.json"
    # print(f"---- DEBUG: {config_output_filename}--------")
    save_simulation_configuration(
        filepath=config_output_filename,
        simulation_settings=SIMULATION_SETTINGS,
        global_param_overrides=GLOBAL_PARAM_OVERRIDES, # 
        learnable_params_list=LEARNABLE_PARAMS_LIST,
        watershed_obj=watershed_manager_p1, # Pass the Watershed object
        element_properties_map=watershed_manager_p1.element_properties,
        soil_parameters_map=watershed_manager_p1.soil_parameters,
        simulation_order=watershed_manager_p1.simulation_order,
        connectivity_map=watershed_manager_p1.connectivity
    )

    # 4. Initialize ResultSaver for Phase 1
    result_saver_phase1 = ResultSaver_v2(
        filepath=HDF5_OUTPUT_PATH_phase1, # Phase 1 output path
        element_props_map=watershed_manager_p1.element_properties, # Pass the dict of ElementProperties
        sim_duration_s=SIMULATION_SETTINGS['sim_duration_min'] * 60.0,
        save_interval_s=SIMULATION_SETTINGS['save_interval_min'] * 60.0,
        rainfall_mmhr_cpu=torch.zeros(1),
        batch_save_size=10, 
        enable_batching=True 
    )

    # 5. Initialize and run the Phase 1 Engine
    engine_p1 = PhasedSimulationEngine_v3(
        watershed_obj=watershed_manager_p1,
        rainfall_manager=rainfall_manager,
        result_saver_planes=result_saver_phase1, # Tao: 
        result_saver_channels=None, # Phase 1 does not use channels
        simulation_settings=SIMULATION_SETTINGS,
        device=DEVICE,
        dtype=DTYPE
    )

    print("\n--- Running Phase 1 Simulation ---")
    # returns mass balance results; hydrographs are saved internally
    mb_results_p1 = engine_p1._run_phase1_planes() # 

    # Finalize the timeseries data HDF5 file
    if engine_p1.result_saver_planes:
        engine_p1.result_saver_planes.finalize()
    print(f"Phase 1 simulation completed. Results and hydrographs saved to {HDF5_OUTPUT_PATH_phase1}")

    # 6. Clean up all Phase 1 objects to free memory
    print("\n--- Cleaning Up Phase 1 Resources ---")
    del engine_p1
    del watershed_manager_p1
    del result_saver_phase1
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "="*50)
    print("PHASE 1 COMPLETE. VRAM CLEANED.")
    print(f"CUDA Memory Summary After Cleanup:")
    print(torch.cuda.memory_summary())
    print("="*50 + "\n")

    # ========================================================================
    # --- PART 2: LOAD AND RUN PHASE 2 ---
    # ========================================================================
    print("--- Re-initializing Watershed and Engine for Phase 2 ---")

    # 1. Re-initialize Watershed and Saver for Phase 2
    watershed_manager_p2 = Watershed_im(
        basin_json_path=BASIN_JSON_PATH,
        global_param_overrides=GLOBAL_PARAM_OVERRIDES,
        learnable_params_list=LEARNABLE_PARAMS_LIST,
        device=DEVICE,
        dtype=DTYPE
    )

    result_saver_phase2 = ResultSaver_v2(
        filepath=HDF5_OUTPUT_PATH_phase2, # Phase 2 output path
        element_props_map=watershed_manager_p2.element_properties, 
        sim_duration_s=SIMULATION_SETTINGS['sim_duration_min'] * 60.0,
        save_interval_s=SIMULATION_SETTINGS['save_interval_min'] * 60.0,
        rainfall_mmhr_cpu=torch.zeros(1),
        batch_save_size=10, 
        enable_batching=True 
    )

    # 2. Initialize the Phase 2 Engine
    engine_p2 = PhasedSimulationEngine_v3(
        watershed_obj=watershed_manager_p2,
        rainfall_manager=rainfall_manager,
        result_saver_planes=None, # Tao: 
        result_saver_channels=result_saver_phase2, # Phase 1 does not use channels
        simulation_settings=SIMULATION_SETTINGS,
        device=DEVICE,
        dtype=DTYPE,
        phase1_results=mb_results_p1
    )

    # 3. Load the hydrograph data
    engine_p2.load_and_resample_plane_hydrographs(HDF5_OUTPUT_PATH_phase1)

    print("\n--- Running Phase 2 Simulation ---")
    engine_p2._run_phase2_channels()

    # 4. Finalize results and report
    print("\n--- Finalizing Phase 2 ---")
    if engine_p2.result_saver_channels:
        engine_p2.result_saver_channels.finalize()

    engine_p2.report_mass_balance()

    print("\n--- Main script finished. ---")


if __name__ == '__main__':
    # If you ever use multiprocessing, this is essential
    # import torch.multiprocessing as mp
    # mp.set_start_method('spawn', force=True)
    main_simulation_run()