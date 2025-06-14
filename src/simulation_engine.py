# src/simulation_engine.py

import torch
import numpy as np
from tqdm import tqdm
# from collections import defaultdict

# Import from our defined structures and modules
from src.data_structures import OverlandFlowState, InfiltrationState
from src.watershed import Watershed # To type hint the watershed_obj
from src.io.results_handler import ResultSaver # For type hinting result_saver
from src.core.physics_formulas import (get_h_from_trapezoid_area, get_plane_h_from_area,
                                       get_trapezoid_wp_from_h, get_trapezoid_topwidth_from_h,
                                       calculate_q_manning, calculate_dqda_manning_general)
# Rainfall generator if used directly by engine, or engine gets pre-generated series
from src.utils.rainfall_generator import generate_triangular_rainfall 


# --- Constants & Configuration ---
EPSILON = 1e-9

class SimulationEngine:

    def __init__(self,
                 watershed_obj: Watershed,
                 result_saver: ResultSaver,
                 simulation_settings: dict, # Contains durations, CFL, save_interval, etc.
                 device: torch.device,
                 dtype: torch.dtype):
        """
        Initializes the simulation engine.

        Args:
            watershed_obj (Watershed): Initialized Watershed object containing element modules and topology.
            result_saver (ResultSaver): Initialized ResultSaver object.
            simulation_settings (dict): Dictionary of simulation control parameters.
            device (torch.device): PyTorch device.
            dtype (torch.dtype): PyTorch dtype.
        """
        self.watershed = watershed_obj
        self.result_saver = result_saver
        self.sim_settings = simulation_settings
        self.device = device
        self.dtype = dtype

        # Unpack simulation settings for convenience
        self.sim_duration_seconds = self.sim_settings['sim_duration_min'] * 60.0
        self.save_interval_seconds = self.sim_settings['save_interval_min'] * 60.0
        self.max_dt_seconds = self.sim_settings['max_dt_min'] * 60.0
        self.min_dt_seconds = self.sim_settings['min_dt_min'] * 60.0
        self.cfl_number = self.sim_settings['cfl_number']
        
        # Mass balance trackers (as tensors on device)
        self.total_precip_volume_basin = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.total_infil_volume_basin = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.total_outlet_flow_volume_basin = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.initial_surface_storage_basin = self._calculate_total_surface_storage()
        self.initial_soil_moisture_storage_basin = self._calculate_total_soil_moisture_storage()
        print(f"  Initial surface storage: {self.initial_surface_storage_basin.item():.4f} m^3")
        print(f"  Initial soil moisture storage: {self.initial_soil_moisture_storage_basin.item():.4f} m^3")

    def _calculate_total_surface_storage(self) -> torch.Tensor:
        """Calculates total current surface water storage in all element modules."""
        total_storage = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        for eid, module in self.watershed.element_modules.items():
            props = module.props # ElementProperties stored in module
            if props.num_nodes > 0:
                if props.dx_segments.numel() > 0: # For elements with defined segments
                    segment_areas_avg = (module.area[:-1] + module.area[1:]) / 2.0
                    total_storage += torch.sum(segment_areas_avg * props.dx_segments)
                elif props.num_nodes == 1: # Single node element
                    total_storage += module.area[0] * props.LEN # A (tensor) * LEN (tensor)
        return total_storage

    def _calculate_total_soil_moisture_storage(self) -> torch.Tensor:
        """Calculates total current soil moisture storage in all element modules."""
        total_storage = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        for eid, module in self.watershed.element_modules.items():
            props = module.props
            soil_p = module.soil_params
            # Approx element surface area for infiltration = LEN * WID
            # Both LEN and WID should be 0-dim tensors in props
            element_surface_area = props.LEN * props.WID
            total_storage += module.theta_current * soil_p.effective_depth * element_surface_area
        return total_storage

    def _calculate_cfl_dt_for_element(self, element_module: torch.nn.Module) -> float:
        """Calculates stable dt for a single element module based on its current state."""
        props = element_module.props
        if props.num_nodes < 2: return float('inf') # CFL not well-defined

        A_state = element_module.area # Current area from buffer
         
        # if props.dx_segments.numel() > 0: 
        #     min_dx_val = torch.min(props.dx_segments) # .item()  
        # else: 
        #     min_dx_val = props.dx_avg # .item()

        # if min_dx_val <= EPSILON: return float('inf')

        # Calculate Q based on current A, then dQ/dA
        if props.geom_type == 'channel':
            min_dx_val  = 10 ##Tao: The minimum dx is preprocessed.
            # W0_nodes is a tensor, SS1/SS2 are 0-dim tensors
            h_state = get_h_from_trapezoid_area(A_state, props.W0_nodes, props.SS1, props.SS2)
            wp_state = get_trapezoid_wp_from_h(h_state, props.W0_nodes, props.SS1, props.SS2)
            topwidth_state = get_trapezoid_topwidth_from_h(h_state, props.W0_nodes, props.SS1, props.SS2)
        else: # Plane
            min_dx_val  = 4 ##Tao: The minimum dx is preprocessed.
            h_state = get_plane_h_from_area(A_state, props.WID) # WID is tensor
            wp_state = torch.full_like(A_state, props.WID.item()) # full_like needs float fill_value
            topwidth_state = None # Not strictly needed for plane dQdA if using simplified version
        
        Q_state = calculate_q_manning(A_state, wp_state, props.MAN, props.SL)
        
        dqda = calculate_dqda_manning_general(A_state, wp_state, topwidth_state, Q_state, 
                                              props.SS1, props.SS2, props.geom_type)
        
        max_celerity = torch.max(torch.abs(dqda)).item()
        if max_celerity < EPSILON: return float('inf')

        dt_stable = self.cfl_number * min_dx_val / max_celerity
        return dt_stable

    def run(self):
        """Executes the main simulation loop."""
        print(f"\n--- Starting Simulation Engine ({'GPU' if self.device.type=='cuda' else 'CPU'}) ---")

        # --- Prepare Rainfall Time Series ---
        # (Assuming generate_triangular_rainfall is available and adapted)
        sim_settings_rain = self.sim_settings # Alias for easier access
        # Ensure high_res_dt_min is positive for rainfall generation
        min_dt_for_rain_gen = self.min_dt_seconds / 60.0 if self.min_dt_seconds > 0 else 0.01
        high_res_dt_min = max(0.001, min(0.1, min_dt_for_rain_gen / 5.0))

        rain_times_s_for_interp, rain_rates_ms_for_interp, _ = generate_triangular_rainfall(
            sim_settings_rain['sim_duration_min'], sim_settings_rain['rain_event_dur_min'],
            sim_settings_rain['rain_peak_t_min'], sim_settings_rain['rain_peak_mmhr'],
            dt_min_for_gen=high_res_dt_min,
            save_interval_min=sim_settings_rain['save_interval_min'] # This was for mmhr_save, might not be needed for interp
        )
        # Move to device and ensure correct dtype
        rain_times_s_for_interp = rain_times_s_for_interp.to(device=self.device, dtype=self.dtype)
        rain_rates_ms_for_interp = rain_rates_ms_for_interp.to(device=self.device, dtype=self.dtype)


        # --- Simulation Loop Variables ---
        current_time_seconds_float = 0.0 # Python float for loop control
        # Cumulative rain for infiltration, keep as tensor
        cumulative_rain_m_tensor = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        
        step_count = 0
        save_count_idx = 0 # Index for HDF5 saving (0 for t=0, 1 for first save, etc.)
        next_save_time_float = 0.0 # Save at t=0

        # Save initial state
        if self.save_interval_seconds >= 0: # Allow disabling save by setting interval < 0
             # Collect current states from modules for saving
            current_basin_states_for_saving = {
                eid: {'flow': OverlandFlowState(mod.t_elapsed, mod.area, mod.depth, mod.discharge),
                      'infil': InfiltrationState(mod.theta_current, mod.F_cumulative)}
                for eid, mod in self.watershed.element_modules.items()
            }
            self.result_saver.save_state(
                current_save_trigger_idx=save_count_idx,
                time_seconds=current_time_seconds_float,
                basin_element_states_gpu=current_basin_states_for_saving,
                step_element_infil_rates_ms_float={} # No infil rates at t=0
            )
            save_count_idx += 1
            next_save_time_float = self.save_interval_seconds


        pbar = tqdm(total=int(self.sim_duration_seconds), desc="SimEngine", unit="sim_s")
        
        while current_time_seconds_float < self.sim_duration_seconds - EPSILON:
            # 1. Determine dt_step (adaptive timestep)
            min_dt_stable_elements = float('inf')
            if step_count > 0: # Don't calculate CFL at t=0 if states are all zero
                for element_module in self.watershed.element_modules.values():
                    dt_elem = self._calculate_cfl_dt_for_element(element_module) #.item()
                    min_dt_stable_elements = min(min_dt_stable_elements, dt_elem) # .item()
            
            dt_step_float = min(min_dt_stable_elements, self.max_dt_seconds) if min_dt_stable_elements != float('inf') else self.max_dt_seconds
            dt_step_float = max(dt_step_float, self.min_dt_seconds)
            
            if current_time_seconds_float + dt_step_float > self.sim_duration_seconds:
                dt_step_float = self.sim_duration_seconds - current_time_seconds_float
            
            if dt_step_float <= EPSILON: break
            
            dt_s_tensor = torch.tensor(dt_step_float, device=self.device, dtype=self.dtype)

            # 2. Get Current Rainfall Rate (interpolated scalar tensor)
            current_t_for_rain_lookup = current_time_seconds_float # Or current_time + dt_step_float / 2.0
            
            # Find index for interpolation (can be optimized if rain_times is regular)
            # searchsorted needs CPU numpy array
            rain_idx = np.searchsorted(rain_times_s_for_interp.cpu().numpy(), current_t_for_rain_lookup, side='right') - 1
            rain_idx = max(0, min(rain_idx, len(rain_rates_ms_for_interp) - 2)) # Ensure idx and idx+1 valid

            t0_rain = rain_times_s_for_interp[rain_idx] # Tensor
            R0_rain = rain_rates_ms_for_interp[rain_idx] # Tensor
            t1_rain = rain_times_s_for_interp[rain_idx+1] # Tensor
            R1_rain = rain_rates_ms_for_interp[rain_idx+1] # Tensor

            if t1_rain > t0_rain + EPSILON: # Avoid division by zero
                interp_factor = (current_t_for_rain_lookup - t0_rain) / (t1_rain - t0_rain)
                current_rain_rate_tensor = R0_rain + (R1_rain - R0_rain) * interp_factor
            else:
                current_rain_rate_tensor = R0_rain
            current_rain_rate_tensor = torch.clamp(current_rain_rate_tensor, min=0.0)

            # --- Update Mass Balance: Precipitation Input ---
            for eid_mb, module_mb in self.watershed.element_modules.items():
                props_mb = module_mb.props
                # LEN and WID are 0-dim tensors in props
                self.total_precip_volume_basin += current_rain_rate_tensor * props_mb.WID * props_mb.LEN * dt_s_tensor

            # --- Simulation Step for Each Group ---
            cumulative_rain_m_at_step_start = cumulative_rain_m_tensor.clone() # Pass clone
            cumulative_rain_m_tensor += current_rain_rate_tensor * dt_s_tensor # Update main tracker

            # Store tensor outflows from channels for routing between groups
            current_step_channel_outflows_tensor_map: dict[int, torch.Tensor] = {} 
            current_step_element_infil_rates_float_map: dict[int, float] = {} # For saving

            for group_id in self.watershed.simulation_order:
                group_config = self.watershed.group_elements_map.get(group_id)
                if not group_config: continue

                # --- Determine Upstream Inflow for this Group's Channel (as Tensor) ---
                upstream_q_for_channel_tensor = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                for up_grp_id, down_grp_id in self.watershed.connectivity.items():
                    if down_grp_id == group_id:
                        # Get tensor outflow from map
                        upstream_q_for_channel_tensor += current_step_channel_outflows_tensor_map.get(
                            up_grp_id, torch.tensor(0.0, device=self.device, dtype=self.dtype)
                        )
                
                # --- Process Planes in the group ---
                plane_outflows_to_channel_lateral_tensor = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                for plane_elem_id in group_config['planes']:
                    plane_module = self.watershed.element_modules[plane_elem_id]
                    
                    p_outflow_q, p_infil_r, p_infil_d = plane_module(
                        current_rain_rate_ms_tensor=current_rain_rate_tensor,
                        dt_s_tensor=dt_s_tensor,
                        cumulative_rain_m_start_tensor=cumulative_rain_m_at_step_start
                    )
                    current_step_element_infil_rates_float_map[plane_elem_id] = p_infil_r.item()
                    self.total_infil_volume_basin += p_infil_d * plane_module.props.WID * plane_module.props.LEN

                    if plane_module.props.side == 'head':
                        upstream_q_for_channel_tensor += p_outflow_q # Add tensor to tensor
                    else:
                        plane_outflows_to_channel_lateral_tensor += p_outflow_q
                
                # --- Process Channel in the group ---
                channel_elem_id = group_config['channel']
                if channel_elem_id is not None:
                    channel_module = self.watershed.element_modules[channel_elem_id]
                    
                    c_outflow_q, c_infil_r, c_infil_d = channel_module(
                        current_rain_rate_ms_tensor=current_rain_rate_tensor,
                        upstream_q_total_tensor=upstream_q_for_channel_tensor, # Pass tensor
                        plane_lateral_q_total_tensor=plane_outflows_to_channel_lateral_tensor, # Pass tensor
                        dt_s_tensor=dt_s_tensor,
                        cumulative_rain_m_start_tensor=cumulative_rain_m_at_step_start
                    )
                    current_step_channel_outflows_tensor_map[group_id] = c_outflow_q # Store tensor
                    current_step_element_infil_rates_float_map[channel_elem_id] = c_infil_r.item()
                    self.total_infil_volume_basin += c_infil_d * channel_module.props.WID * channel_module.props.LEN # Approx area

                    # Update basin outlet flow
                    if group_id not in self.watershed.connectivity: # Simple outlet check
                        self.total_outlet_flow_volume_basin += c_outflow_q * dt_s_tensor
            
            # --- Update Time and Save ---
            current_time_seconds_float += dt_step_float # Use float dt for loop control
            step_count += 1
            pbar.update(dt_step_float)

            if self.save_interval_seconds >= 0 and current_time_seconds_float >= next_save_time_float - EPSILON :
                current_basin_states_for_saving = {
                    eid: {'flow': OverlandFlowState(mod.t_elapsed, mod.area, mod.depth, mod.discharge),
                          'infil': InfiltrationState(mod.theta_current, mod.F_cumulative)}
                    for eid, mod in self.watershed.element_modules.items()
                }
                self.result_saver.save_state(
                    current_save_trigger_idx=save_count_idx,
                    time_seconds=current_time_seconds_float,
                    basin_element_states_gpu=current_basin_states_for_saving,
                    step_element_infil_rates_ms_float=current_step_element_infil_rates_float_map
                )
                save_count_idx += 1
                next_save_time_float += self.save_interval_seconds
        
        pbar.close()
        print(f"--- Simulation Engine Finished ({step_count} steps) ---")

        # --- Final Save and Mass Balance Report ---
        # Save last state if not perfectly aligned and save interval is active
        if self.save_interval_seconds >= 0 and \
           abs(current_time_seconds_float - (next_save_time_float - self.save_interval_seconds)) > EPSILON * self.save_interval_seconds and \
           current_time_seconds_float < self.sim_duration_seconds: # Avoid re-saving if already saved at duration
            print(f"Buffering final state at t={current_time_seconds_float:.2f}s (Save index {save_count_idx})")
            current_basin_states_for_saving = {
                eid: {'flow': OverlandFlowState(mod.t_elapsed, mod.area, mod.depth, mod.discharge),
                      'infil': InfiltrationState(mod.theta_current, mod.F_cumulative)}
                for eid, mod in self.watershed.element_modules.items()
            }
            self.result_saver.save_state(
                current_save_trigger_idx=save_count_idx,
                time_seconds=current_time_seconds_float,
                basin_element_states_gpu=current_basin_states_for_saving,
                step_element_infil_rates_ms_float=current_step_element_infil_rates_float_map
            )
        
        self.result_saver.finalize() # Flushes buffer and closes HDF5

        # self.report_mass_balance()

    def report_mass_balance(self):
        """Calculates and prints the final mass balance summary."""
        final_surface_storage_basin = self._calculate_total_surface_storage()
        final_soil_moisture_storage_basin = self._calculate_total_soil_moisture_storage()

        delta_surface_storage = final_surface_storage_basin - self.initial_surface_storage_basin
        delta_soil_storage = final_soil_moisture_storage_basin - self.initial_soil_moisture_storage_basin
        
        # All terms are now tensors, convert to item for printing
        total_outflow_plus_storage_change = (self.total_outlet_flow_volume_basin + 
                                             self.total_infil_volume_basin + 
                                             delta_surface_storage + 
                                             delta_soil_storage)
        
        mass_balance_error_abs = self.total_precip_volume_basin - total_outflow_plus_storage_change
        mass_balance_error_rel = (mass_balance_error_abs / (self.total_precip_volume_basin + EPSILON)) * 100.0

        print("\n--- Mass Balance Summary (all Tensors converted to item for report) ---")
        print(f"  Total Precipitation Volume: {self.total_precip_volume_basin.item():.4f} m^3")
        print(f"  Total Infiltration Volume:  {self.total_infil_volume_basin.item():.4f} m^3")
        print(f"  Total Basin Outlet Flow:    {self.total_outlet_flow_volume_basin.item():.4f} m^3")
        print(f"  Change in Surface Storage:  {delta_surface_storage.item():.4f} m^3")
        print(f"  Change in Soil Storage:     {delta_soil_storage.item():.4f} m^3")
        print(f"  Sum of Outputs & Storage Change: {total_outflow_plus_storage_change.item():.4f} m^3")
        print(f"  Mass Balance Error (Input - Output): {mass_balance_error_abs.item():.4f} m^3")
        print(f"  Relative Error: {mass_balance_error_rel.item():.4f} %")

