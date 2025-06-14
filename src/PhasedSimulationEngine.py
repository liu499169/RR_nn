# src/phased_engine.py 

import torch
import gc
from multiprocessing import Queue
import torch.multiprocessing as mp 
import numpy as np
from tqdm import tqdm
import h5py
from collections import defaultdict

from src.data_structures import ElementProperties, OverlandFlowState, InfiltrationState
from src.watershed import Watershed
from src.io.results_handler import ResultSaver
from src.core.physics_formulas import (get_plane_h_from_area, calculate_q_manning, 
                                       calculate_dqda_manning_general, 
                                       get_h_from_trapezoid_area, get_trapezoid_wp_from_h,
                                       get_trapezoid_topwidth_from_h) # etc.
from src.utils.rainfall_generator import generate_triangular_rainfall
from src.components.plane_element import PlaneElement # For isinstance check
from src.components.channel_element import ChannelElement # For isinstance check


EPSILON = 1e-9

# In a new file src/simulation_phases.py (or at the top of PhasedSimulationEngine.py)

import torch
from tqdm import tqdm
from src.data_structures import OverlandFlowState, InfiltrationState

EPSILON = 1e-9 # Make sure to define or import it

class PhasedSimulationEngine:
    def __init__(self,
                 watershed_obj: Watershed,
                 result_saver_planes: ResultSaver, 
                 result_saver_channels: ResultSaver, 
                 simulation_settings: dict,
                 device: torch.device,
                 dtype: torch.dtype,
                 phase1_results: dict = None):
        
        self.watershed = watershed_obj
        self.result_saver_planes = result_saver_planes
        self.result_saver_channels = result_saver_channels
        self.sim_settings = simulation_settings
        self.device = device
        self.dtype = dtype
        self.phase1_results = phase1_results

        # Unpack common settings
        self.sim_duration_s = self.sim_settings['sim_duration_min'] * 60.0
        self.save_interval_seconds = self.sim_settings['save_interval_min'] * 60.0
        self.max_dt_s = self.sim_settings['max_dt_min'] * 60.0
        self.min_dt_s = self.sim_settings['min_dt_min'] * 60.0
        self.cfl_number = self.sim_settings['cfl_number']

        # Mass balance trackers
        self.total_precip_volume_basin = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.total_outlet_flow_volume_basin = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        if self.phase1_results:
            # We are in Phase 2, initialize with results from Phase 1
            print("Phase 2 Engine initialized with mass balance state from Phase 1.")
            self.total_precip_volume_basin += self.phase1_results['precip_volume']
            self.initial_surface_storage_basin = self.phase1_results['initial_surface_storage']
            self.initial_soil_moisture_storage_basin = self.phase1_results['initial_soil_moisture_storage']
        else:
            # We are in Phase 1, calculate initial state from scratch
            print("Phase 1 Engine: Calculating initial basin storage.")
            all_modules = list(self.watershed.element_modules.values())
            self.initial_surface_storage_basin = self._calculate_overall_surface_storage(all_modules, self.device, self.dtype)
            self.initial_soil_moisture_storage_basin = self._calculate_overall_soil_moisture_storage(all_modules, self.device, self.dtype)
        
        print(f"Initial surface storage (m^3): {self.initial_surface_storage_basin.item():.2f}")
        print(f"Initial soil moisture storage (m^3): {self.initial_soil_moisture_storage_basin.item():.2f}")

    @staticmethod
    def _calculate_total_infiltration_volume(module_list: list, device, dtype) -> tuple:
        """Calculates total infiltration volume for a given list of modules."""
        total_infiltration_volume_planes = torch.tensor(0.0, device=device, dtype=dtype)
        total_infiltration_volume_channels = torch.tensor(0.0, device=device, dtype=dtype)
        for module in module_list:
            props = module.props
            if props.geom_type == 'plane':
                total_infiltration_volume_planes += module.F_cumulative * props.LEN * props.WID
            elif props.geom_type == 'channel':
                total_infiltration_volume_channels += module.F_cumulative * props.WID * props.LEN
        total_infiltration_volume = total_infiltration_volume_planes + total_infiltration_volume_channels
        return total_infiltration_volume_planes, total_infiltration_volume_channels, total_infiltration_volume

    @staticmethod
    def _calculate_overall_surface_storage(module_list: list, device, dtype) -> torch.Tensor:
        """Calculates total surface water storage for a given list of modules."""
        total_storage = torch.tensor(0.0, device=device, dtype=dtype)
        for module in module_list:
            props = module.props 
            if props.num_nodes > 0:
                if props.dx_segments.numel() > 0:
                    flow_area = module.area[:-1]
                    dx_tensor = props.dx_avg * torch.ones_like(flow_area, device=device, dtype=dtype)
                    total_storage += torch.sum(flow_area * dx_tensor)
        return total_storage
    
    @staticmethod
    def _calculate_overall_soil_moisture_storage(module_list: list, device, dtype) -> torch.Tensor:
        """Calculates total soil moisture storage for a given list of modules."""
        drying_storage = torch.tensor(0.0, device=device, dtype=dtype)
        total_storage = torch.tensor(0.0, device=device, dtype=dtype)
        for module in module_list:
            props = module.props
            soil_p = module.soil_params
            element_surface_area = props.LEN * props.WID
            drying_storage += module.drying_cumulative * soil_p.effective_depth * element_surface_area 
            total_storage += module.theta_current.mean() * soil_p.effective_depth * element_surface_area 
        return total_storage - drying_storage

    def _calculate_dt_for_module_list(self, modules_to_check: list[torch.nn.Module], phase:int) -> float:
        """Calculates a common dt for a list of modules based on min CFL."""
        min_dt_overall = float('inf')
        if not modules_to_check: return self.max_dt_s 
            
        active_modules_for_cfl = [m for m in modules_to_check if m.props.num_nodes >=2]
        if not active_modules_for_cfl : return self.max_dt_s

        for module in active_modules_for_cfl:
            props = module.props
            A_state = module.area
            # min_dx_val = torch.min(props.dx_segments).item() if props.dx_segments.numel() > 0 else props.dx_avg.item()
            # if min_dx_val <= EPSILON: dt_elem = self.max_dt_s; continue

            if phase == 1: # Phase 1: Planes
                min_dx_val = 4 # Tao: the mini_dx is preprocessed in the watershed loader
                h_state = get_plane_h_from_area(A_state, props.WID)
                wp_state = props.WID.expand_as(A_state)
                topwidth_state = props.WID.expand_as(A_state)
            
            elif phase == 2: # Phase 2: Channels
                min_dx_val = 10 # Tao: the mini_dx is preprocessed in the watershed loader
                h_state = get_h_from_trapezoid_area(A_state, props.W0_nodes, props.SS1, props.SS2)
                wp_state = get_trapezoid_wp_from_h(h_state, props.W0_nodes, props.SS1, props.SS2)
                topwidth_state = get_trapezoid_topwidth_from_h(h_state, props.W0_nodes, props.SS1, props.SS2)

            else: # unknown phase 
                raise ValueError(f"Unknown phase {phase} for dt calculation")
            
            Q_state = calculate_q_manning(A_state, wp_state, props.MAN, props.SL)
            dqda = calculate_dqda_manning_general(A_state, wp_state, topwidth_state, Q_state, 
                                                  props.SS1, props.SS2, props.geom_type)
            max_celerity = torch.max(torch.abs(dqda)).item()
            if max_celerity < EPSILON: dt_elem = self.max_dt_s
            else: dt_elem = self.cfl_number * min_dx_val / max_celerity
            min_dt_overall = min(min_dt_overall, dt_elem)
        
        dt_chosen = min(min_dt_overall, self.max_dt_s) if min_dt_overall != float('inf') else self.max_dt_s
        dt_chosen = max(dt_chosen, self.min_dt_s)
        return dt_chosen

    def _get_interpolated_rainfall(self, current_time_s: float, 
                                  rain_times_s_cpu: torch.Tensor, 
                                  rain_rates_ms_device: torch.Tensor) -> torch.Tensor:
        rain_idx = np.searchsorted(rain_times_s_cpu.numpy(), current_time_s, side='right') - 1
        rain_idx = max(0, min(rain_idx, len(rain_rates_ms_device) - 2))
        t0_r, t1_r = rain_times_s_cpu[rain_idx].item(), rain_times_s_cpu[rain_idx+1].item()
        R0_r, R1_r = rain_rates_ms_device[rain_idx], rain_rates_ms_device[rain_idx+1]
        rain_rate = R0_r + (R1_r - R0_r) * ((current_time_s - t0_r) / (t1_r - t0_r + EPSILON)) \
                           if (t1_r > t0_r + EPSILON) else R0_r
        return torch.clamp(rain_rate, min=0.0)

    def _interpolate_hydrograph(self, target_time_s: float, 
                                source_times_s: torch.Tensor, # 1D tensor of times for source hydrograph
                                source_q_m3s: torch.Tensor   # 1D tensor of Q values for source hydrograph
                                ) -> torch.Tensor:
        """Linearly interpolates Q from source hydrograph to target_time_s."""
        if source_times_s.numel() == 0 or source_q_m3s.numel() == 0:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)
        if source_times_s.numel() == 1: # Only one point in source hydrograph
            return source_q_m3s[0]

        # Find insertion point for target_time_s in source_times_s
        # np.interp is simpler if tensors are on CPU. For GPU tensors:
        # Move to CPU for np.interp, or implement GPU linear interpolation
        # For now, using np.interp for simplicity, assuming source_times/q can be moved to CPU
        # This is a potential bottleneck if called very frequently for many hydrographs
        # A GPU-based interpolation would be better for performance.
        q_interp_np = np.interp(target_time_s, 
                                source_times_s.cpu().numpy(), 
                                source_q_m3s.cpu().numpy(),
                                left=source_q_m3s[0].item(), # Value if target_time is before first source_time
                                right=source_q_m3s[-1].item()) # Value if target_time is after last source_time
        return torch.tensor(q_interp_np, device=self.device, dtype=self.dtype)

    def _run_phase1_planes(self, plane_modules: list, rainfall_times_s_cpu: torch.Tensor, 
                           rainfall_rates_ms_device: torch.Tensor,
                           output_save_interval_s: float):
        print("--- Starting Phase 1: Independent Plane Simulations ---")
        # plane_modules = [m for m in self.watershed.element_modules.values() if isinstance(m, PlaneElement)]
        if not plane_modules:
            print("No plane elements found for Phase 1.")
            if self.result_saver_planes: self.result_saver_planes.finalize() # Finalize if created
            return
        
        # --- Pre-allocate storage for hydrographs --
        max_hydrograph_points = int(self.sim_duration_s / output_save_interval_s) + 2 # +2 for t=0 and final state
        num_planes = len(plane_modules)
        plane_ids = [p.props.element_id for p in plane_modules]
        self.plane_id_to_idx_map_phase1 = {pid: i for i, pid in enumerate(plane_ids)}

        prealloc_times = torch.full((num_planes, max_hydrograph_points), -1.0, device=self.device, dtype=self.dtype)
        prealloc_q = torch.zeros((num_planes, max_hydrograph_points), device=self.device, dtype=self.dtype)

        points_saved_count = torch.zeros(num_planes, dtype=torch.long, device=self.device)

        # --- Store initial state (t=0) ---
        prealloc_times[:, 0] = 0.0
        for p in plane_modules:
            idx = self.plane_id_to_idx_map_phase1[p.props.element_id]
            if p.props.num_nodes > 0:
                prealloc_q[idx, 0] = p.discharge[-1].clone().detach()
        points_saved_count += 1

        current_time_s_float = 0.0
        cumulative_rain_m_tensor = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        # For HDF5 saving during Phase 1
        save_count_idx_planes = 0
        next_hdf5_save_time_s = 0.0 # Save t=0 for HDF5

        # For inter-phase hydrograph logging
        next_output_log_time_s = 0.0 # Also log t=0 for hydrograph

        if output_save_interval_s <= 0: # If not logging for inter-phase, set to infinity
            next_output_log_time_s = float('inf')
        else:
            next_output_log_time_s = output_save_interval_s

        # step_count = 0
        pbar = tqdm(total=int(self.sim_duration_s), desc="Phase 1 (Planes)", unit="sim_s")
        while current_time_s_float < self.sim_duration_s - EPSILON:
            dt_float = self._calculate_dt_for_module_list(plane_modules, phase=1) # CFL for planes
            if current_time_s_float + dt_float > self.sim_duration_s:
                dt_float = self.sim_duration_s - current_time_s_float
            if dt_float <= EPSILON: break
            dt_tensor = torch.tensor(dt_float, device=self.device, dtype=self.dtype)

            current_rain = self._get_interpolated_rainfall(current_time_s_float, rainfall_times_s_cpu, rainfall_rates_ms_device)
            cum_rain_start_step = cumulative_rain_m_tensor.clone()
            cumulative_rain_m_tensor += current_rain * dt_tensor

            current_step_plane_infil_rates_float_map = {} # For HDF5 saving

            total_precip_volume_basin_phase1 = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            for plane_module in plane_modules:
                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                    _, p_infil_r, infil_d = plane_module(current_rain, dt_tensor, cum_rain_start_step)
                # self.total_infil_volume_basin_planes += infil_d * plane_module.props.WID * plane_module.props.LEN
                pid = plane_module.props.element_id
                current_step_plane_infil_rates_float_map[pid] = p_infil_r.item()
                # Precip for planes is accounted for globally for the basin in the main run method

                # Global precip for this step (for mass balance)
                props_mb = plane_module.props
                total_precip_volume_basin_phase1 += current_rain * props_mb.WID * props_mb.LEN * dt_tensor
            
            current_time_s_float += dt_float
            # step_count +=1
            pbar.update(dt_float)
            
            # Log for inter-phase hydrograph
            if output_save_interval_s > 0 and current_time_s_float >= next_output_log_time_s - EPSILON:
                # time_now_tensor = torch.tensor(current_time_s_float, device=self.device, dtype=self.dtype)
                # for plane_module in plane_modules:
                #     pid = plane_module.props.element_id
                #     if plane_module.props.num_nodes > 0:
                #         q_out = plane_module.discharge[-1].clone().detach()
                #     else:
                #         q_out = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                #     plane_outflow_times_lists[pid].append(time_now_tensor)
                #     plane_outflow_q_lists[pid].append(q_out)
                # next_output_log_time_s += output_save_interval_s

                save_idx = points_saved_count[0].item() # All planes have same number of points
                if save_idx < max_hydrograph_points:
                    prealloc_times[:, save_idx] = current_time_s_float
                    for p in plane_modules:
                        plane_idx = self.plane_id_to_idx_map_phase1[p.props.element_id]
                        if p.props.num_nodes > 0:
                            prealloc_q[plane_idx, save_idx] = p.discharge[-1] # No need for clone().detach() if writing to a slice
                    points_saved_count += 1
                next_output_log_time_s += output_save_interval_s

            # Save to HDF5 if needed
            if self.result_saver_planes and current_time_s_float >= next_hdf5_save_time_s - EPSILON:
                plane_states_to_save = {
                    p.props.element_id: {
                        'flow': OverlandFlowState(p.t_elapsed.clone(), p.area.clone(), p.depth.clone(), 
                                                  p.discharge.clone(), p.max_cfl.clone()),
                        'infil': InfiltrationState(p.theta_current.clone(), p.F_cumulative.clone(),p.drying_cumulative.clone())
                    } for p in plane_modules
                }
                self.result_saver_planes.save_state(save_count_idx_planes, current_time_s_float, plane_states_to_save, current_step_plane_infil_rates_float_map)
                save_count_idx_planes += 1
                next_hdf5_save_time_s += self.save_interval_seconds
        pbar.close()

        # --- AT THE END OF THE FUNCTION ---
        # 1. Prepare hydrographs to return
        plane_hydrographs_to_return = {}
        plane_ids = [p.props.element_id for p in plane_modules]
        max_pts = points_saved_count.max().item()
        for i, pid in enumerate(plane_ids):
            valid_points = points_saved_count[i].item()
            final_times = prealloc_times[i, :valid_points].clone()
            final_q = prealloc_q[i, :valid_points].clone()
            plane_hydrographs_to_return[pid] = (final_times, final_q)

        # 2. Prepare mass balance results to return
        print("Calculating final mass balance state for Phase 1...")
        final_infil_p, _, _ = self._calculate_total_infiltration_volume(plane_modules, self.device, self.dtype)
        
        mass_balance_results = {
            'precip_volume': total_precip_volume_basin_phase1,
            'initial_surface_storage': self.initial_surface_storage_basin,
            'initial_soil_moisture_storage': self.initial_soil_moisture_storage_basin,
            'final_surface_storage_planes': self._calculate_overall_surface_storage(plane_modules, self.device, self.dtype),
            'final_soil_moisture_storage_planes': self._calculate_overall_soil_moisture_storage(plane_modules, self.device, self.dtype),
            'final_infiltration_volume_planes': final_infil_p,
        }

        # Clean up large allocated tensors
        del prealloc_times, prealloc_q, points_saved_count

        print(f"--- Phase 1 Finished, returning collected hydrographs and mass balance state. ---")
        return plane_hydrographs_to_return, mass_balance_results 
    
    def _save_hydrographs_to_file(self, hydrograph_dict: dict, hdf5_path: str):
        """
        Opens an HDF5 file in append mode and saves full hydrographs.
        This is a one-off operation after Phase 1.
        """
        print(f"Opening {hdf5_path} to append full hydrographs...")
        try:
            with h5py.File(hdf5_path, 'a') as hf: # 'a' for append mode
                hydro_group = hf.require_group("hydrographs")
                
                for element_id, (times, q_values) in hydrograph_dict.items():
                    elem_group = hydro_group.require_group(str(element_id))
                    
                    # Move data to CPU for h5py
                    times_cpu = times.cpu().numpy()
                    q_cpu = q_values.detach().cpu().numpy()

                    # Overwrite if exists
                    if "time_s" in elem_group: del elem_group["time_s"]
                    if "discharge_m3s" in elem_group: del elem_group["discharge_m3s"]
                    
                    elem_group.create_dataset("time_s", data=times_cpu, compression="gzip")
                    elem_group.create_dataset("discharge_m3s", data=q_cpu, compression="gzip")
            
            print(f"Successfully appended hydrographs for {len(hydrograph_dict)} elements.")
        except Exception as e:
            print(f"!!! ERROR saving hydrographs to HDF5: {e}")
            raise

    def load_and_resample_plane_hydrographs(self, hdf5_path: str):
        """Loads plane hydrographs from an HDF5 file and resamples them."""
        
        plane_hydrographs_from_file = {}
        print(f"Loading full hydrographs from {hdf5_path}...")
        with h5py.File(hdf5_path, 'r') as hf:
            if "hydrographs" not in hf:
                raise ValueError(f"Could not find 'hydrographs' group in HDF5 file: {hdf5_path}")
            
            hydro_group = hf["hydrographs"]
            for elem_id_str in tqdm(hydro_group.keys(), desc="Loading Hydrographs"):
                elem_id = int(elem_id_str)
                times = torch.from_numpy(hydro_group[elem_id_str]["time_s"][:]).to(device=self.device, dtype=self.dtype)
                q_values = torch.from_numpy(hydro_group[elem_id_str]["discharge_m3s"][:]).to(device=self.device, dtype=self.dtype)
                plane_hydrographs_from_file[elem_id] = (times, q_values)

        # Call the resampling logic on the loaded data
        self.plane_outflow_hydrographs = plane_hydrographs_from_file
        self._resample_plane_hydrographs() # This will create self.resampled_plane_q

    def _resample_plane_hydrographs(self):
            """
            Performs a one-time resampling of all plane hydrographs to a common,
            uniform time grid after Phase 1 is complete. This avoids repeated
            interpolation during the Phase 2 loop.
            """
            print("\n--- Pre-resampling plane hydrographs for Phase 2 ---")
            if not self.plane_outflow_hydrographs:
                return

            # 1. Define the common, high-resolution time grid on the GPU
            # Use a resolution at least as fine as your min_dt_s for accuracy.
            # For a 12-min (720s) event, 1-second resolution is a good start.
            num_points = int(self.sim_duration_s) + 1 # e.g., 721 points for 0 to 720s
            self.resampled_time_grid = torch.linspace(0, self.sim_duration_s, num_points, device=self.device, dtype=self.dtype)

            # 2. Group all plane hydrographs for batch interpolation
            plane_ids = list(self.plane_outflow_hydrographs.keys())
            # We will create one large tensor: (num_planes, num_resampled_points)
            resampled_q_tensor = torch.zeros(len(plane_ids), num_points, device=self.device, dtype=self.dtype)

            pbar = tqdm(total=len(plane_ids), desc="Resampling", unit="plane")
            for i, pid in enumerate(plane_ids):
                source_t, source_q = self.plane_outflow_hydrographs[pid]
                if source_t.numel() < 2:
                    continue # This plane had no significant flow, leave as zeros

                # 3. Perform efficient, BATCHED GPU-based interpolation
                # torch.searchsorted finds the indices, then we do linear interp
                indices = torch.searchsorted(source_t, self.resampled_time_grid, right=True) - 1
                indices = torch.clamp(indices, 0, source_t.numel() - 2)

                t0, t1 = source_t[indices], source_t[indices + 1]
                q0, q1 = source_q[indices], source_q[indices + 1]

                # Handle division by zero for segments with same timestamp
                dt_segment = t1 - t0
                interp_factor = torch.where(dt_segment > EPSILON, (self.resampled_time_grid - t0) / dt_segment, 0)

                resampled_q_tensor[i, :] = q0 + interp_factor * (q1 - q0)
                pbar.update(1)
            pbar.close()

            # 4. Store the final resampled tensor and a map from plane_id to its row index
            self.resampled_plane_q = resampled_q_tensor
            self.plane_id_to_idx_map = {pid: i for i, pid in enumerate(plane_ids)}

            # 5. [IMPORTANT FOR VRAM] Clean up the original, ragged hydrographs
            print("Clearing original ragged hydrographs to save VRAM.")
            del self.plane_outflow_hydrographs
            self.plane_outflow_hydrographs = {}
            torch.cuda.empty_cache() # deleting large tensors

    def _run_phase2_channels(self, channel_modules: list, rainfall_times_s_cpu: torch.Tensor, 
                             rainfall_rates_ms_device: torch.Tensor):
        print("--- Starting Phase 2: Channel Network Routing ---")
        # channel_modules = [m for m in self.watershed.element_modules.values() if isinstance(m, ChannelElement)]
        if not channel_modules:
            print("No channel elements found for Phase 2.")
            self.result_saver.finalize() # Finalize saver if no channels to save
            return

        # Use the main HDF5 saver for Phase 2 outputs
        save_interval_s_phase2 = self.sim_settings['save_interval_min'] * 60.0

        current_time_s_float = 0.0
        cumulative_rain_m_tensor = torch.tensor(0.0, device=self.device, dtype=self.dtype) # Separate for channels
        
        # step_count = 0
        # ResultSaver handles t=0 save via its own __init__ or first call from SimulationEngine
        # Here, we need to align with the SimulationEngine's save_count_idx for HDF5
        save_count_idx_phase2 = 0 # Start from 0 for HDF5 indexing by ResultSaver
        if save_interval_s_phase2 >= 0 : # Initial save for t=0 for channels
            current_channel_states_to_save = { # Only channel states for this phase's primary output
                eid: {'flow': OverlandFlowState(mod.t_elapsed.clone(), mod.area.clone(), mod.depth.clone(), 
                                                mod.discharge.clone(), mod.max_cfl.clone()),
                      'infil': InfiltrationState(mod.theta_current.clone(), mod.F_cumulative.clone(), 
                                                 mod.drying_cumulative.clone())}
                for eid, mod in self.watershed.element_modules.items() if isinstance(mod, ChannelElement)
            }
           
            self.result_saver_channels.save_state(save_count_idx_phase2, 0.0, current_channel_states_to_save, {})
            save_count_idx_phase2 += 1
        
        next_save_time_float = save_interval_s_phase2 if save_interval_s_phase2 >= 0 else float('inf')


        pbar = tqdm(total=int(self.sim_duration_s), desc="Phase 2 (Channels)", unit="sim_s")
        while current_time_s_float < self.sim_duration_s - EPSILON:
            dt_float = self._calculate_dt_for_module_list(channel_modules, phase=2) # CFL for channels
            if current_time_s_float + dt_float > self.sim_duration_s:
                dt_float = self.sim_duration_s - current_time_s_float
            if dt_float <= EPSILON: break
            # print(f"Current dt: {dt_float:.2f}")
            dt_tensor = torch.tensor(dt_float, device=self.device, dtype=self.dtype)

            current_rain = self._get_interpolated_rainfall(current_time_s_float, rainfall_times_s_cpu, rainfall_rates_ms_device)
            
            cum_rain_start_step = cumulative_rain_m_tensor.clone()
            cumulative_rain_m_tensor += current_rain * dt_tensor
            
            channel_outflows_this_step_map: dict[int, torch.Tensor] = {}
            element_infil_rates_this_step_map: dict[int, float] = {}

            for group_id in self.watershed.simulation_order: # This order includes all groups
                group_config = self.watershed.get_group_config(group_id)
                if not group_config or not group_config['channel']: continue # Skip if no channel in group

                channel_eid = group_config['channel']
                channel_module = self.watershed.get_element_module(channel_eid)
                if not isinstance(channel_module, ChannelElement): continue # Should not happen

                # Global precip for this step (for mass balance)
                props_c = channel_module.props
                self.total_precip_volume_basin += current_rain * props_c.WID * props_c.LEN * dt_tensor

                # --- Upstream Channel Inflow ---
                upstream_q_tensor = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                for up_grp_id, down_grp_id in self.watershed.connectivity.items():
                    if down_grp_id == group_id:
                        upstream_q_tensor += channel_outflows_this_step_map.get(
                            up_grp_id, torch.tensor(0.0, device=self.device, dtype=self.dtype))
                
                # --- Plane Inflows (Interpolated from Phase 1 hydrographs) ---
                head_plane_q_tensor = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                side_plane_q_total_tensor = torch.tensor(0.0, device=self.device, dtype=self.dtype)

                #  Fast index lookup
                time_idx = min(int(current_time_s_float), self.resampled_plane_q.shape[1] - 1)

                plane_eids = group_config['planes']
                if plane_eids:
                    # Get the row indices for all planes in this group
                    plane_row_indices = [self.plane_id_to_idx_map[pid] for pid in plane_eids if pid in self.plane_id_to_idx_map]

                    if plane_row_indices:
                        # Get all Q values for this group's planes at the current time_idx with one lookup
                        current_q_for_group_planes = self.resampled_plane_q[plane_row_indices, time_idx]

                        # We iterate through the original plane_eids and the retrieved q_values together
                    q_idx = 0
                    for plane_eid in plane_eids:
                        if plane_eid in self.plane_id_to_idx_map:
                            plane_props = self.watershed.get_element_properties(plane_eid)
                            q_val = current_q_for_group_planes[q_idx] # Direct lookup, no searching
                            if plane_props.side == 'head':
                                head_plane_q_tensor += q_val
                            elif plane_props.side in ['left', 'right']:
                                side_plane_q_total_tensor += q_val
                            q_idx += 1

                # for plane_eid in group_config['planes']:
                #     plane_props = self.watershed.get_element_properties(plane_eid)
                #     plane_hydrograph_times, plane_hydrograph_q = self.plane_outflow_hydrographs.get(plane_eid, (None, None))
                    
                #     if plane_hydrograph_times is not None and plane_hydrograph_q is not None:
                #         q_interp = self._interpolate_hydrograph(current_time_s_float, plane_hydrograph_times, plane_hydrograph_q)
                #         if plane_props.side == 'head': head_plane_q_tensor += q_interp
                #         elif plane_props.side in ['left','right']: side_plane_q_total_tensor += q_interp
                #         else: print(f"Warning: Unknown plane side '{plane_props.side}' for element {plane_eid}")                

                upstream_q_final_for_channel = upstream_q_tensor + head_plane_q_tensor
                
                # --- Call Channel Forward ---
                with torch.amp.autocast(device_type=self.device.type,enabled=(self.device.type == 'cuda')):
                    c_out_q, c_infil_r, c_infil_d = channel_module(
                        current_rain, upstream_q_final_for_channel,
                        side_plane_q_total_tensor, dt_tensor, cum_rain_start_step)
                
                channel_outflows_this_step_map[group_id] = c_out_q
                element_infil_rates_this_step_map[channel_eid] = c_infil_r.item()
                props_c = channel_module.props
                # self.total_infil_volume_basin_channels += c_infil_d * props_c.W0_nodes.mean() * props_c.LEN

                if group_id not in self.watershed.connectivity:
                    self.total_outlet_flow_volume_basin += c_out_q * dt_tensor
            
            current_time_s_float += dt_float
            # step_count += 1
            pbar.update(dt_float)

            if save_interval_s_phase2 >=0 and current_time_s_float >= next_save_time_float - EPSILON:
                all_current_states_for_saving = {
                    eid: {'flow': OverlandFlowState(mod.t_elapsed.clone(), mod.area.clone(), mod.depth.clone(), 
                                                    mod.discharge.clone(), mod.max_cfl.clone()),
                          'infil': InfiltrationState(mod.theta_current.clone(), mod.F_cumulative.clone(), 
                                                     mod.drying_cumulative.clone())}
                    for eid, mod in self.watershed.element_modules.items() if isinstance(mod, ChannelElement)
                }
                self.result_saver_channels.save_state(save_count_idx_phase2, current_time_s_float,
                                             all_current_states_for_saving, element_infil_rates_this_step_map)
                save_count_idx_phase2 += 1
                next_save_time_float += save_interval_s_phase2
        pbar.close()
        print(f"--- Phase 2 Finished ---")  # ({step_count} steps)

    def report_mass_balance(self):
        """Calculates and prints the final mass balance by combining Phase 1
        results with the final state of the Phase 2 engine."""
        if not self.phase1_results:
            print("Warning: Reporting mass balance without Phase 1 results. Report will be incomplete.")
            return

        # Get final state of channels from this engine's watershed object
        print("Calculating final mass balance state for Phase 2...")
        channel_modules = [m for m in self.watershed.element_modules.values() if isinstance(m, ChannelElement)]
        final_surface_storage_channels = self._calculate_overall_surface_storage(channel_modules, self.device, self.dtype)
        final_soil_moisture_storage_channels = self._calculate_overall_soil_moisture_storage(channel_modules, self.device, self.dtype)
        _, final_infil_c, _ = self._calculate_total_infiltration_volume(channel_modules, self.device, self.dtype)
        
        # Combine with stored Phase 1 results
        final_surface_storage_basin_end = self.phase1_results['final_surface_storage_planes'] + final_surface_storage_channels
        final_soil_moisture_storage_basin_end = self.phase1_results['final_soil_moisture_storage_planes'] + final_soil_moisture_storage_channels
        
        print(f"Final surface storage (m^3): {final_surface_storage_basin_end.item():.2f}")
        print(f"Final soil moisture storage (m^3): {final_soil_moisture_storage_basin_end.item():.2f}")
        
        # Calculate deltas using initial values stored from Phase 1
        delta_surface = final_surface_storage_basin_end - self.initial_surface_storage_basin
        delta_soil = final_soil_moisture_storage_basin_end - self.initial_soil_moisture_storage_basin
        
        # Total precipitation volume is already accumulated correctly in self.total_precip_volume_basin
        # during the Phase 2 loop.
        
        # Correct mass balance: Precip = Outlet_Flow + d(Surface) + d(Soil)
        total_change_and_outflow = self.total_outlet_flow_volume_basin + delta_surface + delta_soil
        
        error_abs = self.total_precip_volume_basin - total_change_and_outflow
        error_rel = (error_abs / (self.total_precip_volume_basin + EPSILON)) * 100.0

        # --- For Reporting Only ---
        final_infil_volume_planes = self.phase1_results['final_infiltration_volume_planes']
        final_infil_volume_basin = final_infil_volume_planes + final_infil_c

        # --- Final Printout ---
        print("\n--- Overall Mass Balance Summary (Decoupled) ---")
        print(f"  Total Precipitation Volume: {self.total_precip_volume_basin.item():.4f} m^3")
        print(f"  Total Infiltration (Planes): {final_infil_volume_planes.item():.4f} m^3")
        print(f"  Total Infiltration (Channels): {final_infil_c.item():.4f} m^3")
        print(f"  Total Infiltration (Combined): {final_infil_volume_basin.item():.4f} m^3")
        print(f"  Total Basin Outlet Flow:    {self.total_outlet_flow_volume_basin.item():.4f} m^3")
        print(f"  Change in Surface Storage:  {delta_surface.item():.4f} m^3")
        print(f"  Change in Soil Storage:     {delta_soil.item():.4f} m^3")
        print(f"  Sum of Outputs & Storage Change: {total_change_and_outflow.item():.4f} m^3")
        print(f"  Mass Balance Error (In - Out): {error_abs.item():.4f} m^3 ({error_rel.item():.4f} %)")