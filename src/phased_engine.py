# src/phased_engine.py 

import torch
import numpy as np
from tqdm import tqdm
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

class PhasedSimulationEngine:
    def __init__(self,
                 watershed_obj: Watershed,
                 result_saver_planes: ResultSaver, 
                 result_saver_channels: ResultSaver, 
                 simulation_settings: dict,
                 device: torch.device,
                 dtype: torch.dtype):
        
        self.watershed = watershed_obj
        self.result_saver_planes = result_saver_planes
        self.result_saver_channels = result_saver_channels
        self.sim_settings = simulation_settings
        self.device = device
        self.dtype = dtype

        # Unpack common settings
        self.sim_duration_s = self.sim_settings['sim_duration_min'] * 60.0
        self.save_interval_seconds = self.sim_settings['save_interval_min'] * 60.0
        self.max_dt_s = self.sim_settings['max_dt_min'] * 60.0
        self.min_dt_s = self.sim_settings['min_dt_min'] * 60.0
        self.cfl_number = self.sim_settings['cfl_number']
        
        # For storing intermediate results between phases (GPU tensors)
        self.plane_outflow_hydrographs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # Tao: Tested that this is not a good approach --> add for headwater channels if Phase 1b is implemented: 
        # self.headwater_channel_hydrographs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        self.total_plane_area = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.total_channel_surface_area = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        for module in self.watershed.element_modules.values():
            props = module.props
            if props.geom_type == 'plane':
                self.total_plane_area += props.LEN * props.WID
            elif props.geom_type == 'channel':
                # Assuming props.WID for channel is its average top width for precip
                self.total_channel_surface_area += props.LEN * props.WID
        
        self.total_precip_receiving_area = self.total_plane_area + self.total_channel_surface_area
        print(f"  Total calculated plane area: {self.total_plane_area.item() / 1e6:.2f} km^2")
        print(f"  Total calculated channel surface area: {self.total_channel_surface_area.item() / 1e6:.2f} km^2")
        print(f"  Total precipitation receiving area: {self.total_precip_receiving_area.item() / 1e6:.2f} km^2")

        # Mass balance (can be initialized here and updated per phase)
        self.calculate_area = torch.tensor(0.0, device=self.device, dtype=self.dtype) # Total area for mass balance
        self.total_precip_volume_basin = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        # self.total_infil_volume_basin_planes = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        # self.total_infil_volume_basin_channels = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.total_outlet_flow_volume_basin = torch.tensor(0.0, device=self.device, dtype=self.dtype) # From final phase
        
        # Initial storage calculation can be done once here for overall balance
        self.initial_surface_storage_basin = self._calculate_overall_surface_storage()
        self.initial_soil_moisture_storage_basin = self._calculate_overall_soil_moisture_storage()
        print(f"Initial surface storage (m^3): {self.initial_surface_storage_basin.item():.2f}")
        print(f"Initial soil moisture storage (m^3): {self.initial_soil_moisture_storage_basin.item():.2f}")

    def _calculate_total_infiltration_volume(self) -> torch.Tensor:
        """Calculates total infiltration volume across all elements."""
        total_infiltration_volume_planes = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        total_infiltration_volume_channels = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        for module in self.watershed.element_modules.values():
            # print(f"Cumulative infiltration for element {module.props.element_id}: {module.F_cumulative.item()}")
            props = module.props
            if props.geom_type == 'plane':
                total_infiltration_volume_planes += module.F_cumulative * props.LEN * props.WID
            elif props.geom_type == 'channel':
                total_infiltration_volume_channels += module.F_cumulative * props.WID * props.LEN
        total_infiltration_volume = total_infiltration_volume_planes + total_infiltration_volume_channels
        return total_infiltration_volume_planes, total_infiltration_volume_channels, total_infiltration_volume

    def _calculate_overall_surface_storage(self) -> torch.Tensor:
        total_storage = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        for module in self.watershed.element_modules.values():
            props = module.props 
            if props.num_nodes > 0:
                if props.dx_segments.numel() > 0:
                    flow_area = module.area[:-1] # 
                    dx_tensor = props.dx_avg * torch.ones_like(flow_area, device=self.device, dtype=self.dtype)
                    total_storage += torch.sum(flow_area * dx_tensor)
                # elif props.num_nodes == 1: 
                #     total_storage += module.area[0] * props.LEN
        return total_storage
    
    def _calculate_overall_soil_moisture_storage(self) -> torch.Tensor:
        drying_storage = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        total_storage = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        for module in self.watershed.element_modules.values():
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

    def _run_phase1_planes(self, rainfall_times_s_cpu: torch.Tensor, rainfall_rates_ms_device: torch.Tensor,
                           output_save_interval_s: float):
        print("--- Starting Phase 1: Independent Plane Simulations ---")
        plane_modules = [m for m in self.watershed.element_modules.values() if isinstance(m, PlaneElement)]
        if not plane_modules:
            print("No plane elements found for Phase 1.")
            if self.result_saver_planes: self.result_saver_planes.finalize() # Finalize if created
            return

        # Initialize storage for inter-phase hydrographs
        plane_outflow_times_lists: dict[int, list] = {p.props.element_id: [] for p in plane_modules}
        plane_outflow_q_lists: dict[int, list] = {p.props.element_id: [] for p in plane_modules}

        for p in plane_modules:
            pid = p.props.element_id
            plane_outflow_times_lists[pid].append(torch.tensor(0.0, device=self.device, dtype=self.dtype)) # Start at t=0
            if p.props.num_nodes > 0:
                init_q = p.discharge[-1].clone().detach()
            else:
                init_q = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            plane_outflow_q_lists[pid].append(init_q) 

        current_time_s_float = 0.0
        cumulative_rain_m_tensor = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        next_log_time_s = 0.0
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

            for plane_module in plane_modules:
                _, p_infil_r, infil_d = plane_module(current_rain, dt_tensor, cum_rain_start_step)
                # self.total_infil_volume_basin_planes += infil_d * plane_module.props.WID * plane_module.props.LEN
                current_step_plane_infil_rates_float_map[pid] = p_infil_r.item()
                # Precip for planes is accounted for globally for the basin in the main run method

                # Global precip for this step (for mass balance)
                props_mb = plane_module.props
                self.total_precip_volume_basin += current_rain * props_mb.WID * props_mb.LEN * dt_tensor
            
            current_time_s_float += dt_float
            # step_count +=1
            pbar.update(dt_float)
            
            # Log for inter-phase hydrograph
            if output_save_interval_s > 0 and current_time_s_float >= next_output_log_time_s - EPSILON:
                time_now_tensor = torch.tensor(current_time_s_float, device=self.device, dtype=self.dtype)
                for plane_module in plane_modules:
                    pid = plane_module.props.element_id
                    if plane_module.props.num_nodes > 0:
                        q_out = plane_module.discharge[-1].clone().detach()
                    else:
                        q_out = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                    plane_outflow_times_lists[pid].append(time_now_tensor)
                    plane_outflow_q_lists[pid].append(q_out)
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
                self.result_saver_planes.save_state(save_count_idx_planes, current_time_s_float, plane_states_to_save, 
                                                    current_step_plane_infil_rates_float_map)
                save_count_idx_planes += 1
                next_hdf5_save_time_s += self.save_interval_seconds
        pbar.close()
        
        # Capture final state
        if output_save_interval_s > 0 and \
           abs(current_time_s_float - (next_log_time_s - output_save_interval_s)) > EPSILON * output_save_interval_s and \
           current_time_s_float <= self.sim_duration_s:
            final_time_tensor = torch.tensor(current_time_s_float, device=self.device, dtype=self.dtype)
            for plane_module in plane_modules:
                pid = plane_module.props.element_id
                if not plane_outflow_times_lists[pid] or \
                   abs(plane_outflow_times_lists[pid][-1].item() - current_time_s_float) > EPSILON :
                    if plane_module.props.num_nodes > 0:
                        q_final = plane_module.discharge[-1].clone().detach()
                    else:
                        q_final = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                    plane_outflow_times_lists[pid].append(final_time_tensor)
                    plane_outflow_q_lists[pid].append(q_final)

        for plane_module in plane_modules:
            pid = plane_module.props.element_id
            if plane_outflow_times_lists[pid] and plane_outflow_q_lists[pid]: 
                self.plane_outflow_hydrographs[pid] = (
                    torch.stack(plane_outflow_times_lists[pid]), torch.stack(plane_outflow_q_lists[pid])
                )
            else:
                self.plane_outflow_hydrographs[pid] = (
                    torch.empty(0, device=self.device, dtype=self.dtype), torch.empty(0, device=self.device, dtype=self.dtype)
                )

        if self.result_saver_planes and self.save_interval_seconds >= 0:
            self.result_saver_planes.finalize()

        print(f"--- Phase 1 Finished ---") # ({step_count} steps) 

    def _run_phase2_channels(self, rainfall_times_s_cpu: torch.Tensor, rainfall_rates_ms_device: torch.Tensor):
        print("--- Starting Phase 2: Channel Network Routing ---")
        channel_modules = [m for m in self.watershed.element_modules.values() if isinstance(m, ChannelElement)]
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

                for plane_eid in group_config['planes']:
                    plane_props = self.watershed.get_element_properties(plane_eid)
                    plane_hydrograph_times, plane_hydrograph_q = self.plane_outflow_hydrographs.get(plane_eid, (None, None))
                    
                    if plane_hydrograph_times is not None and plane_hydrograph_q is not None:
                        q_interp = self._interpolate_hydrograph(current_time_s_float, plane_hydrograph_times, plane_hydrograph_q)
                        if plane_props.side == 'head': head_plane_q_tensor += q_interp
                        elif plane_props.side in ['left','right']: side_plane_q_total_tensor += q_interp
                        else: print(f"Warning: Unknown plane side '{plane_props.side}' for element {plane_eid}")                
                upstream_q_final_for_channel = upstream_q_tensor + head_plane_q_tensor
                
                # --- Call Channel Forward ---
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

    def run_decoupled_simulation(self):
        """Runs the full decoupled simulation (Phase 1 planes, then Phase 2 channels)."""
        
        # Prepare rainfall series once
        sim_settings_rain = self.sim_settings
        min_dt_for_rain_gen = self.min_dt_s / 60.0 if self.min_dt_s > 0 else 0.01
        high_res_dt_min = max(0.001, min(0.1, min_dt_for_rain_gen / 5.0))
        rain_times_s_cpu, rain_rates_ms_device, _ = generate_triangular_rainfall(
            sim_settings_rain['sim_duration_min'], sim_settings_rain['rain_event_dur_min'],
            sim_settings_rain['rain_peak_t_min'], sim_settings_rain['rain_peak_mmhr'],
            dt_min_for_gen=high_res_dt_min, save_interval_min=None 
        )
        rain_rates_ms_device = rain_rates_ms_device.to(device=self.device, dtype=self.dtype)

        # Phase 1: Planes
        # Decide on the output_save_interval_s for inter-phase hydrographs
        # This should be small enough to capture hydrograph shape, e.g., min_dt_s or a fixed value like 60s
        phase1_output_interval_s = min(60.0, self.min_dt_s if self.min_dt_s > EPSILON else 60.0)
        if self.sim_settings['save_interval_min'] * 60.0 > EPSILON : # If HDF5 saving is active, align with it or make finer
            phase1_output_interval_s = min(phase1_output_interval_s, self.sim_settings['save_interval_min'] * 60.0)

        self._run_phase1_planes(rain_times_s_cpu, rain_rates_ms_device, phase1_output_interval_s)
        # Phase 2: Channels
        self._run_phase2_channels(rain_times_s_cpu, rain_rates_ms_device)
        
        # Finalize HDF5 saving after Phase 2 is complete
        self.result_saver_channels.finalize()
        self.report_mass_balance() # Report overall mass balance

    def report_mass_balance(self):
        final_surface_storage_basin_end = self._calculate_overall_surface_storage() # Assumes states in modules are final
        final_soil_moisture_storage_basin_end = self._calculate_overall_soil_moisture_storage()
        print(f"Final surface storage (m^3): {final_surface_storage_basin_end.item():.2f}")
        print(f"Final soil moisture storage (m^3): {final_soil_moisture_storage_basin_end.item():.2f}")

        final_infil_volume_planes, final_infil_volume_channels, final_infil_volume_basin = \
            self._calculate_total_infiltration_volume()
        
        delta_surface = final_surface_storage_basin_end - self.initial_surface_storage_basin
        delta_soil = final_soil_moisture_storage_basin_end - self.initial_soil_moisture_storage_basin
                
        total_change_and_outflow = self.total_outlet_flow_volume_basin + delta_surface + delta_soil
        
        error_abs = self.total_precip_volume_basin - total_change_and_outflow
        error_rel = (error_abs / (self.total_precip_volume_basin + EPSILON)) * 100.0

        print("\n--- Overall Mass Balance Summary (Decoupled) ---")
        print(f"  Total Precipitation Volume: {self.total_precip_volume_basin.item():.4f} m^3")
        print(f"  Total Infiltration (Planes): {final_infil_volume_planes.item():.4f} m^3")
        print(f"  Total Infiltration (Channels): {final_infil_volume_channels.item():.4f} m^3")
        print(f"  Total Infiltration (Combined): {final_infil_volume_basin.item():.4f} m^3")
        print(f"  Total Basin Outlet Flow:    {self.total_outlet_flow_volume_basin.item():.4f} m^3")
        print(f"  Change in Surface Storage:  {delta_surface.item():.4f} m^3")
        print(f"  Change in Soil Storage:     {delta_soil.item():.4f} m^3")
        print(f"  Sum of Outputs & Storage Change: {total_change_and_outflow.item():.4f} m^3")
        print(f"  Mass Balance Error (In - Out): {error_abs.item():.4f} m^3 ({error_rel.item():.4f} %)")

