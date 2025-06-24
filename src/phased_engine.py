# src/phased_engine.py 

import torch
import numpy as np
import h5py
from tqdm import tqdm
from collections import defaultdict

from src.data_structures import ElementProperties, OverlandFlowState, InfiltrationState
from src.watershed import Watershed, Watershed_, Watershed_im
from src.io.event_loader import PreloadedRainfallManager, AbstractRainfallManager
from src.io.results_handler import ResultSaver, ResultSaver_v2
from src.core.physics_formulas import (get_plane_h_from_area, calculate_q_manning, 
                                       calculate_dqda_manning_general, 
                                       get_h_from_trapezoid_area, get_trapezoid_wp_from_h,
                                       get_trapezoid_topwidth_from_h) # etc.
from src.utils.rainfall_generator import generate_triangular_rainfall
from src.components.plane_element import PlaneElement, PlaneElement_, PlaneElement_im # For isinstance check
from src.components.channel_element import ChannelElement, ChannelElement_, ChannelElement_im # For isinstance check


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
                total_precip_volume_basin_phase1 += current_rain * props_mb.WID * props_mb.LEN * dt_tensor
            
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

        # --- NEW: Convert lists to a dictionary of tensors ---
        hydrographs_to_save = {}
        for plane_module in plane_modules:
            pid = plane_module.props.element_id
            if plane_outflow_times_lists[pid] and plane_outflow_q_lists[pid]: 
                hydrographs_to_save[pid] = (
                    torch.stack(plane_outflow_times_lists[pid]),
                    torch.stack(plane_outflow_q_lists[pid])
                )

        if self.result_saver_planes:
            self.result_saver_planes.save_hydrographs(hydrographs_to_save)

        # --- NEW: Prepare the mass balance results to be returned ---
        mass_balance_results = {
            'precip_volume': total_precip_volume_basin_phase1,
            'initial_surface_storage': self.initial_surface_storage_basin,
            'initial_soil_moisture_storage': self.initial_soil_moisture_storage_basin,
            'final_surface_storage_planes': self._calculate_overall_surface_storage(plane_modules, self.device, self.dtype),
            'final_soil_moisture_storage_planes': self._calculate_overall_soil_moisture_storage(plane_modules, self.device, self.dtype),
            'final_infiltration_volume_planes': self._calculate_total_infiltration_volume(plane_modules, self.device, self.dtype)[0],
        }

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

class PhasedSimulationEngine_v2:
    def __init__(self,
                 watershed_obj: Watershed_,
                 rainfall_manager: AbstractRainfallManager,
                 result_saver_planes: ResultSaver_v2, 
                 result_saver_channels: ResultSaver_v2, 
                 simulation_settings: dict,
                 device: torch.device,
                 dtype: torch.dtype,
                 phase1_results: dict = None):
        
        self.watershed = watershed_obj
        self.rainfall_manager = rainfall_manager
        self.result_saver_planes = result_saver_planes
        self.phase1_results = phase1_results
        self.result_saver_channels = result_saver_channels
        self.sim_settings = simulation_settings
        self.device = device
        self.dtype = dtype
        self.phase1_results = phase1_results

        self.sim_duration_s = self.sim_settings['sim_duration_min'] * 60.0
        self.save_interval_s = self.sim_settings['save_interval_min'] * 60.0
        self.max_dt_s = self.sim_settings['max_dt_min'] * 60.0
        self.min_dt_s = self.sim_settings['min_dt_min'] * 60.0
        self.cfl_number = self.sim_settings['cfl_number']
        
        self.total_precip_volume_basin = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.total_outlet_flow_volume_basin = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        if self.phase1_results:
            print("Phase 2 Engine initialized with mass balance state from Phase 1.")
            self.total_precip_volume_basin += self.phase1_results['precip_volume']
            self.initial_surface_storage_basin = self.phase1_results['initial_surface_storage']
            self.initial_soil_moisture_storage_basin = self.phase1_results['initial_soil_moisture_storage']
        else:
            print("Phase 1 Engine: Calculating initial basin storage.")
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
            
            all_modules = list(self.watershed.element_modules.values())
            self.initial_surface_storage_basin = self._calculate_overall_surface_storage(all_modules, self.device, self.dtype)
            self.initial_soil_moisture_storage_basin = self._calculate_overall_soil_moisture_storage(all_modules, self.device, self.dtype)

        
        print(f"  Total calculated plane area: {self.total_plane_area.item() / 1e6:.2f} km^2")
        print(f"  Total calculated channel surface area: {self.total_channel_surface_area.item() / 1e6:.2f} km^2")
        print(f"  Total precipitation receiving area: {self.total_precip_receiving_area.item() / 1e6:.2f} km^2")
        print(f"  Initial surface storage (m^3): {self.initial_surface_storage_basin.item():.2f}")
        print(f"  Initial soil moisture storage (m^3): {self.initial_soil_moisture_storage_basin.item():.2f}")

    @staticmethod
    def _calculate_total_infiltration_volume(module_list: list, device, dtype) -> torch.Tensor:
        """Calculates total infiltration volume across all elements."""
        planes_vol = torch.tensor(0.0, device=device, dtype=dtype)
        chans_vol = torch.tensor(0.0, device=device, dtype=dtype)
        for module in module_list:
            # print(f"Cumulative infiltration for element {module.props.element_id}: {module.F_cumulative.item()}")
            props = module.props
            if isinstance(module, PlaneElement_):
                planes_vol += module.F_cumulative * props.LEN * props.WID
            elif isinstance(module, ChannelElement_):
                chans_vol  += module.F_cumulative * props.WID * props.LEN
        # total_infiltration_volume = planes_vol + chans_vol 
        return planes_vol, chans_vol # , total_infiltration_volume

    @staticmethod
    def _calculate_overall_surface_storage(module_list: list, device, dtype) -> torch.Tensor:
        total_storage = torch.tensor(0.0, device=device, dtype=dtype)
        for module in module_list:
            props = module.props 
            if props.num_nodes > 0 and props.dx_segments.numel() > 0:
                total_storage += torch.sum(module.area[:-1] * props.dx_segments)
        return total_storage
    
    @staticmethod
    def _calculate_overall_soil_moisture_storage(module_list: list, device, dtype) -> torch.Tensor:
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
            props, A_state = module.props, module.area
            if phase == 1: # Phase 1: Planes
                # min_dx_val = 4 # Tao: the mini_dx is preprocessed in the watershed loader
                h_state = get_plane_h_from_area(A_state, props.WID)
                wp_state = props.WID.expand_as(A_state)
                topwidth_state = props.WID.expand_as(A_state)
            
            elif phase == 2: # Phase 2: Channels
                # min_dx_val = 10 # Tao: the mini_dx is preprocessed in the watershed loader
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
            else: min_dx_val = props.dx_avg.item()
            dt_elem = self.cfl_number * min_dx_val / max_celerity
            min_dt_overall = min(min_dt_overall, dt_elem)
        
        dt_chosen = min(min_dt_overall, self.max_dt_s) if min_dt_overall != float('inf') else self.max_dt_s
        dt_chosen = max(dt_chosen, self.min_dt_s)
        return dt_chosen

    def _prepare_vectorized_inputs(self, modules: list, phase: int) -> dict:
        """
        Gathers data from a list of modules and prepares padded, vectorized
        tensors for efficient computation.
        """
        if not modules:
            return None

        # 1. Find the maximum number of nodes to determine padding size
        max_nodes = 0
        for m in modules:
            if m.props.num_nodes > max_nodes:
                max_nodes = m.props.num_nodes
                
        if max_nodes < 2: # No active elements for CFL
            return None

        # 2. Initialize lists to hold the data before stacking
        areas, slopes, wids, w0s, ss1s, ss2s = [], [], [], [], [], []
        dx_avgs, masks = [], []

        for m in modules:
            props = m.props
            num_nodes = props.num_nodes
            
            # --- Padding ---
            # Calculate how much padding is needed for this element
            pad_size = max_nodes - num_nodes
            padding = (0, pad_size) # Pad only on the right side

            # Pad the area tensor and create a boolean mask
            areas.append(torch.nn.functional.pad(m.area, padding, "constant", 0))
            mask = torch.zeros(max_nodes, dtype=torch.bool, device=self.device)
            if num_nodes > 0:
                mask[:num_nodes] = True
            masks.append(mask)

            dx_avgs.append(props.dx_avg)

            # --- Gather properties, padding or expanding them to max_nodes ---
            slopes_padded = torch.zeros(max_nodes, device=self.device, dtype=self.dtype)
            slopes_padded[:num_nodes] = props.SL
            if pad_size > 0:
                # replicate the last real value into the padding
                slopes_padded[num_nodes:] = props.SL[-1]
            slopes.append(slopes_padded)
            
            if phase == 1: # Planes
                wids.append(props.WID.expand(max_nodes))
            elif phase == 2: # Channels
                w0s.append(torch.nn.functional.pad(props.W0_nodes, padding, "replicate"))
                ss1s.append(props.SS1.expand(max_nodes))
                ss2s.append(props.SS2.expand(max_nodes))

        # 3. Stack all lists into final, large tensors
        prepared_data = {
            "A_state": torch.stack(areas),
            "SL": torch.stack(slopes),
            "dx_avg": torch.stack(dx_avgs),
            "mask": torch.stack(masks)
        }
        
        if phase == 1:
            prepared_data["WID"] = torch.stack(wids)
        elif phase == 2:
            prepared_data["W0"] = torch.stack(w0s)
            prepared_data["SS1"] = torch.stack(ss1s)
            prepared_data["SS2"] = torch.stack(ss2s)
            
        return prepared_data

    # --- This is the new, primary dt calculator ---
    def _calculate_dt_vectorized(self, modules_to_check: list, phase: int) -> float:
        """
        Calculates a common dt for a list of modules using a single,
        vectorized GPU operation.
        """
        active_modules = [m for m in modules_to_check if m.props.num_nodes >= 2]
        if not active_modules:
            return self.max_dt_s

        # 1. Prepare padded tensors
        vec_inputs = self._prepare_vectorized_inputs(active_modules, phase)
        if vec_inputs is None:
            return self.max_dt_s

        A_state = vec_inputs["A_state"]
        mask = vec_inputs["mask"]

        # 2. Perform vectorized geometry and physics calculations
        with torch.no_grad(): # No need for gradients in dt calculation
            if phase == 1:
                h_state = get_plane_h_from_area(A_state, vec_inputs["WID"])
                wp_state = vec_inputs["WID"]
                topwidth_state = vec_inputs["WID"]
                ss1, ss2 = torch.tensor(0.0), torch.tensor(0.0)
                geom_type = 'plane'
            elif phase == 2:
                h_state = get_h_from_trapezoid_area(A_state, vec_inputs["W0"], vec_inputs["SS1"], vec_inputs["SS2"])
                wp_state = get_trapezoid_wp_from_h(h_state, vec_inputs["W0"], vec_inputs["SS1"], vec_inputs["SS2"])
                topwidth_state = get_trapezoid_topwidth_from_h(h_state, vec_inputs["W0"], vec_inputs["SS1"], vec_inputs["SS2"])
                ss1, ss2 = active_modules[0].props.SS1, active_modules[0].props.SS2 # Use props from one element
                geom_type = 'channel'

            # Use MAN from a single representative element (assuming it's global or similar)
            man = active_modules[0].props.MAN
            
            Q_state = calculate_q_manning(A_state, wp_state, man, vec_inputs["SL"])
            dqda = calculate_dqda_manning_general(A_state, wp_state, topwidth_state, Q_state, ss1, ss2, geom_type)

            # 3. Find the max celerity, ignoring padded values
            # Fill padded areas with a very small number so they don't produce max celerity
            dqda[~mask] = -1.0 
            max_celerities_per_element = torch.max(torch.abs(dqda), dim=1).values
            
            # 4. Calculate stable dt for each element
            stable_dts = self.cfl_number * vec_inputs["dx_avg"] / (max_celerities_per_element + EPSILON)
            
            # 5. Find the minimum dt across all elements
            min_dt_overall = torch.min(stable_dts).item()

        dt_chosen = min(min_dt_overall, self.max_dt_s)
        return max(dt_chosen, self.min_dt_s)

    def _run_phase1_planes(self):
        print("--- Starting Phase 1: Independent Plane Simulations ---")
        plane_modules = []
        for m in self.watershed.element_modules.values():
            # print(f"Checking module {m.props.element_id} of type {type(m)}")
            if isinstance(m, PlaneElement_):
                plane_modules.append(m)

        print(f"Found {len(plane_modules)} plane elements for Phase 1.")

        if not plane_modules:
            print("No plane elements found for Phase 1.")
            return {}
        
        output_save_interval_s = min(60.0, self.sim_settings['min_dt_min'] * 60.0) # Tao: Future edit
        total_precip_volume_basin_phase1 = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        # Initialize storage for inter-phase hydrographs
        plane_outflow_times_lists: dict[int, list] = {p.props.element_id: [
            torch.tensor(0.0, device=self.device, dtype=self.dtype)] for p in plane_modules}
        plane_outflow_q_lists: dict[int, list] = {p.props.element_id: [
            p.discharge[-1].clone().detach() 
            if p.props.num_nodes > 0 
            else torch.tensor(0.0, device=self.device, dtype=self.dtype)] for p in plane_modules}

        current_time_s_float = 0.0
        cumulative_rain_m_tensor = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        save_count_idx_planes = 0

        next_hdf5_save_time_s = 0.0 # Save t=0 for HDF5
        next_output_log_time_s = output_save_interval_s

        pbar = tqdm(total=int(self.sim_duration_s), desc="Phase 1 (Planes)", unit="sim_s")
        while current_time_s_float < self.sim_duration_s - EPSILON:
            # dt_float = self._calculate_dt_for_module_list(plane_modules, phase=1) # CFL for planes
            dt_float = self._calculate_dt_vectorized(plane_modules, phase=1)
            if current_time_s_float + dt_float > self.sim_duration_s:
                dt_float = self.sim_duration_s - current_time_s_float
            if dt_float <= EPSILON: break
            dt_tensor = torch.tensor(dt_float, device=self.device, dtype=self.dtype)

            # Get element-specific rainfall from the manager
            element_rain_rates = self.rainfall_manager.get_rainfall_at_time(current_time_s_float)            
            current_step_infil_rates = {}      

            # This is the cumulative rainfall for the whole system, needed for infiltration
            # It's an approximation using the average rainfall rate
            avg_rain_rate = torch.mean(torch.stack(list(element_rain_rates.values())))
            cum_rain_start_step = cumulative_rain_m_tensor.clone()
            cumulative_rain_m_tensor += avg_rain_rate * dt_tensor

            for plane_module in plane_modules:
                pid = plane_module.props.element_id
                rain_for_this_element = element_rain_rates.get(pid, self.rainfall_manager.zero_rain_tensor)

                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                    _, p_infil_r, infil_d = plane_module(rain_for_this_element, dt_tensor, cum_rain_start_step)
                # self.total_infil_volume_basin_planes += infil_d * plane_module.props.WID * plane_module.props.LEN
                current_step_infil_rates[pid] = p_infil_r.item()
                # Precip for planes is accounted for globally for the basin in the main run method

                # Global precip for this step (for mass balance)
                props_mb = plane_module.props
                total_precip_volume_basin_phase1  += rain_for_this_element * props_mb.WID * props_mb.LEN * dt_tensor
            
            current_time_s_float += dt_float
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
                                                    current_step_infil_rates)
                save_count_idx_planes += 1
                next_hdf5_save_time_s += self.save_interval_s
        pbar.close()

        # --- Convert lists to a dictionary of tensors ---
        hydrographs_to_save = {}
        for plane_module in plane_modules:
            pid = plane_module.props.element_id
            if plane_outflow_times_lists[pid] and plane_outflow_q_lists[pid]: 
                hydrographs_to_save[pid] = (
                    torch.stack(plane_outflow_times_lists[pid]),
                    torch.stack(plane_outflow_q_lists[pid])
                )

        # --- Save the hydrographs to the HDF5 file ---
        # The ResultSaver for planes will handle saving this new data.
        if self.result_saver_planes:
            self.result_saver_planes.save_hydrographs(hydrographs_to_save)

        # --- Prepare the mass balance results to be returned ---
        mass_balance_results = {
            'precip_volume': total_precip_volume_basin_phase1,
            'initial_surface_storage': self.initial_surface_storage_basin,
            'initial_soil_moisture_storage': self.initial_soil_moisture_storage_basin,
            'final_surface_storage_planes': self._calculate_overall_surface_storage(plane_modules, self.device, self.dtype),
            'final_soil_moisture_storage_planes': self._calculate_overall_soil_moisture_storage(plane_modules, self.device, self.dtype),
            'final_infiltration_volume_planes': self._calculate_total_infiltration_volume(plane_modules, self.device, self.dtype)[0],
        }
        print(f"--- Phase 1 Finished ---") # 
        return mass_balance_results

    def _run_phase2_channels(self):
        print("--- Starting Phase 2: Channel Network Routing ---")
        channel_modules = [m for m in self.watershed.element_modules.values() if isinstance(m, ChannelElement_)]
        if not channel_modules:
            print("No channel elements found for Phase 2.")
            return

        # Use the main HDF5 saver for Phase 2 outputs
        save_interval_s_phase2 = self.sim_settings['save_interval_min'] * 60.0
        next_save_time_float = save_interval_s_phase2 if save_interval_s_phase2 >= 0 else float('inf')

        current_time_s_float = 0.0
        cumulative_rain_m_tensor = torch.tensor(0.0, device=self.device, dtype=self.dtype) # Separate for channels

        pbar = tqdm(total=int(self.sim_duration_s), desc="Phase 2 (Channels)", unit="sim_s")
        while current_time_s_float < self.sim_duration_s - EPSILON:
            # dt_float = self._calculate_dt_for_module_list(channel_modules, phase=2) # CFL for channels
            dt_float = self._calculate_dt_vectorized(channel_modules, phase=2)
            if current_time_s_float + dt_float > self.sim_duration_s:
                dt_float = self.sim_duration_s - current_time_s_float
            if dt_float <= EPSILON: break
            # print(f"Current dt: {dt_float:.2f}")
            dt_tensor = torch.tensor(dt_float, device=self.device, dtype=self.dtype)

            element_rain_rates  = self.rainfall_manager.get_rainfall_at_time(current_time_s_float)
            avg_rain_rate = torch.mean(torch.stack(list(element_rain_rates.values())))
            cum_rain_start_step = cumulative_rain_m_tensor.clone()
            cumulative_rain_m_tensor += avg_rain_rate * dt_tensor

            
            channel_outflows_this_step = {}
            element_infil_rates_this_step: dict[int, float] = {}
            time_idx = min(int(current_time_s_float), self.resampled_plane_q.shape[1] - 1)

            for group_id in self.watershed.simulation_order: # This order includes all groups
                group_config = self.watershed.get_group_config(group_id)
                if not group_config or not group_config['channel']: continue # Skip if no channel in group

                channel_eid = group_config['channel']
                channel_module = self.watershed.get_element_module(channel_eid)
                if not isinstance(channel_module, ChannelElement_): continue # Should not happen

                # Global precip for this step (for mass balance)
                props_c = channel_module.props
                total_precip_volume_basin_phase2  += element_rain_rates * props_c.WID * props_c.LEN * dt_tensor

                # --- Upstream Channel Inflow ---
                upstream_q_tensor = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                for up_grp_id, down_grp_id in self.watershed.connectivity.items():
                    if down_grp_id == group_id:
                        upstream_q_tensor += channel_outflows_this_step.get(
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

                rain_for_channel = element_rain_rates.get(channel_eid, self.rainfall_manager.zero_rain_tensor)
                
                # --- Call Channel Forward ---
                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                    c_out_q, c_infil_r, c_infil_d = channel_module(
                    rain_for_channel, upstream_q_final_for_channel,side_plane_q_total_tensor, dt_tensor, cum_rain_start_step)
                
                channel_outflows_this_step[group_id] = c_out_q
                element_infil_rates_this_step[channel_eid] = c_infil_r.item()
                props_c = channel_module.props
                self.total_infil_volume_basin_channels += c_infil_d * props_c.W0_nodes.mean() * props_c.LEN
                self.total_precip_volume_basin += rain_for_channel * props_c.LEN * props_c.WID * dt_tensor

                if group_id not in self.watershed.connectivity:
                    self.total_outlet_flow_volume_basin += c_out_q * dt_tensor
            
            current_time_s_float += dt_float
            pbar.update(dt_float)

            if save_interval_s_phase2 >=0 and current_time_s_float >= next_save_time_float - EPSILON:
                all_current_states_for_saving = {
                    eid: {'flow': OverlandFlowState(mod.t_elapsed.clone(), mod.area.clone(), mod.depth.clone(), 
                                                    mod.discharge.clone(), mod.max_cfl.clone()),
                          'infil': InfiltrationState(mod.theta_current.clone(), mod.F_cumulative.clone(), 
                                                     mod.drying_cumulative.clone())}
                    for eid, mod in self.watershed.element_modules.items() if isinstance(mod, ChannelElement_)
                }
                self.result_saver_channels.save_state(save_count_idx_phase2, current_time_s_float,
                                             all_current_states_for_saving, element_infil_rates_this_step)
                save_count_idx_phase2 += 1
                next_save_time_float += save_interval_s_phase2
        pbar.close()
        print(f"--- Phase 2 Finished ---")  # ({step_count} steps)

    def load_and_resample_plane_hydrographs(self, hdf5_path: str):
        print(f"Loading and resampling hydrographs from {hdf5_path}...")
        with h5py.File(hdf5_path, 'r') as hf:
            if "hydrographs" not in hf: raise ValueError("Hydrographs not in HDF5 file.")
            plane_hydrographs = {int(k): (torch.from_numpy(v['time_s'][:]), torch.from_numpy(v['discharge_m3s'][:])) for k, v in hf['hydrographs'].items()}
        
        num_points = int(self.sim_duration_s) + 1
        self.resampled_time_grid = torch.linspace(0, self.sim_duration_s, num_points, device=self.device, dtype=self.dtype)
        
        plane_ids = list(plane_hydrographs.keys())
        self.resampled_plane_q = torch.zeros(len(plane_ids), num_points, device=self.device, dtype=self.dtype)
        self.plane_id_to_idx_map = {pid: i for i, pid in enumerate(plane_ids)}

        for i, pid in enumerate(plane_ids):
            source_t, source_q = plane_hydrographs[pid]
            source_t, source_q = source_t.to(self.device), source_q.to(self.device)
            if source_t.numel() < 2: continue
            
            indices = torch.searchsorted(source_t, self.resampled_time_grid, right=True) - 1
            indices.clamp_(0, source_t.numel() - 2)
            t0, t1 = source_t[indices], source_t[indices + 1]
            q0, q1 = source_q[indices], source_q[indices + 1]
            
            dt_segment = t1 - t0
            interp_factor = torch.where(dt_segment > EPSILON, (self.resampled_time_grid - t0) / dt_segment, 0)
            self.resampled_plane_q[i, :] = q0 + interp_factor * (q1 - q0)
        print("Hydrograph resampling complete.")

    def _get_plane_inflows_for_group(self, group_id: int, time_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        head_q = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        side_q = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        plane_eids = self.watershed.get_group_config(group_id)['planes']
        if not plane_eids: return head_q, side_q
        
        plane_row_indices = [self.plane_id_to_idx_map[pid] for pid in plane_eids if pid in self.plane_id_to_idx_map]
        if not plane_row_indices: return head_q, side_q

        q_values_for_planes = self.resampled_plane_q[plane_row_indices, time_idx]
        
        q_idx = 0
        for plane_eid in plane_eids:
            if plane_eid in self.plane_id_to_idx_map:
                if self.watershed.get_element_properties(plane_eid).side == 'head':
                    head_q += q_values_for_planes[q_idx]
                else:
                    side_q += q_values_for_planes[q_idx]
                q_idx += 1
        return head_q, side_q

    def report_mass_balance(self):
        if not self.phase1_results:
            print("Warning: Reporting mass balance without Phase 1 results.")
            return

        channel_modules = [m for m in self.watershed.element_modules.values() if isinstance(m, ChannelElement_)]
        final_surface_storage_channels = self._calculate_overall_surface_storage(channel_modules)
        final_soil_moisture_storage_channels = self._calculate_overall_soil_moisture_storage(channel_modules)
        final_infil_volume_planes, final_infil_volume_channels, final_infil_volume_basin = \
            self._calculate_total_infiltration_volume(channel_modules)
        
        final_surface_storage_end = self.phase1_results['final_surface_storage_planes'] + final_surface_storage_channels
        final_soil_moisture_storage_end = self.phase1_results['final_soil_moisture_storage_planes'] + final_soil_moisture_storage_channels
        
        delta_surface = final_surface_storage_end - self.initial_surface_storage_basin
        delta_soil = final_soil_moisture_storage_end - self.initial_soil_moisture_storage_basin
        
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


class PhasedSimulationEngine_v3: # using the implicit solver
    def __init__(self,
                 watershed_obj: Watershed_im,
                 rainfall_manager: AbstractRainfallManager,
                 result_saver_planes: ResultSaver_v2, 
                 result_saver_channels: ResultSaver_v2, 
                 simulation_settings: dict,
                 device: torch.device,
                 dtype: torch.dtype,
                 phase1_results: dict = None):
        
        self.watershed = watershed_obj
        self.rainfall_manager = rainfall_manager
        self.result_saver_planes = result_saver_planes
        self.phase1_results = phase1_results
        self.result_saver_channels = result_saver_channels
        self.sim_settings = simulation_settings
        self.device = device
        self.dtype = dtype
        self.phase1_results = phase1_results

        self.sim_duration_s = self.sim_settings['sim_duration_min'] * 60.0
        self.save_interval_s = self.sim_settings['save_interval_min'] * 60.0

        self.total_plane_area = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.total_channel_surface_area = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        for module in self.watershed.element_modules.values():
            props = module.props
            if isinstance(module, PlaneElement):
                self.total_plane_area += props.LEN * props.WID
            elif isinstance(module, ChannelElement):
                self.total_channel_surface_area += props.LEN * props.WID
        self.total_precip_receiving_area = self.total_plane_area + self.total_channel_surface_area

        # --- Mass Balance Initialization ---
        self.total_precip_volume_basin = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.total_outlet_flow_volume_basin = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        if self.phase1_results:
            print("Phase 2 Engine initialized with mass balance state from Phase 1.")
            self.total_precip_volume_basin += self.phase1_results['precip_volume']
            self.initial_surface_storage_basin = self.phase1_results['initial_surface_storage']
            self.initial_soil_moisture_storage_basin = self.phase1_results['initial_soil_moisture_storage']
        else:
            print("Phase 1 Engine: Calculating initial basin storage.")
            all_modules = list(self.watershed.element_modules.values())
            self.initial_surface_storage_basin = self._calculate_overall_surface_storage(all_modules)
            self.initial_soil_moisture_storage_basin = self._calculate_overall_soil_moisture_storage(all_modules)
        
        print(f"  Total calculated plane area: {self.total_plane_area.item() / 1e6:.2f} km^2")
        print(f"  Total precipitation receiving area: {self.total_precip_receiving_area.item() / 1e6:.2f} km^2")
        print(f"  Initial surface storage (m^3): {self.initial_surface_storage_basin.item():.2f}")
        print(f"  Initial soil moisture storage (m^3): {self.initial_soil_moisture_storage_basin.item():.2f}")

    def _calculate_total_infiltration_volume(self, module_list: list, device, dtype) -> torch.Tensor:
        """Calculates total infiltration volume across all elements."""
        planes_vol = torch.tensor(0.0, device=device, dtype=dtype)
        chans_vol = torch.tensor(0.0, device=device, dtype=dtype)
        for module in module_list:
            # print(f"Cumulative infiltration for element {module.props.element_id}: {module.F_cumulative.item()}")
            props = module.props
            if isinstance(module, PlaneElement_):
                planes_vol += module.F_cumulative * props.LEN * props.WID
            elif isinstance(module, ChannelElement_):
                chans_vol  += module.F_cumulative * props.WID * props.LEN
        total_infiltration_volume = planes_vol + chans_vol 
        return planes_vol, chans_vol, total_infiltration_volume

    def _calculate_overall_surface_storage(self, module_list: list) -> torch.Tensor:
        total_storage = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        for module in module_list:
            props = module.props 
            if props.num_nodes > 0 and props.dx_segments.numel() > 0:
                total_storage += torch.sum(module.area[:-1] * props.dx_segments)
        return total_storage
    
    def _calculate_overall_soil_moisture_storage(self, module_list: list) -> torch.Tensor:
        zero_tensor = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        drying_storage = zero_tensor
        total_storage = zero_tensor
        for module in module_list:
            props = module.props
            soil_p = module.soil_params
            element_surface_area = props.LEN * props.WID
            drying_storage += torch.sum(module.drying_cumulative * soil_p.effective_depth * element_surface_area) 
            total_storage += torch.sum(module.theta_current.mean() * soil_p.effective_depth * element_surface_area )
        return total_storage - drying_storage

    def _run_phase1_planes(self):
        print("--- Starting Phase 1: Independent Plane Simulations ---")
        plane_modules = []
        for m in self.watershed.element_modules.values():
            # print(f"Checking module {m.props.element_id} of type {type(m)}")
            if isinstance(m, PlaneElement_im):
                plane_modules.append(m)

        print(f"Found {len(plane_modules)} plane elements for Phase 1.")

        if not plane_modules:
            print("No plane elements found for Phase 1.")
            return {}
        
        dt_s = self.sim_settings['fixed_dt_min'] * 60.0
        num_steps = int(self.sim_duration_s / dt_s)
        dt_tensor = torch.tensor(dt_s, device=self.device, dtype=self.dtype)

        total_precip_volume_basin_phase1 = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        hydrographs_to_save = {p.props.element_id: ([], []) for p in plane_modules}

        # Initial state for hydrographs at t=0
        for p in plane_modules:
            hydrographs_to_save[p.props.element_id][0].append(0.0)
            hydrographs_to_save[p.props.element_id][1].append(p.discharge[-1].item()
                                                              if p.props.num_nodes > 0 else 0.0)

        pbar = tqdm(total=num_steps, desc="Phase 1 (Planes)", unit="steps")
        for i in range(num_steps):
            dt_float = i * dt_s 
            element_rain_rates = self.rainfall_manager.get_rainfall_at_time(dt_float)            

            for plane_module in plane_modules:
                pid = plane_module.props.element_id
                rain_for_this_element = element_rain_rates.get(pid, self.rainfall_manager.zero_rain_tensor)

                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                    outlet_q, p_infil_r, infil_d = plane_module(rain_for_this_element, dt_tensor)

                hydrographs_to_save[pid][0].append(dt_float)
                hydrographs_to_save[pid][1].append(outlet_q.item())
                total_precip_volume_basin_phase1 += rain_for_this_element * plane_module.props.LEN * plane_module.props.WID * dt_tensor

            pbar.update(1)
        pbar.update(dt_float)

        # Convert hydrograph lists to tensors
        final_hydrographs = {pid: (torch.tensor(t_list), torch.tensor(q_list)) for pid, (t_list, q_list) in hydrographs_to_save.items()}
            
        mass_balance_results = {
            'precip_volume': total_precip_volume_basin_phase1,
            'initial_surface_storage': self.initial_surface_storage_basin,
            'initial_soil_moisture_storage': self.initial_soil_moisture_storage_basin,
            'final_surface_storage_planes': self._calculate_overall_surface_storage(plane_modules),
            'final_soil_moisture_storage_planes': self._calculate_overall_soil_moisture_storage(plane_modules),
            'final_infiltration_volume_planes': self._calculate_total_infiltration_volume(plane_modules)[0],
        }
        print(f"--- Phase 1 Finished ---")
        return final_hydrographs, mass_balance_results

    def _run_phase2_channels(self):
        print("--- Starting Phase 2: Channel Network Routing ---")
        channel_modules = [m for m in self.watershed.element_modules.values() if isinstance(m, ChannelElement_im)]
        if not channel_modules:
            print("No channel elements found for Phase 2.")
            return

        # Use the main HDF5 saver for Phase 2 outputs
        save_interval_s_phase2 = self.sim_settings['save_interval_min'] * 60.0
        next_save_time_float = save_interval_s_phase2 if save_interval_s_phase2 >= 0 else float('inf')

        dt_s = self.sim_settings['fixed_dt_min'] * 60.0
        num_steps = int(self.sim_duration_s / dt_s)
        dt_tensor = torch.tensor(dt_s, device=self.device, dtype=self.dtype)

        channel_outflows_this_step = {}
        element_infil_rates_this_step: dict[int, float] = {}

        pbar = tqdm(total=num_steps, desc="Phase 2 (Channels)", unit="steps")
        for i in range(num_steps):
            current_time_s_float = i * dt_s
            element_rain_rates = self.rainfall_manager.get_rainfall_at_time(current_time_s_float)

            time_idx = min(int(current_time_s_float), self.resampled_plane_q.shape[1] - 1)

            
            for group_id in self.watershed.simulation_order: # This order includes all groups
                group_config = self.watershed.get_group_config(group_id)
                if not group_config or not group_config['channel']: continue # Skip if no channel in group

                channel_eid = group_config['channel']
                channel_module = self.watershed.get_element_module(channel_eid)
                if not isinstance(channel_module, ChannelElement_): continue # Should not happen

                # --- Upstream Channel Inflow ---
                upstream_q_tensor = torch.tensor(0.0, device=self.device, dtype=self.dtype)
                for up_grp_id, down_grp_id in self.watershed.connectivity.items():
                    if down_grp_id == group_id:
                        upstream_q_tensor += channel_outflows_this_step.get(
                            up_grp_id, torch.tensor(0.0))
                
                head_plane_q, side_plane_q = self._get_plane_inflows_for_group(group_id, time_idx)          
                upstream_q_final_for_channel = upstream_q_tensor + head_plane_q

                rain_for_channel = element_rain_rates.get(channel_eid, self.rainfall_manager.zero_rain_tensor)
                
                # --- Call Channel Forward ---
                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                    c_out_q, c_infil_r, c_infil_d = channel_module(
                    rain_for_channel, upstream_q_final_for_channel,side_plane_q, dt_tensor)
                
                channel_outflows_this_step[group_id] = c_out_q
                element_infil_rates_this_step[channel_eid] = c_infil_r.item()

                # Mass balance accumulation
                self.total_precip_volume_basin += rain_for_channel * channel_module.props.LEN * channel_module.props.WID * dt_tensor
                if group_id not in self.watershed.connectivity:
                    self.total_outlet_flow_volume_basin += c_out_q * dt_tensor

            if save_interval_s_phase2 >=0 and current_time_s_float >= next_save_time_float - EPSILON:
                all_current_states_for_saving = {
                    eid: {'flow': OverlandFlowState(mod.t_elapsed.clone(), mod.area.clone(), mod.depth.clone(), 
                                                    mod.discharge.clone(), mod.max_cfl.clone()),
                          'infil': InfiltrationState(mod.theta_current.clone(), mod.F_cumulative.clone(), 
                                                     mod.drying_cumulative.clone())}
                    for eid, mod in self.watershed.element_modules.items() if isinstance(mod, ChannelElement_)
                }
                self.result_saver_channels.save_state(save_count_idx_phase2, current_time_s_float,
                                             all_current_states_for_saving, element_infil_rates_this_step)
                save_count_idx_phase2 += 1
                next_save_time_float += save_interval_s_phase2

            pbar.update(1)
        pbar.close()
        print(f"--- Phase 2 Finished ---")  # ({step_count} steps)

    def load_and_resample_plane_hydrographs(self, hdf5_path: str):
        print(f"Loading and resampling hydrographs from {hdf5_path}...")
        with h5py.File(hdf5_path, 'r') as hf:
            if "hydrographs" not in hf: raise ValueError("Hydrographs not in HDF5 file.")
            plane_hydrographs = {int(k): (torch.from_numpy(v['time_s'][:]), torch.from_numpy(v['discharge_m3s'][:])) for k, v in hf['hydrographs'].items()}
        
        num_points = int(self.sim_duration_s) + 1
        self.resampled_time_grid = torch.linspace(0, self.sim_duration_s, num_points, device=self.device, dtype=self.dtype)
        
        plane_ids = list(plane_hydrographs.keys())
        self.resampled_plane_q = torch.zeros(len(plane_ids), num_points, device=self.device, dtype=self.dtype)
        self.plane_id_to_idx_map = {pid: i for i, pid in enumerate(plane_ids)}

        for i, pid in enumerate(plane_ids):
            source_t, source_q = plane_hydrographs[pid]
            source_t, source_q = source_t.to(self.device), source_q.to(self.device)
            if source_t.numel() < 2: continue
            
            indices = torch.searchsorted(source_t, self.resampled_time_grid, right=True) - 1
            indices.clamp_(0, source_t.numel() - 2)
            t0, t1 = source_t[indices], source_t[indices + 1]
            q0, q1 = source_q[indices], source_q[indices + 1]
            
            dt_segment = t1 - t0
            interp_factor = torch.where(dt_segment > EPSILON, (self.resampled_time_grid - t0) / dt_segment, 0)
            self.resampled_plane_q[i, :] = q0 + interp_factor * (q1 - q0)
        print("Hydrograph resampling complete.")

    def _get_plane_inflows_for_group(self, group_id: int, time_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        head_q = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        side_q = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        plane_eids = self.watershed.get_group_config(group_id)['planes']
        if not plane_eids: return head_q, side_q
        
        plane_row_indices = [self.plane_id_to_idx_map[pid] for pid in plane_eids if pid in self.plane_id_to_idx_map]
        if not plane_row_indices: return head_q, side_q

        q_values_for_planes = self.resampled_plane_q[plane_row_indices, time_idx]
        
        q_idx = 0
        for plane_eid in plane_eids:
            if plane_eid in self.plane_id_to_idx_map:
                if self.watershed.get_element_properties(plane_eid).side == 'head':
                    head_q += q_values_for_planes[q_idx]
                else:
                    side_q += q_values_for_planes[q_idx]
                q_idx += 1
        return head_q, side_q

    def report_mass_balance(self):
        if not self.phase1_results:
            print("Warning: Reporting mass balance without Phase 1 results.")
            return

        channel_modules = [m for m in self.watershed.element_modules.values() if isinstance(m, ChannelElement_)]
        final_surface_storage_channels = self._calculate_overall_surface_storage(channel_modules, self.device, self.dtype)
        final_soil_moisture_storage_channels = self._calculate_overall_soil_moisture_storage(channel_modules, self.device, self.dtype)
        final_infil_volume_planes, final_infil_volume_channels, final_infil_volume_basin = \
            self._calculate_total_infiltration_volume(channel_modules, self.device, self.dtype)
        
        final_surface_storage_end = self.phase1_results['final_surface_storage_planes'] + final_surface_storage_channels
        final_soil_moisture_storage_end = self.phase1_results['final_soil_moisture_storage_planes'] + final_soil_moisture_storage_channels
        
        delta_surface = final_surface_storage_end - self.initial_surface_storage_basin
        delta_soil = final_soil_moisture_storage_end - self.initial_soil_moisture_storage_basin
        
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