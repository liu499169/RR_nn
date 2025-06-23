# src/components/channel_element.py
import torch
import torch.nn as nn

from src.data_structures import ElementProperties, SoilPropertiesIntermediate, OverlandFlowState, InfiltrationState
from src.core.infiltration import infiltration_step_intermediate
from src.core.kinematic_wave_solvers import explicit_muscl_yu_duan_with_plane_contrib
from src.core.physics_formulas import (get_h_from_trapezoid_area, get_trapezoid_wp_from_h,
                                       get_trapezoid_topwidth_from_h, calculate_q_manning,
                                       calculate_froude_number, calculate_cfl_number)
# @torch.compile
class ChannelElement_(nn.Module):
    def __init__(self, 
                 props: ElementProperties, 
                 soil_params: SoilPropertiesIntermediate,
                 device: torch.device, 
                 dtype: torch.dtype):
        super().__init__()
        
        # Store Properties and Parameters (assumed to be on correct device/dtype)
        self.props: ElementProperties = props # self._move_props_to_device(props, device, dtype)
        self.soil_params: SoilPropertiesIntermediate = soil_params # self._move_soil_params_to_device(soil_params, device, dtype)
        
        self.device = device
        self.dtype = dtype

        # --- Initialize State Buffers ---
        self.register_buffer('area', torch.zeros(self.props.num_nodes, device=self.device, dtype=self.dtype))
        self.register_buffer('depth', torch.zeros(self.props.num_nodes, device=self.device, dtype=self.dtype))
        self.register_buffer('discharge', torch.zeros(self.props.num_nodes, device=self.device, dtype=self.dtype))
        self.register_buffer('t_elapsed', torch.tensor(0.0, device=self.device, dtype=self.dtype))
        self.register_buffer('max_cfl', torch.tensor(0.0, device=device, dtype=dtype))

        init_theta = self.soil_params.theta_init_condition.clone().detach()
        self.register_buffer('theta_current', init_theta)
        self.register_buffer('F_cumulative', torch.tensor(0.0, device=self.device, dtype=self.dtype))
        self.register_buffer('drying_cumulative', torch.tensor(0.0, device=self.device, dtype=self.dtype)) # 

    @torch.compile
    def forward(self, 
                current_rain_rate_ms_tensor: torch.Tensor,    # Scalar tensor
                upstream_q_total_tensor: torch.Tensor,        # Scalar tensor (total Q m^3/s)
                plane_lateral_q_total_tensor: torch.Tensor,   # Scalar tensor (total Q m^3/s from side planes)
                dt_s_tensor: torch.Tensor,                    # Scalar tensor
                cumulative_rain_m_start_tensor: torch.Tensor # Scalar tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # outflow_q, infil_rate, infil_depth
        """
        Performs one time step update for the channel element.
        Updates internal states. Returns key results.
        """
        current_internal_infil_state = InfiltrationState(
            theta_current=self.theta_current, 
            F_cumulative=self.F_cumulative,
            drying_cumulative=self.drying_cumulative
        )

        # --- Infiltration ---
        # head_input_for_infil = self.depth.mean() if self.props.num_nodes > 0 else \
        #                        torch.tensor(0.0, device=self.device, dtype=self.dtype)
        
        updated_infil_state, infil_rate_tensor_ms, infil_depth_tensor_m = infiltration_step_intermediate(
            state=current_internal_infil_state, params=self.soil_params,
            rain_rate=current_rain_rate_ms_tensor, surface_head=self.depth, #head_input_for_infil,
            dt=dt_s_tensor, cumulative_rain_start=cumulative_rain_m_start_tensor
        )
        self.theta_current.copy_(updated_infil_state.theta_current)
        self.F_cumulative.copy_(updated_infil_state.F_cumulative)
        self.drying_cumulative.copy_(updated_infil_state.drying_cumulative)

        # --- Lateral Inflow for Solver (from direct precipitation on channel surface) ---
        rain_rate_tensor = torch.clamp(current_rain_rate_ms_tensor - infil_rate_tensor_ms, min=0.0) # 
        
        if self.props.num_nodes > 0:
            avg_h_for_tw = self.depth.mean() if self.props.num_nodes > 0 else \
                           torch.tensor(0.0, device=self.device, dtype=self.dtype)
            
            # props.W0_nodes is a tensor for channels
            w0_for_tw_calc = self.props.W0_nodes[0] if self.props.num_nodes == 1 else self.props.W0_nodes

            avg_tw = get_trapezoid_topwidth_from_h(avg_h_for_tw, w0_for_tw_calc, 
                                                  self.props.SS1, self.props.SS2) 
            q_lat_surface_head = self.depth * avg_tw
            q_lat_nodes_prcp = current_rain_rate_ms_tensor.expand(self.props.num_nodes) * avg_tw 
            q_lat_nodes_infil = infil_rate_tensor_ms.expand(self.props.num_nodes) * avg_tw 
            # q_lat_nodes_precip_expanded = q_lat_nodes_prcp.expand(self.props.num_nodes)
            q_lat = torch.clamp(q_lat_surface_head/dt_s_tensor +  q_lat_nodes_prcp - q_lat_nodes_infil, min=0.0)
        else:
            q_lat = torch.empty(0, device=self.device, dtype=self.dtype)

        # --- Channel Flow Solver (MUSCL) ---
        # Solvers require num_nodes >= 2. Handle n_nodes < 2 gracefully.
        if self.props.num_nodes >= 2:
            dx_at_nodes = torch.zeros_like(self.area)
            if self.props.num_nodes > 2: # for interior nodes
                 dx_at_nodes[1:-1] = (self.props.dx_segments[:-1] + self.props.dx_segments[1:]) / 2.
            # Boundary nodes
            dx_at_nodes[0] = self.props.dx_segments[0]
            dx_at_nodes[-1] = self.props.dx_segments[-1]
            dx_segment_tensor_for_cfl = dx_at_nodes # Shape (num_nodes,)

            A_next = explicit_muscl_yu_duan_with_plane_contrib(
                A_curr=self.area, 
                q_lat_nodes_prcp=q_lat, # lateral influx (m^2/s)
                element_props=self.props, 
                dt=dt_s_tensor, 
                upstream_Q=upstream_q_total_tensor, 
                plane_lat_Q_total=plane_lateral_q_total_tensor
            )
            # Update flow state buffers
            self.area.copy_(torch.clamp(A_next, min=0.0))
            self.depth.copy_(get_h_from_trapezoid_area(self.area, self.props.W0_nodes, 
                                                      self.props.SS1, self.props.SS2))
            wp_next = get_trapezoid_wp_from_h(self.depth, self.props.W0_nodes, 
                                              self.props.SS1, self.props.SS2)
            self.discharge.copy_(calculate_q_manning(self.area, wp_next, self.props.MAN, self.props.SL))
            
            outflow_q_tensor = self.discharge[-1]
        # elif self.props.num_nodes == 1: # Simplified storage update for 1-node channel
        #      # Inflow = upstream_Q + plane_lateral_Q (total) + q_lat_nodes_precip_expanded[0] * LEN
        #      # Outflow = Q_manning(A_curr)
        #      # d(Area*LEN)/dt = Inflow - Outflow
        #      q_in_total = upstream_q_total_tensor + plane_lateral_q_total_tensor + \
        #                   (q_lat_nodes_precip_expanded[0] * self.props.LEN if q_lat_nodes_precip_expanded.numel()>0 else torch.tensor(0.0, device=self.device, dtype=self.dtype))
             
        #      # Current outflow (approximate for single cell)
        #      wp_curr = get_trapezoid_wp_from_h(self.depth, self.props.W0_nodes, self.props.SS1, self.props.SS2)
        #      q_out_curr = calculate_q_manning(self.area, wp_curr, self.props.MAN, self.props.SL)

        #      delta_A = (q_in_total - q_out_curr) * (dt_s_tensor / self.props.LEN) if self.props.LEN > 0 else torch.tensor(0.0, device=self.device, dtype=self.dtype)
        #      self.area.copy_(torch.clamp(self.area + delta_A, min=0.0))
        #      self.depth.copy_(get_h_from_trapezoid_area(self.area, self.props.W0_nodes, self.props.SS1, self.props.SS2))
        #      wp_next = get_trapezoid_wp_from_h(self.depth, self.props.W0_nodes, self.props.SS1, self.props.SS2)
        #      self.discharge.copy_(calculate_q_manning(self.area, wp_next, self.props.MAN, self.props.SL))
        #      outflow_q_tensor = self.discharge[-1]

        #      dx_segment_tensor_for_cfl = self.props.LEN

        else: # num_nodes == 0
            outflow_q_tensor = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        self.t_elapsed.copy_(self.t_elapsed + dt_s_tensor)

        current_cfl_numbers_nodes = calculate_cfl_number(
                self.discharge, self.area, dx_segment_tensor_for_cfl, dt_s_tensor
            )
        max_cfl_this_step = torch.max(current_cfl_numbers_nodes) if current_cfl_numbers_nodes.numel() > 0 else \
              torch.tensor(0.0, device=self.device, dtype=self.dtype)
        
        # Update the element's all-time max Froude number
        self.max_cfl.copy_(
                torch.maximum(self.max_cfl, max_cfl_this_step)
            )

        return outflow_q_tensor, infil_rate_tensor_ms, infil_depth_tensor_m
    

class ChannelElement(nn.Module):
    def __init__(self, 
                 props: ElementProperties, 
                 soil_params: SoilPropertiesIntermediate,
                 device: torch.device, 
                 dtype: torch.dtype):
        super().__init__()
        
        # Store Properties and Parameters (assumed to be on correct device/dtype)
        self.props: ElementProperties = props # self._move_props_to_device(props, device, dtype)
        self.soil_params: SoilPropertiesIntermediate = soil_params # self._move_soil_params_to_device(soil_params, device, dtype)
        
        self.device = device
        self.dtype = dtype

        # --- Initialize State Buffers ---
        self.register_buffer('area', torch.zeros(self.props.num_nodes, device=self.device, dtype=self.dtype))
        self.register_buffer('depth', torch.zeros(self.props.num_nodes, device=self.device, dtype=self.dtype))
        self.register_buffer('discharge', torch.zeros(self.props.num_nodes, device=self.device, dtype=self.dtype))
        self.register_buffer('t_elapsed', torch.tensor(0.0, device=self.device, dtype=self.dtype))
        self.register_buffer('max_cfl', torch.tensor(0.0, device=device, dtype=dtype))

        init_theta = self.soil_params.theta_init_condition.clone().detach()
        self.register_buffer('theta_current', init_theta)
        self.register_buffer('F_cumulative', torch.tensor(0.0, device=self.device, dtype=self.dtype))
        self.register_buffer('drying_cumulative', torch.tensor(0.0, device=self.device, dtype=self.dtype)) # 

    def forward(self, 
                current_rain_rate_ms_tensor: torch.Tensor,    # Scalar tensor
                upstream_q_total_tensor: torch.Tensor,        # Scalar tensor (total Q m^3/s)
                plane_lateral_q_total_tensor: torch.Tensor,   # Scalar tensor (total Q m^3/s from side planes)
                dt_s_tensor: torch.Tensor,                    # Scalar tensor
                cumulative_rain_m_start_tensor: torch.Tensor # Scalar tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # outflow_q, infil_rate, infil_depth
        """
        Performs one time step update for the channel element.
        Updates internal states. Returns key results.
        """
        current_internal_infil_state = InfiltrationState(
            theta_current=self.theta_current, 
            F_cumulative=self.F_cumulative,
            drying_cumulative=self.drying_cumulative
        )

        # --- Infiltration ---
        updated_element_state, infil_rate_NODES, infil_depth_NODES = \
            infiltration_step_intermediate( # Call the new/modified function
                current_internal_infil_state,
                params=self.soil_params,
                rain_rate=current_rain_rate_ms_tensor, # Still element-scalar rain
                surface_head_nodes=self.depth, # Pass the array of depths
                dt=dt_s_tensor,
                cumulative_rain_start=cumulative_rain_m_start_tensor
            )
        
        self.theta_current.copy_(updated_element_state.theta_current)
        self.F_cumulative.copy_(updated_element_state.F_cumulative)
        self.drying_cumulative.copy_(updated_element_state.drying_cumulative)

        # --- Lateral Inflow for Solver (from direct precipitation on channel surface) ---
        rain_rate_nodes = current_rain_rate_ms_tensor.expand_as(infil_rate_NODES)
        net_rain_rate_nodes = torch.clamp(rain_rate_nodes - infil_rate_NODES, min=0.0) # 
        
        if self.props.num_nodes > 0:
            avg_h_for_tw = self.depth.mean() if self.props.num_nodes > 0 else \
                           torch.tensor(0.0, device=self.device, dtype=self.dtype)
            
            # props.W0_nodes is a tensor for channels
            w0_for_tw_calc = self.props.W0_nodes[0] if self.props.num_nodes == 1 else self.props.W0_nodes

            avg_tw = get_trapezoid_topwidth_from_h(avg_h_for_tw, w0_for_tw_calc, 
                                                  self.props.SS1, self.props.SS2) 
            q_lat_nodes_precip = net_rain_rate_nodes * avg_tw 
            q_lat_nodes_precip_expanded = q_lat_nodes_precip.expand(self.props.num_nodes)
        else:
            q_lat_nodes_precip_expanded = torch.empty(0, device=self.device, dtype=self.dtype)

        # --- Channel Flow Solver (MUSCL) ---
        # Solvers require num_nodes >= 2. Handle n_nodes < 2 gracefully.
        if self.props.num_nodes >= 2:
            dx_at_nodes = torch.zeros_like(self.area)
            if self.props.num_nodes > 2: # for interior nodes
                dx_at_nodes[1:-1] = (self.props.dx_segments[:-1] + self.props.dx_segments[1:]) / 2.
                # Boundary nodes
                dx_at_nodes[0] = self.props.dx_segments[0]
                dx_at_nodes[-1] = self.props.dx_segments[-1]
            
            elif self.props.num_nodes == 2:
                # Boundary nodes
                dx_at_nodes[0] = self.props.dx_segments
                dx_at_nodes[-1] = self.props.dx_segments
                
            dx_segment_tensor_for_cfl = dx_at_nodes # Shape (num_nodes,)

            A_next = explicit_muscl_yu_duan_with_plane_contrib(
                A_curr=self.area, 
                q_lat_nodes_prcp=q_lat_nodes_precip_expanded, # Net rain * top_width (m^2/s)
                element_props=self.props, 
                dt=dt_s_tensor, 
                upstream_Q=upstream_q_total_tensor, 
                plane_lat_Q_total=plane_lateral_q_total_tensor
            )
            # Update flow state buffers
            self.area.copy_(torch.clamp(A_next, min=0.0))
            self.depth.copy_(get_h_from_trapezoid_area(self.area, self.props.W0_nodes, 
                                                      self.props.SS1, self.props.SS2))
            wp_next = get_trapezoid_wp_from_h(self.depth, self.props.W0_nodes, 
                                              self.props.SS1, self.props.SS2)
            self.discharge.copy_(calculate_q_manning(self.area, wp_next, self.props.MAN, self.props.SL))
            
            outflow_q_tensor = self.discharge[-1]
        elif self.props.num_nodes == 1: # Simplified storage update for 1-node channel
             # Inflow = upstream_Q + plane_lateral_Q (total) + q_lat_nodes_precip_expanded[0] * LEN
             # Outflow = Q_manning(A_curr)
             # d(Area*LEN)/dt = Inflow - Outflow
             q_in_total = upstream_q_total_tensor + plane_lateral_q_total_tensor + \
                          (q_lat_nodes_precip_expanded[0] * self.props.LEN if q_lat_nodes_precip_expanded.numel()>0 else torch.tensor(0.0, device=self.device, dtype=self.dtype))
             
             # Current outflow (approximate for single cell)
             wp_curr = get_trapezoid_wp_from_h(self.depth, self.props.W0_nodes, self.props.SS1, self.props.SS2)
             q_out_curr = calculate_q_manning(self.area, wp_curr, self.props.MAN, self.props.SL)

             delta_A = (q_in_total - q_out_curr) * (dt_s_tensor / self.props.LEN) if self.props.LEN > 0 else torch.tensor(0.0, device=self.device, dtype=self.dtype)
             self.area.copy_(torch.clamp(self.area + delta_A, min=0.0))
             self.depth.copy_(get_h_from_trapezoid_area(self.area, self.props.W0_nodes, self.props.SS1, self.props.SS2))
             wp_next = get_trapezoid_wp_from_h(self.depth, self.props.W0_nodes, self.props.SS1, self.props.SS2)
             self.discharge.copy_(calculate_q_manning(self.area, wp_next, self.props.MAN, self.props.SL))
             outflow_q_tensor = self.discharge[-1]

             dx_segment_tensor_for_cfl = self.props.LEN

        else: # num_nodes == 0
            outflow_q_tensor = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        self.t_elapsed.copy_(self.t_elapsed + dt_s_tensor)

        current_cfl_numbers_nodes = calculate_cfl_number(
                self.discharge, self.area, dx_segment_tensor_for_cfl, dt_s_tensor
            )
        max_cfl_this_step = torch.max(current_cfl_numbers_nodes) if current_cfl_numbers_nodes.numel() > 0 else \
              torch.tensor(0.0, device=self.device, dtype=self.dtype)
        
        # Update the element's all-time max Froude number
        self.max_cfl.copy_(
                torch.maximum(self.max_cfl, max_cfl_this_step)
            )

        if self.props.num_nodes > 0:
            infil_rate_element = torch.mean(infil_rate_NODES) 
            infil_depth_element = torch.mean(infil_depth_NODES) 
        else:
            infil_rate_element = torch.tensor(0.0, device=self.device)
            infil_depth_element = torch.tensor(0.0, device=self.device)

        return outflow_q_tensor, infil_rate_element, infil_depth_element