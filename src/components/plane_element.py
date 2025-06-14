# src/components/plane_element.py
import torch
import torch.nn as nn

# Import from our defined structures and core physics/solvers
from src.data_structures import ElementProperties, SoilPropertiesIntermediate, OverlandFlowState, InfiltrationState
from src.core.infiltration import infiltration_step_intermediate
from src.core.kinematic_wave_solvers import explicit_step_lax_friedrichs # Specific solver for planes
from src.core.kinematic_wave_solvers import explicit_muscl_yu_duan_with_plane_contrib
from src.core.physics_formulas import (get_plane_h_from_area, calculate_q_manning, calculate_froude_number,
                                       calculate_cfl_number, get_trapezoid_wp_from_h)    

# @torch.compile
class PlaneElement_(nn.Module):
    def __init__(self, 
                 element_props: ElementProperties, 
                 soil_params: SoilPropertiesIntermediate,
                 device: torch.device, # device and dtype are now primarily for buffer initialization
                 dtype: torch.dtype):
        super().__init__()

        # --- Store Properties and Parameters ---
        # These are assumed to be already on the correct device/dtype with requires_grad set
        # by the basin_loader.parse_element_override function.
        self.props: ElementProperties = element_props
        self.soil_params: SoilPropertiesIntermediate = soil_params
        
        # Store device and dtype for internal use if needed, though states will inherit from inputs
        self.device = device
        self.dtype = dtype

        # --- Initialize State Buffers ---
        # Buffers are part of the module's state_dict, move with .to(device), but are not parameters.
        
        # Flow State
        # Initial area, depth, discharge are zeros. t_elapsed is zero.
        self.register_buffer('area', torch.zeros(self.props.num_nodes, device=self.device, dtype=self.dtype))
        self.register_buffer('depth', torch.zeros(self.props.num_nodes, device=self.device, dtype=self.dtype))
        self.register_buffer('discharge', torch.zeros(self.props.num_nodes, device=self.device, dtype=self.dtype))
        self.register_buffer('t_elapsed', torch.tensor(0.0, device=self.device, dtype=self.dtype))
        self.register_buffer('max_cfl', torch.tensor(0.0, device=device, dtype=dtype))

        # Infiltration State
        # Initial theta_current is from soil_params.theta_init_condition. F_cumulative is zero.
        # Ensure soil_params.theta_init_condition is a tensor. It should be from basin_loader.
        init_theta = self.soil_params.theta_init_condition.clone().detach() # Clone for safety
        self.register_buffer('theta_current', init_theta)
        self.register_buffer('F_cumulative', torch.tensor(0.0, device=self.device, dtype=self.dtype))
        self.register_buffer('drying_cumulative', torch.tensor(0.0, device=self.device, dtype=self.dtype))

    def forward(self, 
                current_rain_rate_ms_tensor: torch.Tensor, # Scalar tensor
                dt_s_tensor: torch.Tensor,                 # Scalar tensor
                cumulative_rain_m_start_tensor: torch.Tensor # Scalar tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs one time step update for the plane element.
        Updates internal state buffers.

        Args:
            current_rain_rate_ms_tensor (torch.Tensor): Rainfall rate for this step (m/s).
            dt_s_tensor (torch.Tensor): Timestep duration (s).
            cumulative_rain_m_start_tensor (torch.Tensor): Cumulative rain up to start of step (m).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - outflow_q_tensor (torch.Tensor): Discharge from the last node (m^3/s), scalar tensor.
                - infil_rate_tensor_ms (torch.Tensor): Infiltration rate for the step (m/s), scalar tensor.
                - infil_depth_tensor_m (torch.Tensor): Infiltration depth for the step (m), scalar tensor.
        """
        # Current states from buffers
        current_internal_infil_state = InfiltrationState(
            theta_current=self.theta_current, 
            F_cumulative=self.F_cumulative,
            drying_cumulative=self.drying_cumulative
        )
        
        # --- Infiltration ---
        head_input_for_infil = self.depth.mean() if self.props.num_nodes > 0 else \
                               torch.tensor(0.0, device=self.device, dtype=self.dtype)

        updated_infil_state, infil_rate_tensor_ms, infil_depth_tensor_m = infiltration_step_intermediate(
            state=current_internal_infil_state, 
            params=self.soil_params, 
            rain_rate=current_rain_rate_ms_tensor,
            surface_head=head_input_for_infil, 
            dt=dt_s_tensor, 
            cumulative_rain_start=cumulative_rain_m_start_tensor
        )
        # Update infiltration state buffers
        self.theta_current.copy_(updated_infil_state.theta_current) # In-place update of buffer
        self.F_cumulative.copy_(updated_infil_state.F_cumulative)   # In-place update
        self.drying_cumulative.copy_(updated_infil_state.drying_cumulative) # Update cumulative drying 

        # --- Lateral Inflow for Solver (from net precipitation) ---
        # q_lat_nodes_precip is per unit length (m^2/s)
        net_rain_rate_tensor = torch.clamp(current_rain_rate_ms_tensor - infil_rate_tensor_ms, min=0.0)
        
        # Calculate lateral inflow at nodes
        q_lat_nodes_precip = net_rain_rate_tensor * self.props.WID 
        
        if self.props.num_nodes > 0:
            q_lat_nodes_precip_expanded = q_lat_nodes_precip.repeat(self.props.num_nodes)
        else: # Should ideally not happen for a simulated element
            q_lat_nodes_precip_expanded = torch.empty(0, device=self.device, dtype=self.dtype)

        zero_upstream_q_tensor = torch.tensor(0.0, device=self.device, dtype=self.dtype)


        # --- Overland Flow Solver () ---
        # Solvers require num_nodes >= 2 typically. If num_nodes < 2, handle gracefully.
        if self.props.num_nodes >= 2:
            # dx_at_nodes = torch.zeros_like(self.area)
            # if self.props.num_nodes > 2: # for interior nodes
            #      dx_at_nodes[1:-1] = (self.props.dx_segments[:-1] + self.props.dx_segments[1:]) / 2.
            # # Boundary nodes
            # dx_at_nodes[0] = self.props.dx_segments[0]
            # dx_at_nodes[-1] = self.props.dx_segments[-1]
            # dx_tensor_for_cfl = dx_at_nodes # Shape (num_nodes,)
            # A_next = explicit_step_lax_friedrichs(
            #     A_curr=self.area, 
            #     q_lat_nodes=q_lat_nodes_precip_expanded, 
            #     element_props=self.props, 
            #     dt=dt_s_tensor, 
            #     upstream_Q=zero_upstream_q_tensor
            # )
            dx_tensor_for_cfl = self.props.dx_avg * torch.ones_like(self.area, device=self.device, dtype=self.dtype)
            A_next = explicit_muscl_yu_duan_with_plane_contrib(
                A_curr=self.area, 
                q_lat_nodes_prcp=q_lat_nodes_precip_expanded, # Net rain * top_width (m^2/s)
                element_props=self.props, 
                dt=dt_s_tensor, 
                upstream_Q=zero_upstream_q_tensor, 
                plane_lat_Q_total=zero_upstream_q_tensor
            )
            
            # Update flow state buffers
            self.area.copy_(torch.clamp(A_next, min=0.0)) # In-place update
            self.depth.copy_(get_plane_h_from_area(self.area, self.props.WID)) # WID is tensor
            
            wp_next = torch.full_like(self.area, self.props.WID.item()) 
            self.discharge.copy_(calculate_q_manning(self.area, wp_next, self.props.MAN, self.props.SL))
            
            outflow_q_tensor = self.discharge[-1]
        elif self.props.num_nodes == 1: # Handle 1-node elements with a simple storage update
            # Total lateral inflow volume rate = q_lat_nodes_precip (m^2/s) * LEN (m)
            # Change in storage = Inflow_vol - Outflow_vol
            # Storage = Area_avg * LEN. Assume self.area[0] is Area_avg
            # A_next = A_old + (dt/LEN) * (q_lat_nodes_precip * LEN - Q_out_old)
            # Simplified: dA/dt = q_lat_nodes_precip (if Q_out is handled by scheme implicitly)
            # For a single cell, explicit mass balance:
            # (A_next * WID * dx_avg - A_curr * WID * dx_avg) / dt = q_lat_nodes_precip * WID - Q_out
            # This is complex if dx_avg != LEN.
            # Let's use the solver's n_nodes=1 logic if it's robust, or simplify.
            # For now, if explicit_step_lax_friedrichs handles n_nodes=1, let it.
            # But its n_nodes=1 case was: A_next = A_curr + dt * q_lat_nodes
            # which uses q_lat_nodes (per unit length). Needs q_lat_nodes_precip_expanded[0]
            if q_lat_nodes_precip_expanded.numel() > 0:
                 self.area.copy_(torch.clamp(self.area + dt_s_tensor * q_lat_nodes_precip_expanded[0], min=0.0))
            self.depth.copy_(get_plane_h_from_area(self.area, self.props.WID))
            wp_next = torch.full_like(self.area, self.props.WID)
            self.discharge.copy_(calculate_q_manning(self.area, wp_next, self.props.MAN, self.props.SL))
            outflow_q_tensor = self.discharge[-1] # Indexing will work for single element tensor

            dx_tensor_for_cfl = self.props.LEN

        else: # num_nodes == 0
            outflow_q_tensor = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            # States (area, depth, discharge) remain empty or zeros

        current_cfl_numbers_nodes = calculate_cfl_number(self.discharge, self.area, dx_tensor_for_cfl, dt_s_tensor)
        max_cfl_this_step = torch.max(current_cfl_numbers_nodes) if current_cfl_numbers_nodes.numel() > 0 else \
                                   torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.max_cfl.copy_(torch.maximum(self.max_cfl, max_cfl_this_step))
        self.t_elapsed.copy_(self.t_elapsed + dt_s_tensor) # In-place update

        return outflow_q_tensor, infil_rate_tensor_ms, infil_depth_tensor_m

class PlaneElement(nn.Module):
    def __init__(self, 
                 element_props: ElementProperties, 
                 soil_params: SoilPropertiesIntermediate,
                 device: torch.device, # device and dtype are now primarily for buffer initialization
                 dtype: torch.dtype):
        super().__init__()

        # --- Store Properties and Parameters ---
        # These are assumed to be already on the correct device/dtype with requires_grad set
        # by the basin_loader.parse_element_override function.
        self.props: ElementProperties = element_props
        self.soil_params: SoilPropertiesIntermediate = soil_params
        
        # Store device and dtype for internal use if needed, though states will inherit from inputs
        self.device = device
        self.dtype = dtype

        # --- Initialize State Buffers ---
        # Buffers are part of the module's state_dict, move with .to(device), but are not parameters.
        
        # Flow State
        # Initial area, depth, discharge are zeros. t_elapsed is zero.
        self.register_buffer('area', torch.zeros(self.props.num_nodes, device=self.device, dtype=self.dtype))
        self.register_buffer('depth', torch.zeros(self.props.num_nodes, device=self.device, dtype=self.dtype))
        self.register_buffer('discharge', torch.zeros(self.props.num_nodes, device=self.device, dtype=self.dtype))
        self.register_buffer('t_elapsed', torch.tensor(0.0, device=self.device, dtype=self.dtype))
        self.register_buffer('max_cfl', torch.tensor(0.0, device=device, dtype=dtype))

        # Infiltration State
        # Initial theta_current is from soil_params.theta_init_condition. F_cumulative is zero.
        # Ensure soil_params.theta_init_condition is a tensor. It should be from basin_loader.
        init_theta = self.soil_params.theta_init_condition.clone().detach() # Clone for safety
        self.register_buffer('theta_current', init_theta)
        self.register_buffer('F_cumulative', torch.tensor(0.0, device=self.device, dtype=self.dtype))
        self.register_buffer('drying_cumulative', torch.tensor(0.0, device=self.device, dtype=self.dtype))

    def forward(self, 
                current_rain_rate_ms_tensor: torch.Tensor, # Scalar tensor
                dt_s_tensor: torch.Tensor,                 # Scalar tensor
                cumulative_rain_m_start_tensor: torch.Tensor # Scalar tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs one time step update for the plane element.
        Updates internal state buffers.

        Args:
            current_rain_rate_ms_tensor (torch.Tensor): Rainfall rate for this step (m/s).
            dt_s_tensor (torch.Tensor): Timestep duration (s).
            cumulative_rain_m_start_tensor (torch.Tensor): Cumulative rain up to start of step (m).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - outflow_q_tensor (torch.Tensor): Discharge from the last node (m^3/s), scalar tensor.
                - infil_rate_tensor_ms (torch.Tensor): Infiltration rate for the step (m/s), scalar tensor.
                - infil_depth_tensor_m (torch.Tensor): Infiltration depth for the step (m), scalar tensor.
        """
        # Current states from buffers
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
        
        # Update infiltration state buffers
        self.theta_current.copy_(updated_element_state.theta_current) # In-place update of buffer
        self.F_cumulative.copy_(updated_element_state.F_cumulative)   # In-place update
        self.drying_cumulative.copy_(updated_element_state.drying_cumulative) # Update cumulative drying 

        rain_depth_nodes = current_rain_rate_ms_tensor.expand_as(infil_rate_NODES) * dt_s_tensor
        # net_rain_depth_nodes = rain_depth_nodes - infil_depth_NODES, min=0.0)
        surface_head = torch.clamp(self.depth +  rain_depth_nodes - infil_depth_NODES, min=0.0)
        self.depth.copy_(surface_head)
        self.area.copy_(self.depth * self.props.WID.expand_as(self.depth)); # print(self.area)

        zero_lat_q_tensor = torch.tensor(0, device=self.device, dtype=self.dtype)
        zero_upstream_q_tensor = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        # --- Overland Flow Solver () ---
        # Solvers require num_nodes >= 2 typically. If num_nodes < 2, handle gracefully.
        if self.props.num_nodes >= 2:
            dx_tensor_for_cfl = self.props.dx_avg * torch.ones_like(self.area, device=self.device, dtype=self.dtype)

            A_next = explicit_muscl_yu_duan_with_plane_contrib(
                A_curr=self.area, 
                q_lat_nodes_prcp=zero_lat_q_tensor, # zero for plane elements
                element_props=self.props, 
                dt=dt_s_tensor, 
                upstream_Q=zero_upstream_q_tensor, 
                plane_lat_Q_total=zero_upstream_q_tensor
            )
            
            # Update flow state buffers
            self.area.copy_(torch.clamp(A_next, min=0.0)) # In-place update
            self.depth.copy_(get_plane_h_from_area(self.area, self.props.WID)) # WID is tensor
            
            # wp_next = torch.full_like(self.area, self.props.WID.item()) # not adding depth to wp_next
            wp_next = get_trapezoid_wp_from_h(self.depth, self.props.WID, self.props.SS1, self.props.SS2)
            self.discharge.copy_(calculate_q_manning(self.area, wp_next, self.props.MAN, self.props.SL))
            
            outflow_q_tensor = self.discharge[-1]
        

        else: # num_nodes == 0
            outflow_q_tensor = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            # States (area, depth, discharge) remain empty or zeros

        current_cfl_numbers_nodes = calculate_cfl_number(self.discharge, self.area, dx_tensor_for_cfl, dt_s_tensor)
        max_cfl_this_step = torch.max(current_cfl_numbers_nodes) if current_cfl_numbers_nodes.numel() > 0 else \
                                   torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.max_cfl.copy_(torch.maximum(self.max_cfl, max_cfl_this_step))
        self.t_elapsed.copy_(self.t_elapsed + dt_s_tensor) # In-place update

        if self.props.num_nodes > 0:
            infil_rate_element = torch.mean(infil_rate_NODES) 
            infil_depth_element = torch.mean(infil_depth_NODES) 
        else:
            infil_rate_element = torch.tensor(0.0, device=self.device)
            infil_depth_element = torch.tensor(0.0, device=self.device)

        return outflow_q_tensor, infil_rate_element, infil_depth_element