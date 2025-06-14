# src/core/infiltration.py

import os
# os.environ['KMP_DUPLICATE_LIB_OK']='TRUE' # Keep if needed for your environment
import torch
from src.data_structures import SoilPropertiesIntermediate, InfiltrationState

# --- Constants ---
EPSILON = 1e-9

# --- Intermediate Infiltration Function ---

def infiltration_step_intermediate_(
    state: InfiltrationState,
    params: SoilPropertiesIntermediate, # Contains tensors for soil properties
    rain_rate: torch.Tensor,     # Scalar tensor (m/s)
    surface_head: torch.Tensor,  # Scalar tensor (m)
    dt: torch.Tensor,            # CHANGED: Now a scalar tensor (s)
    cumulative_rain_start: torch.Tensor # CHANGED: Now a scalar tensor (m)
    ) -> tuple[InfiltrationState, torch.Tensor, torch.Tensor]:
    """
    Calculates infiltration using a modified Green-Ampt style method
    where effective capillary head depends on current soil moisture.

    Args:
        state (InfiltrationState): Current infiltration state (theta_current, F_cumulative).
                                   (Expected to contain tensors on the correct device/dtype).
        params (SoilPropertiesIntermediate): Soil properties (Expected to contain tensors).
        rain_rate (torch.Tensor): Input rainfall rate for the timestep (m/s).
        surface_head (torch.Tensor): Water depth on the surface at the start of the timestep (m).
        dt (torch.Tensor): Timestep duration (s) as a 0-dim tensor.
        cumulative_rain_start (torch.Tensor): Cumulative rainfall depth (m) supplied to the
                                                system up to the beginning of this timestep,
                                                as a 0-dim tensor.

    Returns:
        tuple[InfiltrationState, torch.Tensor, torch.Tensor]: A tuple containing:
            - updated_state (InfiltrationState): The new infiltration state.
            - infiltration_rate (torch.Tensor): Avg infiltration rate for the timestep (m/s), scalar tensor.
            - infiltration_depth (torch.Tensor): Total infiltration depth for the timestep (m), scalar tensor.
    """
    # state.theta_current and state.F_cumulative are already tensors.
    # rain_rate and surface_head are passed in as tensors.
    # dt and cumulative_rain_start are now also passed as tensors.
    
    theta_current = state.theta_current
    F_cumulative = state.F_cumulative
    drying_cumulative = state.drying_cumulative

    # --- 1. Calculate Wetting Front Depth (ZF) ---
    delta_theta_total = torch.clamp(params.theta_s - params.theta_init_condition, min=EPSILON)
    ZF = torch.clamp(F_cumulative / delta_theta_total, min=EPSILON)

    # --- 2. Calculate Effective Capillary Head (HF_eff) ---
    delta_theta_range = torch.clamp(params.theta_s - params.theta_r, min=EPSILON)
    effective_saturation = torch.clamp((theta_current - params.theta_r) / delta_theta_range, min=0.0, max=1.0)
    HF_eff = params.HF_max * torch.pow(1.0 - effective_saturation + EPSILON, params.m_exponent)
    HF_eff = torch.clamp(HF_eff, min=0.0) 
    # HF_eff = params.HF_max

    # --- 3. Calculate Infiltration Capacity (f_cap_depth_potential) ---
    h_surf_safe = torch.clamp(surface_head, min=0.0)
    f_potential_rate = params.Ks * (1.0 + (HF_eff + h_surf_safe) / ZF) # 
    f_potential_rate = torch.clamp(f_potential_rate, min=0.0)
    f_cap_depth_potential = f_potential_rate * dt  # 

    # --- 4. Determine Actual Infiltration (supply or capacity limited) ---
    available_water_depth = rain_rate * dt + h_surf_safe # USE dt
    infiltration_supply_limited = torch.min(available_water_depth, f_cap_depth_potential)
    infiltration_supply_limited = torch.clamp(infiltration_supply_limited, min=0.0)

    # --- 5. Enforce Cumulative Water Balance ---
    cumulative_rain_end = cumulative_rain_start + rain_rate * dt # 
    total_water_supply = cumulative_rain_end + h_surf_safe # h_surf_safe from surface_head
    
    cumulative_increase = total_water_supply - F_cumulative # 
    cumulative_increase = torch.clamp(cumulative_increase, min=0.0)

    infiltration_depth = torch.min(infiltration_supply_limited, cumulative_increase)
    infiltration_depth = torch.clamp(infiltration_depth, min=0.0)

    # --- 6. Update Soil Moisture State (theta_current) ---
    potential_delta_theta_wetting = infiltration_depth / torch.clamp(params.effective_depth, min=EPSILON)
    potential_theta_after_wetting = theta_current + potential_delta_theta_wetting

    excess_theta_above_s = torch.clamp(potential_theta_after_wetting - params.theta_s, min=0.0)
    rejected_infiltration_depth = excess_theta_above_s * params.effective_depth

    # Actual infiltration depth that is accepted by the soil column
    actual_infiltration_depth_into_soil = torch.clamp(infiltration_depth - rejected_infiltration_depth, min=0.0)
    infiltration_rate = actual_infiltration_depth_into_soil / dt # USE dt

    delta_theta_wetting_actual = actual_infiltration_depth_into_soil / torch.clamp(params.effective_depth, min=EPSILON)
    theta_after_wetting = theta_current + delta_theta_wetting_actual

    delta_theta_drying = params.k_drain * torch.clamp(theta_after_wetting - params.theta_r, min=0.0) * dt # USE dt
    theta_next = theta_after_wetting - delta_theta_drying
    theta_next_clamped = torch.clamp(theta_next, min=params.theta_r, max=params.theta_s)

    # --- 7. Update Cumulative Infiltration ---
    F_cumulative_next = F_cumulative + actual_infiltration_depth_into_soil
    drying_next = drying_cumulative + delta_theta_drying 

    # --- 8. Assemble Updated State ---
    updated_state = InfiltrationState(
        theta_current=theta_next_clamped,
        F_cumulative=F_cumulative_next,
        drying_cumulative=drying_next
    )

    return updated_state, infiltration_rate, actual_infiltration_depth_into_soil

def infiltration_step_intermediate(
    state: InfiltrationState,
    params: SoilPropertiesIntermediate, # Contains tensors for soil properties
    rain_rate: torch.Tensor,     # Scalar tensor for the element (m/s)
    surface_head_nodes: torch.Tensor,  # 1D tensor for head at each node (m)
    dt: torch.Tensor,            # Scalar tensor (s)
    cumulative_rain_start: torch.Tensor # Scaler tensor now # 1D tensor for cumulative rain at each node (m)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
    # Returns: new_theta_avg, new_F_avg, infil_rate_nodes (array), infil_depth_nodes (array)
    """
    Calculates infiltration using a modified Green-Ampt style method
    where effective capillary head depends on current soil moisture.

    Args:
        state (InfiltrationState): Current infiltration state (theta_current, F_cumulative).
                                   (Expected to contain tensors on the correct device/dtype).
        params (SoilPropertiesIntermediate): Soil properties (Expected to contain tensors).
        rain_rate (torch.Tensor): Input rainfall rate for the timestep (m/s).
        surface_head (torch.Tensor): Water depth on the surface at the start of the timestep (m).
        dt (torch.Tensor): Timestep duration (s) as a 0-dim tensor.
        cumulative_rain_start (torch.Tensor): Cumulative rainfall depth (m) supplied to the
                                                system up to the beginning of this timestep,
                                                as a 0-dim tensor.

    Returns:
        tuple[new_theta_avg, new_F_avg, infil_rate_nodes (array), infil_depth_nodes (array)]: A tuple containing:
            - new_theta_avg: The new theta averaged over the nodes of the element.
            - new_F_avg: The new cumulative infiltration averaged over the nodes of the element.
            - infil_rate_nodes (torch.Tensor): Avg infiltration rate for the timestep at each node (m/s).
            - infil_depth_nodes (torch.Tensor): Total infiltration depth for the timestep at each node (m).
    """

    theta_current = state.theta_current
    F_cumulative = state.F_cumulative
    drying_cumulative = state.drying_cumulative

    num_nodes = surface_head_nodes.numel()
    if num_nodes == 0: # Handle elements with no nodes
        return (theta_current, F_cumulative, 
                torch.empty(0, device=dt.device, dtype=dt.dtype),
                torch.empty(0, device=dt.device, dtype=dt.dtype))

    # --- 1. Calculate Wetting Front Depth (ZF) ---
    delta_theta_total = torch.clamp(params.theta_s - params.theta_init_condition, min=EPSILON)
    ZF = torch.clamp(F_cumulative / delta_theta_total, min=EPSILON)

    # --- 2. Calculate Effective Capillary Head (HF_eff) ---
    delta_theta_range = torch.clamp(params.theta_s - params.theta_r, min=EPSILON)
    effective_saturation = torch.clamp(
        (theta_current - params.theta_r) / delta_theta_range, min=EPSILON, max=1.0)
    HF_eff = params.HF_max * torch.pow(1.0 - effective_saturation + EPSILON, params.m_exponent)
    HF_eff = torch.clamp(HF_eff, min=0.0) 
    # HF_eff = params.HF_max

    # --- 3. Calculate Infiltration Capacity (f_cap_depth_potential) ---
    h_surf_safe_nodes = torch.clamp(surface_head_nodes, min=0.0)
    f_potential_rate_nodes = params.Ks * (1.0 + (HF_eff + h_surf_safe_nodes) / ZF) # 
    f_potential_rate_nodes = torch.clamp(f_potential_rate_nodes, min=0.0)
    f_cap_depth_potential_nodes = f_potential_rate_nodes * dt  # 

    # --- 4. Determine Actual Infiltration (supply or capacity limited) ---
    available_water_depth_nodes = rain_rate * dt + h_surf_safe_nodes # USE dt
    infiltration_supply_limited_nodes = torch.min(available_water_depth_nodes, f_cap_depth_potential_nodes)
    infiltration_supply_limited_nodes = torch.clamp(infiltration_supply_limited_nodes, min=0.0)

    # --- 5. Enforce Cumulative Water Balance ---
    avg_infiltration_supply_limited = torch.mean(infiltration_supply_limited_nodes) # element average
    cumulative_rain_end = cumulative_rain_start + rain_rate * dt #
    avg_h_surf_safe = torch.mean(h_surf_safe_nodes) # element average
    total_water_supply = cumulative_rain_end + avg_h_surf_safe # h_surf_safe from surface_head

    cumulative_increase = total_water_supply - F_cumulative #
    cumulative_increase = torch.clamp(cumulative_increase, min=0.0)

    infiltration_depth_element = torch.min(infiltration_supply_limited_nodes, cumulative_increase)
    infiltration_depth_element = torch.clamp(infiltration_depth_element, min=0.0)

    # Distribute infiltration depth to nodes: Proportional to supply limited or cumulative increase
    sum_infiltration_supply_limited_nodes = torch.sum(infiltration_supply_limited_nodes)
    if sum_infiltration_supply_limited_nodes > EPSILON:
        infiltration_depth_nodes = infiltration_supply_limited_nodes * \
            (infiltration_depth_element / (avg_infiltration_supply_limited + EPSILON)) 
    else:
        infiltration_depth_nodes = torch.zeros_like(infiltration_supply_limited_nodes)
    
    zero_tensor = torch.zeros_like(infiltration_depth_nodes)
    infiltration_depth_nodes = torch.clamp(infiltration_depth_nodes, 
                                           min=zero_tensor, max=available_water_depth_nodes) # 
    infiltration_rate_nodes = infiltration_depth_nodes / dt

    # --- 6. Update Soil Moisture State (theta_current) ---
    actual_infiltration_depth_element = torch.mean(infiltration_depth_nodes) # Element average
    delta_theta_wetting = actual_infiltration_depth_element / torch.clamp(params.effective_depth, min=EPSILON)
    theta_after_wetting = theta_current + delta_theta_wetting

    delta_theta_drying = params.k_drain * torch.clamp(theta_after_wetting - params.theta_r, min=0.0) * dt # USE dt
    theta_next = theta_after_wetting - delta_theta_drying
    theta_next_clamped = torch.clamp(theta_next, min=params.theta_r, max=params.theta_s)

    # --- 7. Update Cumulative Infiltration ---
    F_cumulative_next = F_cumulative + actual_infiltration_depth_element
    drying_next = drying_cumulative + delta_theta_drying 

    # --- 8. Assemble Updated State ---
    updated_state = InfiltrationState(
        theta_current=theta_next_clamped,
        F_cumulative=F_cumulative_next,
        drying_cumulative=drying_next
    )

    return updated_state, infiltration_rate_nodes, infiltration_depth_nodes


## Test
# import matplotlib.pyplot as plt 