# src/core/kinematic_wave_solvers.py

import torch
import sys
# from collections import namedtuple # Not needed here

# Import necessary functions and types from other modules
from src.core.physics_formulas import (calculate_q_manning, calculate_dqda_manning_general, 
                                       get_trapezoid_area_from_h, get_plane_area_from_h, 
                                       van_leer_limiter, get_h_from_trapezoid_area,
                                       get_trapezoid_wp_from_h, get_trapezoid_topwidth_from_h, 
                                       get_plane_h_from_area)
from src.data_structures import ElementProperties # Make sure this path is correct

# --- Configuration ---
EPSILON = 1e-9

# --- Explicit Solvers (Operating on Area A) ---

def explicit_step_lax_friedrichs(
    A_curr: torch.Tensor,
    q_lat_nodes: torch.Tensor,
    element_props: ElementProperties,
    dt: torch.Tensor,         # expects a 0-dim tensor
    upstream_Q: torch.Tensor  # expects a 0-dim tensor
    ) -> torch.Tensor:
    """
    Lax-Friedrichs scheme for dA/dt + dQ/dx = q_lat.
    dt and upstream_Q are 0-dim tensors on the same device as A_curr.
    """
    n_nodes = element_props.num_nodes
    
    # --- Handle trivial cases ---
    if n_nodes == 0:
        return torch.tensor([], device=A_curr.device, dtype=A_curr.dtype)
    if n_nodes == 1:
        # Assuming q_lat_nodes is expanded and contains value for node 0
        # A_next = A_curr + dt * q_lat_nodes[0] # If q_lat_nodes[0] is rate for the cell
        # The original LxF n_nodes=1 logic was A_next = A_curr + dt * q_lat_nodes
        # If q_lat_nodes is per unit length, and the node represents the whole element:
        # This implies dA/dt = q_lat_node_0 if element_props.LEN is implicitly 1 here
        # For consistency, it's better that elements have at least 2 nodes for PDE solvers.
        # However, if this case must be handled:
        if q_lat_nodes.numel() > 0: # Ensure q_lat_nodes is not empty
             A_next = A_curr + dt * q_lat_nodes[0]
        else: # Should not happen if n_nodes=1
             A_next = A_curr.clone()
        return torch.clamp(A_next, min=0.0)

    # --- Determine dx characteristics ---
    # dx_segments and other props like WID, SS1, SS2, MAN, SL are already tensors from ElementProperties
    use_variable_dx = (element_props.geom_type == 'plane' and
                       element_props.dx_segments.numel() == n_nodes - 1) # dx_segments is tensor

    dx_avg_tensor = element_props.dx_avg # This is a 0-dim tensor
    dx_segments_tensor = element_props.dx_segments if use_variable_dx else None

    # --- Calculate geometry and Q at all nodes at time n ---
    # Properties like WID, SS1, SS2, MAN, SL are tensors in element_props
    if element_props.geom_type == 'channel':
        W0_nodes_tensor = element_props.W0_nodes # Tensor
        h_curr = get_h_from_trapezoid_area(A_curr, W0_nodes_tensor, element_props.SS1, element_props.SS2)
        wp_curr = get_trapezoid_wp_from_h(h_curr, W0_nodes_tensor, element_props.SS1, element_props.SS2)
        width_ref_for_ghost = W0_nodes_tensor[0] # Use first node's actual W0 (tensor)
    else: # Plane
        h_curr = get_plane_h_from_area(A_curr, element_props.WID) # WID is 0-dim tensor
        wp_curr = element_props.WID.expand_as(A_curr) # Expand WID tensor
        width_ref_for_ghost = element_props.WID # WID tensor

    q_curr = calculate_q_manning(A_curr, wp_curr, element_props.MAN, element_props.SL)
    A_next = A_curr.clone()

    # --- Interior nodes (j from 1 to N-2) ---
    if n_nodes >= 3:
        A_avg_interior = 0.5 * (A_curr[2:] + A_curr[:-2])
        q_diff_interior = q_curr[2:] - q_curr[:-2]
        q_lat_j_interior = q_lat_nodes[1:-1]

        dist_eff_interior = dx_segments_tensor[1:] + dx_segments_tensor[:-1] if use_variable_dx \
                            else (2.0 * dx_avg_tensor)
        
        A_next[1:-1] = A_avg_interior - \
                       (dt / torch.clamp(dist_eff_interior, min=EPSILON)) * q_diff_interior + \
                       dt * q_lat_j_interior

    # --- Upstream Boundary (Node 0) ---
    if n_nodes >= 2:
        A_ghost_L = torch.tensor(0.0, device=A_curr.device, dtype=A_curr.dtype)
        if upstream_Q > EPSILON: # Tensor comparison
            # Use width_ref_for_ghost which is a tensor
            # SL[0] and MAN are already tensors from element_props
            n_safe = torch.clamp(element_props.MAN, min=1e-4)
            s_safe = torch.clamp(element_props.SL[0], min=EPSILON**2)
            w0_safe = torch.clamp(width_ref_for_ghost, min=EPSILON) # This is a tensor
            
            # Approximate h from Q for ghost cell using rectangular assumption first
            h_ghost_rect_approx = torch.pow(
                torch.clamp((upstream_Q * n_safe) / (w0_safe * torch.sqrt(s_safe) + EPSILON), min=0.0),
                3.0/5.0
            )
            if element_props.geom_type == 'channel':
                A_ghost_L = get_trapezoid_area_from_h(h_ghost_rect_approx, w0_safe, 
                                                    element_props.SS1, element_props.SS2)
            else: # Plane
                A_ghost_L = get_plane_area_from_h(h_ghost_rect_approx, w0_safe) # w0_safe is props.WID tensor
            
            A_ghost_L = torch.clamp(A_ghost_L, min=0.0, max=A_curr[0]*5.0 if A_curr[0] > EPSILON else 1.0)

        if n_nodes > 2:
            dx_segment = dx_segments_tensor[0] # First segment
        else: # n_nodes == 2
            dx_segment = dx_segments_tensor 
        dx_eff_0 = (2.0 * dx_segment) if use_variable_dx and dx_segments_tensor.numel() > 0 \
                else (2.0 * dx_avg_tensor)
        
        A_next[0] = 0.5 * (A_curr[1] + A_ghost_L) - \
                    (dt / torch.clamp(dx_eff_0, min=EPSILON)) * (q_curr[1] - upstream_Q) + \
                    dt * q_lat_nodes[0]

    # --- Downstream Boundary (Node N-1) ---
    A_ghost_N = A_curr[-1] # Zero-order extrapolation for Area
    
    # Calculate Q at ghost node N using properties of the last actual node (N-1)
    if element_props.geom_type == 'channel':
        W0_last = W0_nodes_tensor[-1] 
        h_ghost_N = get_h_from_trapezoid_area(A_ghost_N, W0_last, element_props.SS1, element_props.SS2)
        wp_ghost_N = get_trapezoid_wp_from_h(h_ghost_N, W0_last, element_props.SS1, element_props.SS2)
    else: # Plane
        wp_ghost_N = element_props.WID.expand_as(A_ghost_N) # WID tensor expanded
        
    Q_ghost_N = calculate_q_manning(A_ghost_N, wp_ghost_N, element_props.MAN, element_props.SL[-1])

    A_avg_last = 0.5 * (A_ghost_N + A_curr[-2])
    q_diff_last = Q_ghost_N - q_curr[-2]
    q_lat_last = q_lat_nodes[-1]

    if n_nodes == 2:
        dx_eff_last = (2.0 * dx_segments_tensor) if use_variable_dx and dx_segments_tensor.numel() > 0 \
                  else (2.0 * dx_avg_tensor)
    else: # n_nodes > 2
        dx_eff_last = (2.0 * dx_segments_tensor[-1]) if use_variable_dx and dx_segments_tensor.numel() > 0 \
                  else (2.0 * dx_avg_tensor)
    
    A_next[-1] = A_avg_last - \
                 (dt / torch.clamp(dx_eff_last, min=EPSILON)) * q_diff_last + \
                 dt * q_lat_last

    return torch.clamp(A_next, min=0.0)


def explicit_muscl_yu_duan_with_plane_contrib_(
    A_curr: torch.Tensor,
    q_lat_nodes_prcp: torch.Tensor, # Net precip per unit length (m^2/s)
    element_props: ElementProperties,
    dt: torch.Tensor,             # CHANGED to 0-dim tensor
    upstream_Q: torch.Tensor,     # CHANGED to 0-dim tensor (total Q m^3/s)
    plane_lat_Q_total: torch.Tensor # CHANGED to 0-dim tensor (total Q m^3/s from side planes)
    ) -> torch.Tensor:
    """
    MUSCL scheme for dA/dt + dQ/dx = q_lat.
    Uses AVERAGE dx (element_props.dx_avg which is a 0-dim tensor).
    Handles combined lateral inflow from direct precipitation and distributed plane inflow.
    All dynamic scalar inputs (dt, Qs) are 0-dim tensors.
    """
    n_nodes = element_props.num_nodes
    if n_nodes < 2: # MUSCL typically needs at least 2 cells for reconstruction.
        # Fallback or error for < 2 nodes. For simplicity, let's assume n_nodes >=2 here.
        # Or use LxF's n_nodes=1 logic if it must be handled by MUSCL wrapper.
        print(f"Warning: MUSCL solver called for element {element_props.element_id} with {n_nodes} nodes. Requires >=2.")
        if n_nodes == 1 and q_lat_nodes_prcp.numel() > 0 :
             # Simple storage update if only one node
             q_lat_total_for_node = q_lat_nodes_prcp[0] # This is per unit length
             if element_props.geom_type == 'channel' and plane_lat_Q_total > EPSILON:
                 if element_props.LEN > EPSILON: # LEN is tensor
                     q_lat_total_for_node += plane_lat_Q_total / element_props.LEN
             # Add upstream Q if it's a channel and only one node (highly unusual setup)
             # delta_A = (upstream_Q + q_lat_total_for_node * element_props.LEN - Q_out_approx) * dt / element_props.LEN
             # This is becoming too complex for a solver edge case. Best to ensure n_nodes >= 2.
             # For now, mimic LxF's simple update if forced:
             return torch.clamp(A_curr + dt * q_lat_total_for_node, min=0.0)
        return A_curr.clone() # No change or error

    # --- Properties (already tensors from element_props) ---
    dx_tensor = element_props.dx_avg # 0-dim tensor
    slopes_tensor = element_props.SL
    
    if element_props.geom_type == 'channel':
        W0_nodes_tensor = element_props.W0_nodes
        width_ref_for_ghost = W0_nodes_tensor[0] # Upstream actual bottom width
        SS1 = element_props.SS1
        SS2 = element_props.SS2
    else: # Plane
        W0_nodes_tensor = element_props.WID # Use WID as the "bottom width" tensor
        width_ref_for_ghost = element_props.WID
        SS1, SS2 = torch.tensor(0.0, device=W0_nodes_tensor.device), torch.tensor(0.0, device=W0_nodes_tensor.device)

    # --- Combined Lateral Inflow (per unit length, m^2/s) ---
    q_lat_combined_nodes = q_lat_nodes_prcp.clone() # Already per unit length
    if element_props.geom_type == 'channel' and plane_lat_Q_total > EPSILON:
        if element_props.LEN > EPSILON: # LEN is 0-dim tensor
            q_lat_combined_nodes = q_lat_combined_nodes + (plane_lat_Q_total / element_props.LEN)

    # --- Define L(A) operator ---
    def calculate_L_operator(A_state: torch.Tensor) -> torch.Tensor:
        A_state_clamped = torch.clamp(A_state, min=0.0) # Use clamped A for calculations

        # --- 1. Pad A_state with ghost cells ---
        A_ghost_L = torch.tensor(0.0, device=A_state.device, dtype=A_state.dtype)
        if upstream_Q > EPSILON:
            n_safe = torch.clamp(element_props.MAN, min=1e-4)
            s_safe = torch.clamp(slopes_tensor[0], min=EPSILON**2) # Use slope at first node
            w_ghost_ref = torch.clamp(width_ref_for_ghost, min=EPSILON) # This is a tensor

            h_ghost_rect_approx = torch.pow(
                torch.clamp((upstream_Q * n_safe) / (w_ghost_ref * torch.sqrt(s_safe) + EPSILON), min=0.0),
                3.0/5.0
            )
            if element_props.geom_type == 'channel':
                A_ghost_L = get_trapezoid_area_from_h(h_ghost_rect_approx, w_ghost_ref, SS1, SS2)
            else: # Plane
                A_ghost_L = get_plane_area_from_h(h_ghost_rect_approx, w_ghost_ref)
            # A_ghost_L = torch.clamp(A_ghost_L, min=0.0, max=A_state_clamped[0]*5.0 if A_state_clamped[0]>EPSILON else 1.0)
            A_ghost_L = torch.clamp(A_ghost_L, min=0.0)
        
        A_ghost_R = A_state_clamped[-1:] # Zero-order extrapolation for Area
        A_padded = torch.cat((A_ghost_L.unsqueeze(0), A_state_clamped, A_ghost_R))

        # --- 2. Calculate Limited Slopes (van Leer) ---
        # dx_tensor is 0-dim (average dx)
        delta_L = (A_padded[1:-1] - A_padded[:-2]) / element_props.dx_avg 
        delta_R = (A_padded[2:] - A_padded[1:-1]) / element_props.dx_avg
        limited_slope = van_leer_limiter(delta_L, delta_R)

        # --- 3. Reconstruct Interface Values ---
        A_L_recon = A_state_clamped + 0.5 * dx_tensor * limited_slope
        A_R_recon = A_state_clamped - 0.5 * dx_tensor * limited_slope 
        # A_R_jp12 is A_R_recon for cell j+1 at interface j+1/2 (shifted left)
        A_R_interface = torch.cat((A_R_recon[1:], A_R_recon[-1:])) # Uses last cell's A_R for last interface's right state
        
        A_L_interface = torch.clamp(A_L_recon, min=0.0) # This is A_L for cell j at interface j+1/2
        A_R_interface = torch.clamp(A_R_interface, min=0.0)


        # --- 4. Calculate Interface Fluxes Q* (Rusanov / Local Lax-Friedrichs) ---
        if element_props.geom_type == 'channel':
            h_L_iface = get_h_from_trapezoid_area(A_L_interface, W0_nodes_tensor, SS1, SS2)
            wp_L_iface = get_trapezoid_wp_from_h(h_L_iface, W0_nodes_tensor, SS1, SS2)
            h_R_iface = get_h_from_trapezoid_area(A_R_interface, W0_nodes_tensor, SS1, SS2)
            wp_R_iface = get_trapezoid_wp_from_h(h_R_iface, W0_nodes_tensor, SS1, SS2)
            
            # For celerity calculation at cell centers
            h_cell_center = get_h_from_trapezoid_area(A_state_clamped, W0_nodes_tensor, SS1, SS2)
            wp_cell_center = get_trapezoid_wp_from_h(h_cell_center, W0_nodes_tensor, SS1, SS2)
            tw_cell_center = get_trapezoid_topwidth_from_h(h_cell_center, W0_nodes_tensor, SS1, SS2)
        else: # Plane
            # W0_nodes_tensor is props.WID for planes
            h_L_iface = get_plane_h_from_area(A_L_interface, W0_nodes_tensor)
            wp_L_iface = W0_nodes_tensor.expand_as(A_L_interface)
            h_R_iface = get_plane_h_from_area(A_R_interface, W0_nodes_tensor)
            wp_R_iface = W0_nodes_tensor.expand_as(A_R_interface)

            h_cell_center = get_plane_h_from_area(A_state_clamped, W0_nodes_tensor)
            wp_cell_center = W0_nodes_tensor.expand_as(A_state_clamped)
            tw_cell_center = W0_nodes_tensor.expand_as(A_state_clamped) # Topwidth is WID for plane

        q_L_iface = calculate_q_manning(A_L_interface, wp_L_iface, element_props.MAN, slopes_tensor)
        q_R_iface = calculate_q_manning(A_R_interface, wp_R_iface, element_props.MAN, slopes_tensor)
        
        Q_cell_center = calculate_q_manning(A_state_clamped, wp_cell_center, element_props.MAN, slopes_tensor)
        celerity_cell_center = calculate_dqda_manning_general(
            A_state_clamped, wp_cell_center, tw_cell_center, Q_cell_center, 
            SS1, SS2, element_props.geom_type
        )
        
        # Max wave speed at interface j+1/2 based on cells j and j+1
        # Pad celerity for boundary interfaces: use celerity of first cell for first interface, last for last.
        alpha_rusanov = torch.maximum(
            torch.abs(celerity_cell_center), 
            torch.abs(torch.cat((celerity_cell_center[1:], celerity_cell_center[-1:]))) # c_{j+1} approx by c_N for last iface
        )
        q_star_interfaces = 0.5 * (q_L_iface + q_R_iface) - 0.5 * alpha_rusanov * (A_R_interface - A_L_interface)

        # --- 5. Calculate L(A) = - (Q*_{j+1/2} - Q*_{j-1/2})/dx + S_j ---
        # Prepend upstream_Q as the flux at the very first (ghost) interface
        q_star_ghost_L = upstream_Q.reshape(1) # Ensure it's 1D for cat
        q_star_padded_for_diff = torch.cat((q_star_ghost_L, q_star_interfaces))
        
        flux_diff = q_star_padded_for_diff[1:] - q_star_padded_for_diff[:-1]
        L_A = -flux_diff / dx_tensor + q_lat_combined_nodes
        return L_A

    # --- TVD RK2 Time Marching ---
    L_n = calculate_L_operator(A_curr)
    A_1 = A_curr + dt * L_n 
    A_1 = torch.clamp(A_1, min=0.0)
    
    L_1 = calculate_L_operator(A_1)
    A_next = 0.5 * A_curr + 0.5 * (A_1 + dt * L_1)
    
    return torch.clamp(A_next, min=0.0)

def explicit_muscl_yu_duan_with_plane_contrib__(
    A_curr: torch.Tensor,
    q_lat_nodes_prcp: torch.Tensor, # Net precip per unit length (m^2/s)
    element_props: ElementProperties,
    dt: torch.Tensor,             # CHANGED to 0-dim tensor
    upstream_Q: torch.Tensor,     # CHANGED to 0-dim tensor (total Q m^3/s)
    plane_lat_Q_total: torch.Tensor # CHANGED to 0-dim tensor (total Q m^3/s from side planes)
    ) -> torch.Tensor:
    """
    MUSCL scheme for dA/dt + dQ/dx = q_lat.
    Uses AVERAGE dx (element_props.dx_avg which is a 0-dim tensor).
    Handles combined lateral inflow from direct precipitation and distributed plane inflow.
    All dynamic scalar inputs (dt, Qs) are 0-dim tensors.
    """
    n_nodes = element_props.num_nodes
    # if n_nodes < 2: # MUSCL typically needs at least 2 cells for reconstruction.
    #     # Fallback or error for < 2 nodes. For simplicity, let's assume n_nodes >=2 here.
    #     # Or use LxF's n_nodes=1 logic if it must be handled by MUSCL wrapper.
    #     print(f"Warning: MUSCL solver called for element {element_props.element_id} with {n_nodes} nodes. Requires >=2.")
    #     if n_nodes == 1 and q_lat_nodes_prcp.numel() > 0 :
    #          # Simple storage update if only one node
    #          q_lat_total_for_node = q_lat_nodes_prcp[0] # This is per unit length
    #          if element_props.geom_type == 'channel' and plane_lat_Q_total > EPSILON:
    #              if element_props.LEN > EPSILON: # LEN is tensor
    #                  q_lat_total_for_node += plane_lat_Q_total / element_props.LEN
    #          # Add upstream Q if it's a channel and only one node (highly unusual setup)
    #          # delta_A = (upstream_Q + q_lat_total_for_node * element_props.LEN - Q_out_approx) * dt / element_props.LEN
    #          # This is becoming too complex for a solver edge case. Best to ensure n_nodes >= 2.
    #          # For now, mimic LxF's simple update if forced:
    #          return torch.clamp(A_curr + dt * q_lat_total_for_node, min=0.0)
    #     return A_curr.clone() # No change or error

    # --- Properties (already tensors from element_props) ---
    dx_tensor = element_props.dx_avg # 0-dim tensor
    slopes_tensor = element_props.SL
    
    if element_props.geom_type == 'channel':
        W0_nodes_tensor = element_props.W0_nodes
        width_ref_for_ghost = W0_nodes_tensor[0] # Upstream actual bottom width
        SS1 = element_props.SS1
        SS2 = element_props.SS2
    else: # Plane
        W0_nodes_tensor = element_props.WID # Use WID as the "bottom width" tensor
        width_ref_for_ghost = element_props.WID
        SS1, SS2 = torch.tensor(0.0, device=W0_nodes_tensor.device), torch.tensor(0.0, device=W0_nodes_tensor.device)

    # --- Combined Lateral Inflow (per unit length, m^2/s) ---
    q_lat_combined_nodes = q_lat_nodes_prcp.clone() # Already per unit length
    if element_props.geom_type == 'channel' and plane_lat_Q_total > EPSILON:
        if element_props.LEN > EPSILON: # LEN is 0-dim tensor
            q_lat_combined_nodes = q_lat_combined_nodes + (plane_lat_Q_total / element_props.LEN)

    # --- Define L(A) operator ---
    def calculate_L_operator(A_state: torch.Tensor) -> torch.Tensor:
        A_state_clamped = torch.clamp(A_state, min=0.0)

        # --- 1. Pad A_state with ghost cells ---
        A_ghost_L = torch.tensor(0.0, device=A_state.device, dtype=A_state.dtype)
        if upstream_Q > EPSILON:
            n_safe = torch.clamp(element_props.MAN, min=1e-4)
            s_safe = torch.clamp(slopes_tensor[0], min=EPSILON**2) # Use slope at first node
            w_ghost_ref = torch.clamp(width_ref_for_ghost, min=EPSILON) # This is a tensor

            h_ghost_rect_approx = torch.pow(
                torch.clamp((upstream_Q * n_safe) / (w_ghost_ref * torch.sqrt(s_safe) + EPSILON), min=0.0),
                3.0/5.0
            )
            if element_props.geom_type == 'channel':
                A_ghost_L = get_trapezoid_area_from_h(h_ghost_rect_approx, w_ghost_ref, SS1, SS2)
            else: # Plane
                A_ghost_L = get_plane_area_from_h(h_ghost_rect_approx, w_ghost_ref)
            # A_ghost_L = torch.clamp(A_ghost_L, min=0.0, max=A_state_clamped[0]*5.0 if A_state_clamped[0]>EPSILON else 1.0)
            A_ghost_L = torch.clamp(A_ghost_L, min=0.0)

        A_ghost_R = A_state_clamped[-1:]
        A_padded = torch.cat((A_ghost_L.unsqueeze(0), A_state_clamped, A_ghost_R))

        # --- 2. Calculate Limited Slopes (van Leer) ---
        delta_L = (A_padded[1:-1] - A_padded[:-2]) / dx_tensor 
        delta_R = (A_padded[2:] - A_padded[1:-1]) / dx_tensor
        limited_slope = van_leer_limiter(delta_L, delta_R)

        # --- 3. Reconstruct Interface Values ---
        A_L_recon = A_state_clamped + 0.5 * dx_tensor * limited_slope
        A_R_recon = A_state_clamped - 0.5 * dx_tensor * limited_slope 
        A_R_interface = torch.cat((A_R_recon[1:], A_R_recon[-1:]))
        A_L_interface = torch.clamp(A_L_recon, min=0.0)
        A_R_interface = torch.clamp(A_R_interface, min=0.0)

        # --- 4. Calculate Interface Fluxes Q* (Godunov Upwind Type) ---
        # First, calculate the physical fluxes Q(A_L_interface) and Q(A_R_interface)
        # We need geometry (wp, h) for A_L_interface and A_R_interface
        
        # Geometry for A_L_interface
        if element_props.geom_type == 'channel':
            h_L_iface = get_h_from_trapezoid_area(A_L_interface, W0_nodes_tensor, SS1, SS2)
            wp_L_iface = get_trapezoid_wp_from_h(h_L_iface, W0_nodes_tensor, SS1, SS2)
        else: # Plane
            h_L_iface = get_plane_h_from_area(A_L_interface, W0_nodes_tensor) # W0_nodes_tensor is props.WID
            wp_L_iface = W0_nodes_tensor.expand_as(A_L_interface)
        q_L_iface = calculate_q_manning(A_L_interface, wp_L_iface, element_props.MAN, slopes_tensor)

        # # Geometry for A_R_interface
        # if element_props.geom_type == 'channel':
        #     h_R_iface = get_h_from_trapezoid_area(A_R_interface, W0_nodes_tensor, SS1, SS2)
        #     wp_R_iface = get_trapezoid_wp_from_h(h_R_iface, W0_nodes_tensor, SS1, SS2)
        # else: # Plane
        #     h_R_iface = get_plane_h_from_area(A_R_interface, W0_nodes_tensor)
        #     wp_R_iface = W0_nodes_tensor.expand_as(A_R_interface)
        # q_R_iface = calculate_q_manning(A_R_interface, wp_R_iface, element_props.MAN, slopes_tensor)

        # # Estimate wave speed c_k at interfaces.
        # # One way: average A at interface, then get c_k.
        # A_avg_interface = 0.5 * (A_L_interface + A_R_interface)
        # A_avg_interface_clamped = torch.clamp(A_avg_interface, min=0.0)

        # if element_props.geom_type == 'channel':
        #     h_avg_iface = get_h_from_trapezoid_area(A_avg_interface_clamped, W0_nodes_tensor, SS1, SS2)
        #     wp_avg_iface = get_trapezoid_wp_from_h(h_avg_iface, W0_nodes_tensor, SS1, SS2)
        #     tw_avg_iface = get_trapezoid_topwidth_from_h(h_avg_iface, W0_nodes_tensor, SS1, SS2)
        # else: # Plane
        #     h_avg_iface = get_plane_h_from_area(A_avg_interface_clamped, W0_nodes_tensor)
        #     wp_avg_iface = W0_nodes_tensor.expand_as(A_avg_interface_clamped)
        #     tw_avg_iface = W0_nodes_tensor.expand_as(A_avg_interface_clamped)

        # Q_avg_iface = calculate_q_manning(A_avg_interface_clamped, wp_avg_iface, element_props.MAN, slopes_tensor)
        # celerity_at_interface = calculate_dqda_manning_general(
        #     A_avg_interface_clamped, wp_avg_iface, tw_avg_iface, Q_avg_iface,
        #     SS1, SS2, element_props.geom_type
        # )

        # Godunov upwind flux selection
        # For kinematic wave, dQ/dA is generally positive, so wave moves L to R.
        # q_star_interfaces = torch.where(celerity_at_interface >= 0, q_L_iface, q_R_iface)
        # A more robust check for dQ/dA >=0 for kinematic wave (always flows downhill)
        # For simplicity and given KW, dQ/dA > 0, so waves always propagate downstream.
        # Thus, the upwind state is always the left state.
        q_star_interfaces = q_L_iface 
        # Note: This simplification (always taking q_L_iface) is common for kinematic wave
        # because characteristics only go one way. If dQ/dA could be negative (e.g. dynamic wave
        # backwater), then the torch.where based on sign of celerity_at_interface is needed.
        # Let's stick to the simple q_L_iface for KW, assuming positive wave speed.

        # --- 5. Calculate L(A) = - (Q*_{j+1/2} - Q*_{j-1/2})/dx + S_j ---
        q_star_ghost_L = upstream_Q.reshape(1) 
        q_star_padded_for_diff = torch.cat((q_star_ghost_L, q_star_interfaces))
        
        flux_diff = q_star_padded_for_diff[1:] - q_star_padded_for_diff[:-1]
        L_A = -flux_diff / dx_tensor + q_lat_combined_nodes # dx_tensor is element_props.dx_avg
        return L_A

    # --- TVD RK2 Time Marching --- (as before)
    L_n = calculate_L_operator(A_curr)
    A_1 = A_curr + dt * L_n 
    A_1 = torch.clamp(A_1, min=0.0)
    L_1 = calculate_L_operator(A_1)
    A_next = 0.5 * A_curr + 0.5 * (A_1 + dt * L_1)
    
    return torch.clamp(A_next, min=0.0)


def explicit_muscl_yu_duan_with_plane_contrib(
    A_curr: torch.Tensor,
    q_lat_nodes_prcp: torch.Tensor, # Net precip per unit length (m^2/s)
    element_props: ElementProperties,
    dt: torch.Tensor,             # CHANGED to 0-dim tensor
    upstream_Q: torch.Tensor,     # CHANGED to 0-dim tensor (total Q m^3/s)
    plane_lat_Q_total: torch.Tensor # CHANGED to 0-dim tensor (total Q m^3/s from side planes)
    ) -> torch.Tensor:
    """
    MUSCL scheme for dA/dt + dQ/dx = q_lat.
    Uses AVERAGE dx (element_props.dx_avg which is a 0-dim tensor).
    Handles combined lateral inflow from direct precipitation and distributed plane inflow.
    All dynamic scalar inputs (dt, Qs) are 0-dim tensors.
    """

    # --- Properties (already tensors from element_props) ---
    dx_tensor = element_props.dx_avg # 0-dim tensor
    slopes_tensor = element_props.SL
    
    if element_props.geom_type == 'channel':
        W0_nodes_tensor = element_props.W0_nodes
        width_ref_for_ghost = W0_nodes_tensor[0] # Upstream actual bottom width
        SS1 = element_props.SS1
        SS2 = element_props.SS2
    else: # Plane
        W0_nodes_tensor = element_props.WID # Use WID as the "bottom width" tensor
        width_ref_for_ghost = element_props.WID
        SS1, SS2 = torch.tensor(0.0, device=W0_nodes_tensor.device), torch.tensor(0.0, device=W0_nodes_tensor.device)

    # --- Combined Lateral Inflow (per unit length, m^2/s) ---
    q_lat_combined_nodes = q_lat_nodes_prcp.clone() # Already per unit length
    if element_props.geom_type == 'channel' and plane_lat_Q_total > EPSILON:
        if element_props.LEN > EPSILON: # LEN is 0-dim tensor
            q_lat_combined_nodes = q_lat_combined_nodes + (plane_lat_Q_total / element_props.LEN)

    # --- Define L(A) operator ---
    def calculate_L_operator(A_state: torch.Tensor) -> torch.Tensor:
        A_state_clamped = torch.clamp(A_state, min=0.0)

        # --- 1. Pad A_state with ghost cells ---
        A_ghost_L = torch.tensor(0.0, device=A_state.device, dtype=A_state.dtype)
        if upstream_Q > EPSILON:
            n_safe = torch.clamp(element_props.MAN, min=1e-4)
            s_safe = torch.clamp(slopes_tensor[0], min=EPSILON**2) # Use slope at first node
            w_ghost_ref = torch.clamp(width_ref_for_ghost, min=EPSILON) # This is a tensor

            h_ghost_rect_approx = torch.pow(
                torch.clamp((upstream_Q * n_safe) / (w_ghost_ref * torch.sqrt(s_safe) + EPSILON), min=0.0),
                3.0/5.0
            )
            if element_props.geom_type == 'channel':
                A_ghost_L = get_trapezoid_area_from_h(h_ghost_rect_approx, w_ghost_ref, SS1, SS2)
            else: # Plane
                A_ghost_L = get_plane_area_from_h(h_ghost_rect_approx, w_ghost_ref)
            # A_ghost_L = torch.clamp(A_ghost_L, min=0.0, max=A_state_clamped[0]*5.0 if A_state_clamped[0]>EPSILON else 1.0)
            A_ghost_L = torch.clamp(A_ghost_L, min=0.0)

        A_ghost_R = A_state_clamped[-1:]
        A_padded = torch.cat((A_ghost_L.unsqueeze(0), A_state_clamped, A_ghost_R))

        # --- 2. Calculate Limited Slopes (van Leer) ---
        delta_L = (A_padded[1:-1] - A_padded[:-2]) / dx_tensor 
        delta_R = (A_padded[2:] - A_padded[1:-1]) / dx_tensor
        limited_slope = van_leer_limiter(delta_L, delta_R)

        # --- 3. Reconstruct Interface Values ---
        A_L_recon = A_state_clamped + 0.5 * dx_tensor * limited_slope
        A_L_interface = torch.clamp(A_L_recon, min=0.0)

        # --- 4. Calculate Interface Fluxes Q* (Godunov Upwind Type) ---
        # First, calculate the physical fluxes Q(A_L_interface) and Q(A_R_interface)
        # We need geometry (wp, h) for A_L_interface and A_R_interface
        
        # Geometry for A_L_interface
        if element_props.geom_type == 'channel':
            h_L_iface = get_h_from_trapezoid_area(A_L_interface, W0_nodes_tensor, SS1, SS2)
            wp_L_iface = get_trapezoid_wp_from_h(h_L_iface, W0_nodes_tensor, SS1, SS2)
        else: # Plane
            h_L_iface = get_plane_h_from_area(A_L_interface, W0_nodes_tensor) # W0_nodes_tensor is props.WID
            wp_L_iface = W0_nodes_tensor.expand_as(A_L_interface)
        q_L_iface = calculate_q_manning(A_L_interface, wp_L_iface, element_props.MAN, slopes_tensor)

        # Godunov upwind flux selection
        # For kinematic wave, dQ/dA is generally positive, so wave moves L to R.
        q_star_interfaces = q_L_iface 

        # --- 5. Calculate L(A) = - (Q*_{j+1/2} - Q*_{j-1/2})/dx + S_j ---
        q_star_ghost_L = upstream_Q.reshape(1) 
        q_star_padded_for_diff = torch.cat((q_star_ghost_L, q_star_interfaces))
        
        flux_diff = q_star_padded_for_diff[1:] - q_star_padded_for_diff[:-1]
        L_A = -flux_diff / dx_tensor + q_lat_combined_nodes # dx_tensor is element_props.dx_avg; 
        return L_A

    # --- TVD RK2 Time Marching --- (as before)
    L_n = calculate_L_operator(A_curr)
    A_1 = A_curr + dt * L_n 
    A_1 = torch.clamp(A_1, min=0.0)
    L_1 = calculate_L_operator(A_1)
    A_next = 0.5 * A_curr + 0.5 * (A_1 + dt * L_1)
    
    return torch.clamp(A_next, min=0.0)