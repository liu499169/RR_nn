# src/core/physics_formulas.py

import math # Keep for EPSILON comparison if needed, but prefer torch for ops
import torch

# --- Configuration ---
EPSILON = 1e-9
GRAVITY = 9.81 # m/s^2 

# --- Geometry Functions ---

def get_trapezoid_area_from_h(h: torch.Tensor, W0_tensor: torch.Tensor, 
                              SS1_tensor: torch.Tensor, SS2_tensor: torch.Tensor) -> torch.Tensor: # Signatures updated
    """Calculates flow area A for a trapezoidal channel from water depth h.
    All inputs W0_tensor, SS1_tensor, SS2_tensor are expected to be tensors.
    SS1_tensor and SS2_tensor are typically 0-dim.
    """
    h_safe = torch.clamp(h, min=0.0)
    return h_safe * W0_tensor + 0.5 * h_safe**2 * (SS1_tensor + SS2_tensor)

def get_trapezoid_wp_from_h(h: torch.Tensor, W0_tensor: torch.Tensor, 
                            SS1_tensor: torch.Tensor, SS2_tensor: torch.Tensor) -> torch.Tensor: # Signatures updated
    """Calculates wetted perimeter WP for a trapezoidal channel from water depth h."""
    h_safe = torch.clamp(h, min=0.0)
    
    side1 = h_safe * torch.sqrt(1.0 + SS1_tensor**2) # Use torch.sqrt for tensor inputs
    side2 = h_safe * torch.sqrt(1.0 + SS2_tensor**2) # Use torch.sqrt
    return W0_tensor + side1 + side2

def get_trapezoid_topwidth_from_h(h: torch.Tensor, W0_tensor: torch.Tensor, 
                                  SS1_tensor: torch.Tensor, SS2_tensor: torch.Tensor) -> torch.Tensor: # Signatures updated
    """Calculates top width T for a trapezoidal channel from water depth h."""
    h_safe = torch.clamp(h, min=0.0)
    return W0_tensor + h_safe * (SS1_tensor + SS2_tensor)

def get_h_from_trapezoid_area(A: torch.Tensor, W0_tensor: torch.Tensor, 
                              SS1_tensor: torch.Tensor, SS2_tensor: torch.Tensor) -> torch.Tensor: # Signatures updated
    """Calculates water depth h from flow area A for a trapezoidal channel."""
    A_safe = torch.clamp(A, min=0.0)
    
    # a_quad = 0.5 * (SS1 + SS2) -> now tensor operation
    a_quad_tensor = 0.5 * (SS1_tensor + SS2_tensor) # Result is 0-dim tensor
    # Expand if A_safe is not 0-dim and a_quad_tensor is, for broadcasting in b**2 - 4ac
    if A_safe.ndim > 0 and a_quad_tensor.ndim == 0:
        a_quad_expanded = a_quad_tensor.expand_as(A_safe)
    else:
        a_quad_expanded = a_quad_tensor
        
    b_quad_tensor = W0_tensor # W0_tensor is already N-dim or 0-dim
    c_quad_tensor = -A_safe

    discriminant = torch.clamp(b_quad_tensor**2 - 4*a_quad_expanded*c_quad_tensor, min=0.0)
    
    # Numerically stable condition for quadratic formula vs linear (A/W0)
    h = torch.where(
        torch.abs(a_quad_expanded) > EPSILON, # Compare expanded 'a'
        (-b_quad_tensor + torch.sqrt(discriminant)) / (2*a_quad_expanded + EPSILON),
        A_safe / (b_quad_tensor + EPSILON) 
    )
    return torch.clamp(h, min=0.0)

def get_plane_h_from_area(A: torch.Tensor, WID_tensor: torch.Tensor) -> torch.Tensor: # Signature updated
    """Calculates water depth h from flow area A for a plane (wide rectangle)."""
    A_safe = torch.clamp(A, min=0.0)
    WID_safe = torch.clamp(WID_tensor, min=EPSILON) # WID_tensor is 0-dim
    return A_safe / WID_safe

def get_plane_area_from_h(h: torch.Tensor, WID_tensor: torch.Tensor) -> torch.Tensor: # Signature updated
    """Calculates flow area A from water depth h for a plane (wide rectangle)."""
    h_safe = torch.clamp(h, min=0.0)
    return h_safe * WID_tensor # WID_tensor is 0-dim

# --- Physics Functions ---

def calculate_q_manning(A: torch.Tensor, WP: torch.Tensor, 
                        MAN_tensor: torch.Tensor, SL_tensor: torch.Tensor) -> torch.Tensor: # Args are already tensors
    """Calculates discharge Q using Manning's equation."""
    # A, WP, MAN_tensor, SL_tensor are expected to be tensors of correct device/dtype.
    A_safe = torch.clamp(A, min=EPSILON)
    WP_safe = torch.clamp(WP, min=EPSILON)
    n_safe = torch.clamp(MAN_tensor, min=1e-4) # MAN_tensor is 0-dim or N-dim
    slope_safe = torch.clamp(SL_tensor, min=EPSILON**2) # SL_tensor is N-dim

    R_safe = A_safe / WP_safe 
    # R_pow_safe = torch.clamp(R_safe, min=EPSILON) # Clamping R before pow if R can be zero

    q = (1.0 / n_safe) * A_safe * torch.pow(R_safe, 2.0/3.0) * torch.sqrt(slope_safe)
    q = torch.where(A <= EPSILON, torch.zeros_like(q), q)
    return q

def calculate_dqda_manning_general(
    A: torch.Tensor, WP: torch.Tensor, TopWidth: torch.Tensor, Q: torch.Tensor,
    SS1_tensor: torch.Tensor, SS2_tensor: torch.Tensor, # CHANGED to tensors
    geom_type: str
    ) -> torch.Tensor:
    """Calculates dQ/dA for Manning's, dispatching by geom_type."""
    current_dtype = A.dtype # Not strictly needed if all inputs are tensors
    A_safe = torch.clamp(A, min=EPSILON)

    if geom_type == 'channel':
        WP_safe = torch.clamp(WP, min=EPSILON)
        T_safe = torch.clamp(TopWidth, min=EPSILON) # TopWidth is N-dim tensor

        # dWP/dh for a trapezoid: beta_wp = sqrt(1+SS1^2) + sqrt(1+SS2^2)
        # SS1_tensor, SS2_tensor are 0-dim tensors
        beta_wp_tensor = torch.sqrt(1.0 + SS1_tensor**2) + torch.sqrt(1.0 + SS2_tensor**2) # Result is 0-dim tensor

        R_safe = A_safe / WP_safe
        
        # Factor for dQ/dA = (Q/A) * Factor
        # All ops are tensor ops now
        factor = (5.0/3.0) - (2.0/3.0) * (R_safe / (T_safe + EPSILON)) * beta_wp_tensor 
        dqda = factor * (Q / A_safe)

    elif geom_type == 'plane':
        dqda = (5.0 / 3.0) * (Q / A_safe) # Q ~ A^(5/3)
    else:
        raise ValueError(f"Unknown geom_type '{geom_type}' in calculate_dqda_manning_general")

    dqda = torch.where(A <= EPSILON, torch.zeros_like(dqda), dqda)
    return dqda

# --- Helpers for MUSCL Schemes ---
def minmod_limiter(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: # Already tensor-based
    # ... (implementation as before) ...
    signs_match = (torch.sign(a) == torch.sign(b))
    min_abs_val = torch.minimum(torch.abs(a), torch.abs(b))
    return torch.where(signs_match, torch.sign(a) * min_abs_val, torch.zeros_like(a))


def van_leer_limiter(delta_L: torch.Tensor, delta_R: torch.Tensor) -> torch.Tensor: # Already tensor-based
    # ... (implementation as before) ...
    numerator = delta_L * torch.abs(delta_R) + delta_R * torch.abs(delta_L)
    denominator = torch.abs(delta_L) + torch.abs(delta_R) + EPSILON 
    return numerator / denominator

def calculate_froude_number(
    Q: torch.Tensor,      # Discharge at each node (m^3/s)
    A: torch.Tensor,      # Flow area at each node (m^2)
    TopWidth: torch.Tensor, # Top width at each node (m)
    geom_type: str,       # 'plane' or 'channel'
    depth_h: torch.Tensor | None = None # Optional: if already computed for planes
    ) -> torch.Tensor:
    """
    Calculates the Froude number at each node of an element.

    Args:
        Q (torch.Tensor): Discharge (m^3/s).
        A (torch.Tensor): Flow area (m^2).
        TopWidth (torch.Tensor): Flow top width (m). For planes, this is props.WID.
        geom_type (str): 'plane' or 'channel'.
        depth_h (Optional[torch.Tensor]): Pre-calculated depth (m), useful for planes.

    Returns:
        torch.Tensor: Froude number at each node (dimensionless).
    """
    A_safe = torch.clamp(A, min=EPSILON)
    velocity = Q / A_safe

    if geom_type == 'plane':
        if depth_h is None: # Should be passed for planes for efficiency
            raise ValueError("depth_h must be provided for geom_type 'plane' in calculate_froude_number")
        hydraulic_depth = torch.clamp(depth_h, min=EPSILON)
    elif geom_type == 'channel':
        if depth_h is not None:
            hydraulic_depth = torch.clamp(depth_h, min=EPSILON)
        else:
            T_safe = torch.clamp(TopWidth, min=EPSILON)
            hydraulic_depth = A_safe / T_safe
    else:
        raise ValueError(f"Unknown geom_type '{geom_type}' for Froude number calculation.")
    
    hydraulic_depth_safe = torch.clamp(hydraulic_depth, min=EPSILON)
    
    froude_num = torch.abs(velocity) / torch.sqrt(GRAVITY * hydraulic_depth_safe + EPSILON) # Add EPSILON for sqrt
    
    # If area is very small, Froude number is ill-defined or should be 0
    froude_num = torch.where(A <= EPSILON, torch.zeros_like(froude_num), froude_num)
    return froude_num


def calculate_cfl_number(
    Q: torch.Tensor,                # Discharge at each node (m^3/s)
    A: torch.Tensor,                # Flow area at each node (m^2)
    dx_segment_cfl: torch.Tensor,   # dx segment (m)
    dt: torch.Tensor                # The timestep (s)
    ) -> torch.Tensor:
    """
    Calculates the CFL number number at each node of an element.

    Args:
        Q (torch.Tensor): Discharge (m^3/s).
        A (torch.Tensor): Flow area (m^2).
        dx_min (torch.Tensor): The minimum dx segment (m).
        dt (float): The timestep.

    Returns:
        torch.Tensor: CFL number at each node (dimensionless).
    """
    A_safe = torch.clamp(A, min=EPSILON)
    velocity = Q / A_safe

    cfl_num = torch.abs(velocity) * dt / dx_segment_cfl

    return cfl_num