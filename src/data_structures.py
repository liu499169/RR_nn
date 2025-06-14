# src/data_structures.py

import torch
from collections import namedtuple
import math # Keep for sqrt if used on scalars, though torch.sqrt is more general

# Import from your updated flow_physics for consistency if needed by helpers
# (get_depth_from_state calls functions from flow_physics)
from src.core.physics_formulas import (get_h_from_trapezoid_area, get_plane_h_from_area,
                              get_trapezoid_topwidth_from_h, get_trapezoid_wp_from_h)

# --- Configuration ---
EPSILON = 1e-9 # Small number often used to prevent division by zero or for comparisons

# --- Core Data Structure Definitions ---

ElementProperties = namedtuple("ElementProperties", [
    "element_id",       # int: Unique ID for the element.
    "geom_type",        # str: Type of element, e.g., 'plane' or 'channel'.
    "side",             # Optional[str]: For planes, indicates 'head', 'left', 'right'. None for channels.
    "from_node",        # Optional[int]: Upstream node ID in the basin network (for channels).
    "to_node",          # Optional[int]: Downstream node ID in the basin network (for channels).
    "LEN",              # torch.Tensor: Total length of the element (m).
    "WID",              # torch.Tensor: Characteristic width. For planes, it's the plane width (m).
                        #        For channels, it might be an average width or a reference (m).
    "num_nodes",        # int: Number of spatial discretization nodes along the element.
    "dx_segments",      # torch.Tensor: Lengths of segments between nodes (m); shape (num_nodes-1,).
    "dx_avg",           # torch.Tensor: Average segment length (m).
    "node_x",           # torch.Tensor: Distance to each node from the element's upstream end (m); shape (num_nodes,).
    "SL",               # torch.Tensor: Slope at each node (m/m or unitless); shape (num_nodes,).
    "MAN",              # torch.Tensor: Manning's roughness coefficient (unitless); scalar tensor.
    # Channel specific geometry (None or 0.0 for planes):
    "W0_nodes",         # Optional[torch.Tensor]: Bottom width at each node for channels (m); shape (num_nodes,). None for planes.
    "SS1",              # torch.Tensor: Side slope 1 for trapezoidal channels (z:1, H:V, unitless). 0.0 for planes/rect.
    "SS2"               # torch.Tensor: Side slope 2 for trapezoidal channels (z:1, H:V, unitless). 0.0 for planes/rect.
])

OverlandFlowState = namedtuple("OverlandFlowState", [
    "t_elapsed",        # torch.Tensor: Elapsed simulation time for this element (s); scalar tensor.
    "area",             # torch.Tensor: Flow cross-sectional area A at each node (m^2); shape (num_nodes,).
    "depth",            # torch.Tensor: Water depth h at each node (m); shape (num_nodes,). Derived from area.
    "discharge",        # torch.Tensor: Discharge Q at each node (m^3/s); shape (num_nodes,).
    "max_cfl"
])

InfiltrationState = namedtuple("InfiltrationState", [
    "theta_current",    # torch.Tensor: Current surface soil moisture content (m^3/m^3); scalar tensor for the element.
    "F_cumulative",     # torch.Tensor: Total cumulative infiltration depth (m); scalar tensor for the element.
    "drying_cumulative" # torch.Tensor: Total cumulative drying in soil moisture content (m/m); scalar tensor.
])

SoilPropertiesIntermediate = namedtuple("SoilPropertiesIntermediate", [
    "Ks",                   # torch.Tensor: Saturated hydraulic conductivity (m/s); scalar tensor.
    "theta_s",              # torch.Tensor: Saturated water content (m^3/m^3); scalar tensor.
    "theta_r",              # torch.Tensor: Residual water content (m^3/m^3); scalar tensor.
    "theta_init_condition", # torch.Tensor: Initial soil moisture condition for deficit calculation (m^3/m^3); scalar tensor.
    "Smax",                 # Optional[torch.Tensor]: Max saturation fraction (unitless), often used to derive effective theta_s. Can be None if not used.
    "HF_max",               # torch.Tensor: Suction head at the wetting front (m); scalar tensor. (psi_f or G)
    "m_exponent",           # torch.Tensor: Exponent for suction/conductivity relation (unitless); scalar tensor.
    "effective_depth",      # torch.Tensor: Effective depth of the surface soil layer for moisture updates (m); scalar tensor.
    "k_drain"               # torch.Tensor: Simple drainage coefficient (1/s); scalar tensor.
])


# --- Helper Functions for State and Properties ---

def get_depth_from_state(state: OverlandFlowState, props: ElementProperties) -> torch.Tensor:
    """Helper to get water depth from flow area based on element geometry.

    Args:
        state (OverlandFlowState): Current flow state of the element.
        props (ElementProperties): Static properties of the element.

    Returns:
        torch.Tensor: Water depth h at each node (m).
    """
    if props.geom_type == 'channel':
        if props.W0_nodes is None: # Should not happen if props are correctly initialized for channel
            raise ValueError(f"Channel element {props.element_id} is missing W0_nodes.")
        return get_h_from_trapezoid_area(state.area, props.W0_nodes, props.SS1, props.SS2)
    elif props.geom_type == 'plane':
        return get_plane_h_from_area(state.area, props.WID)
    else:
        raise ValueError(f"Unknown geom_type '{props.geom_type}' for element {props.element_id}")

def get_wp_from_state(state: OverlandFlowState, props: ElementProperties) -> torch.Tensor:
    """Helper to get wetted perimeter from flow state based on element geometry.

    Args:
        state (OverlandFlowState): Current flow state of the element.
        props (ElementProperties): Static properties of the element.

    Returns:
        torch.Tensor: Wetted perimeter WP at each node (m).
    """
    if props.geom_type == 'channel':
        depth = get_depth_from_state(state, props) # Depth is needed first
        if props.W0_nodes is None:
            raise ValueError(f"Channel element {props.element_id} is missing W0_nodes.")
        return get_trapezoid_wp_from_h(depth, props.W0_nodes, props.SS1, props.SS2)
    elif props.geom_type == 'plane':
        # For a wide rectangular plane, wetted perimeter is approximately its width.
        # This creates a tensor of the same shape as state.area, filled with props.WID.
        return torch.full_like(state.area, props.WID)
    else:
        raise ValueError(f"Unknown geom_type '{props.geom_type}' for element {props.element_id}")

def get_topwidth_from_state(state: OverlandFlowState, props: ElementProperties) -> torch.Tensor:
    """Helper to get top width from flow state based on element geometry.

    Args:
        state (OverlandFlowState): Current flow state of the element.
        props (ElementProperties): Static properties of the element.

    Returns:
        torch.Tensor: Top width T at each node (m).
    """
    if props.geom_type == 'channel':
        depth = get_depth_from_state(state, props) # Depth is needed first
        if props.W0_nodes is None:
            raise ValueError(f"Channel element {props.element_id} is missing W0_nodes.")
        return get_trapezoid_topwidth_from_h(depth, props.W0_nodes, props.SS1, props.SS2)
    elif props.geom_type == 'plane':
        # For a plane, top width is its characteristic width.
        return torch.full_like(state.area, props.WID)
    else:
        raise ValueError(f"Unknown geom_type '{props.geom_type}' for element {props.element_id}")