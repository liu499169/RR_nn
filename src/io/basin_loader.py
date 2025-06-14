# src/io/basin_loader.py

import math
import torch

from src.data_structures import ElementProperties, SoilPropertiesIntermediate

# Constants 
MMHR_TO_MS = 1.0 / (1000.0 * 3600.0)
MM_TO_M = 1.0 / 1000.0

# --- Helper Function for Parameter Parsing ---
def _get_overridden_param_value(
    key_list_primary: list[str],
    element_data_dict: dict, # The dict for the specific element from JSON
    param_overrides_for_type: dict, # Pre-filtered global overrides for this geom_type
    default_value: float,
    is_soil_param: bool = True, # True if param is in element_data_dict['soil_params']
    element_id_for_warning: int = -1
    ) -> float:
    """
    Gets a parameter value as a float, prioritizing:
    1. param_overrides_for_type
    2. element_data_dict (either top-level or in 'soil_params')
    3. default_value
    """
    val_found = None
    source = "default"

    # 1. Check Overrides
    if param_overrides_for_type:
        for key in key_list_primary:
            if key.lower() in param_overrides_for_type:
                val_found = param_overrides_for_type[key.lower()]
                source = "override"
                break
    
    # 2. Check element_data_dict (JSON)
    if val_found is None:
        target_dict_for_json_lookup = element_data_dict
        if is_soil_param:
            target_dict_for_json_lookup = element_data_dict.get('soil_params', {})
        
        if target_dict_for_json_lookup:
            target_dict_lower = {k.lower(): v for k, v in target_dict_for_json_lookup.items()}
            for key in key_list_primary:
                if key.lower() in target_dict_lower:
                    val_found = target_dict_lower[key.lower()]
                    source = "json"
                    break
    
    # 3. Handle Default or Problematic Found Value
    if val_found is None or (isinstance(val_found, (float, int)) and math.isnan(float(val_found))): # handle int too
        if val_found is not None: # Warn if NaN was explicitly found
             print(f"Warning: Param '{key_list_primary[0]}' for element {element_id_for_warning} from {source} is NaN. Using default {default_value}.")
        val_found = default_value
        source = f"default (used: {default_value})"
    
    try:
        return float(val_found)
    except (ValueError, TypeError):
        print(f"Warning: Could not convert value for '{key_list_primary[0]}' ('{val_found}') from {source} for element {element_id_for_warning} to float. Using default {default_value}.")
        return float(default_value)

# --- Runtime Element Parser ---
def parse_element_override(
    element_data_from_json: dict, 
    global_param_overrides: dict, # e.g., {'plane': {'MAN': 0.1}, 'channel': {'MAN': 0.05}}
    device: torch.device, 
    dtype: torch.dtype,
    learnable_params_list: list[str] | None = None
    ) -> tuple[ElementProperties | None, SoilPropertiesIntermediate | None]:
    """
    Parses properties and parameters for a single element from its JSON data,
    applies runtime overrides, converts necessary fields to tensors on the
    specified device/dtype, and sets `requires_grad=True` for specified learnable parameters.

    Scalar physical properties (LEN, WID, dx_avg, SS1, SS2) are stored as 0-dim tensors.

    Args:
        element_data_from_json (dict): Dictionary for a single element from the processed JSON.
        global_param_overrides (dict): Global parameter overrides from the simulation script.
        device (torch.device): PyTorch device.
        dtype (torch.dtype): PyTorch dtype.
        learnable_params_list (list[str], optional): List of parameter names (lowercase, e.g., 
                                                ['man', 'ks']) that should have `requires_grad=True`.

    Returns:
        tuple[Optional[ElementProperties], Optional[SoilPropertiesIntermediate]]:
            The parsed properties and soil parameters, or (None, None) if parsing fails.
    """
    if learnable_params_list is None:
        learnable_params_list = []
    learnable_params_lower = [p.lower() for p in learnable_params_list]

    if not element_data_from_json: return None, None
    element_id = int(element_data_from_json.get('element_id', -1))
    geom_type = element_data_from_json.get('geom_type')
    if not geom_type:
        print(f"Warning: Element {element_id} missing 'geom_type'. Cannot parse.")
        return None, None

    # Get overrides specific to this element's geometry type
    overrides_for_this_type = {}
    if global_param_overrides:
        global_param_overrides_lower_keys = {k.lower(): v for k, v in global_param_overrides.items()}
        if geom_type.lower() in global_param_overrides_lower_keys:
            # Ensure sub-dictionary keys are also lowercase for matching in _get_overridden_param_value
            overrides_for_this_type = {k.lower(): v for k, v in global_param_overrides_lower_keys[geom_type.lower()].items()}
    
    # --- ElementProperties Fields ---
    props_dict = {}
    props_dict['element_id'] = element_id
    props_dict['geom_type'] = geom_type
    props_dict['side'] = element_data_from_json.get('side') # str or None
    props_dict['from_node'] = element_data_from_json.get('from_node') # int or None
    props_dict['to_node'] = element_data_from_json.get('to_node') # int or None
    
    # Scalar float properties converted to 0-dim Tensors
    props_dict['LEN'] = torch.tensor(float(element_data_from_json.get('LEN', 0.0)), device=device, dtype=dtype)
    props_dict['WID'] = torch.tensor(float(element_data_from_json.get('WID', 0.0)), device=device, dtype=dtype)
    props_dict['num_nodes'] = int(element_data_from_json.get('num_nodes', 0)) # int
    props_dict['dx_avg'] = torch.tensor(float(element_data_from_json.get('dx_avg', 0.0)), device=device, dtype=dtype)
    props_dict['SS1'] = torch.tensor(float(element_data_from_json.get('SS1', 0.0)), device=device, dtype=dtype)
    _ss2_default = props_dict['SS1'].item() # Default SS2 to SS1 if SS2 not present
    props_dict['SS2'] = torch.tensor(float(element_data_from_json.get('SS2', _ss2_default)), device=device, dtype=dtype)

    # Manning's N (Tensor, potentially learnable)
    man_val_float = _get_overridden_param_value(
        key_list_primary=['MAN', 'Mann_N', 'manning_n'],
        element_data_dict=element_data_from_json, # MAN is top-level in JSON element_data
        param_overrides_for_type=overrides_for_this_type,
        default_value=0.03, # Example default
        is_soil_param=False, # MAN is not in 'soil_params' sub-dict
        element_id_for_warning=element_id
    )
    props_dict['MAN'] = torch.tensor(man_val_float, device=device, dtype=dtype, 
                                     requires_grad=('man' in learnable_params_lower or \
                                                    'manning_n' in learnable_params_lower or \
                                                    'mann_n' in learnable_params_lower))
    
    # Array-like properties (already lists in JSON, convert to Tensor)
    props_dict['dx_segments'] = torch.tensor(element_data_from_json.get('dx_segments', []), device=device, dtype=dtype)
    props_dict['node_x'] = torch.tensor(element_data_from_json.get('node_x', []), device=device, dtype=dtype)
    props_dict['SL'] = torch.tensor(element_data_from_json.get('SL', []), device=device, dtype=dtype)
    
    w0_nodes_data = element_data_from_json.get('W0_nodes');# print(f"W0_nodes_data: {w0_nodes_data}")
    props_dict['W0_nodes'] = torch.tensor(w0_nodes_data, device=device, dtype=dtype) if w0_nodes_data is not None else None

    try:
        element_props = ElementProperties(**props_dict)
    except TypeError as e:
        print(f"Error creating ElementProperties for {element_id} (type {geom_type}): {e}.\nProps dict: {props_dict}")
        return None, None

    # --- SoilPropertiesIntermediate Fields ---
    # All fields will be Tensors, some potentially learnable
    # Defaults are illustrative.
    ks_val_f = _get_overridden_param_value(['Ks', 'KS'], element_data_from_json, overrides_for_this_type, 5.0, element_id_for_warning=element_id)
    theta_s_val_f = _get_overridden_param_value(['Por', 'POR', 'porosity', 'theta_s'], element_data_from_json, overrides_for_this_type, 0.45, element_id_for_warning=element_id)
    theta_r_val_f = _get_overridden_param_value(['theta_r'], element_data_from_json, overrides_for_this_type, 0.05, element_id_for_warning=element_id)
    smax_val_f = _get_overridden_param_value(['Smax'], element_data_from_json, overrides_for_this_type, 1.0, element_id_for_warning=element_id)
    effective_theta_s_val_f = theta_s_val_f * smax_val_f
    theta_init_val_f = _get_overridden_param_value(['theta_init_condition', 'theta_init'], element_data_from_json, overrides_for_this_type, 0.15, element_id_for_warning=element_id)
    hf_max_val_f = _get_overridden_param_value(['G', 'HF_max', 'Hf_max_mm'], element_data_from_json, overrides_for_this_type, 50.0, element_id_for_warning=element_id) # JSON in mm
    m_exponent_val_f = _get_overridden_param_value(['m_exponent'], element_data_from_json, overrides_for_this_type, 0.5, element_id_for_warning=element_id)
    eff_depth_val_f = _get_overridden_param_value(['effective_depth', 'AQ_DEPTH'], element_data_from_json, overrides_for_this_type, 100.0, element_id_for_warning=element_id) # JSON in mm
    k_drain_val_f = _get_overridden_param_value(['k_drain'], element_data_from_json, overrides_for_this_type, 1e-7, element_id_for_warning=element_id)

    soil_p = SoilPropertiesIntermediate(
        Ks=torch.tensor(ks_val_f * MMHR_TO_MS, device=device, dtype=dtype, requires_grad='ks' in learnable_params_lower),
        theta_s=torch.tensor(effective_theta_s_val_f, device=device, dtype=dtype, requires_grad='theta_s' in learnable_params_lower),
        theta_r=torch.tensor(theta_r_val_f, device=device, dtype=dtype, requires_grad='theta_r' in learnable_params_lower),
        theta_init_condition=torch.tensor(theta_init_val_f, device=device, dtype=dtype, requires_grad='theta_init_condition' in learnable_params_lower),
        Smax=torch.tensor(smax_val_f, device=device, dtype=dtype) if smax_val_f is not None else None, # Smax might not be directly learnable
        HF_max=torch.tensor(hf_max_val_f * MM_TO_M, device=device, dtype=dtype, requires_grad='hf_max' in learnable_params_lower),
        m_exponent=torch.tensor(m_exponent_val_f, device=device, dtype=dtype, requires_grad='m_exponent' in learnable_params_lower),
        effective_depth=torch.tensor(eff_depth_val_f * MM_TO_M, device=device, dtype=dtype, requires_grad='effective_depth' in learnable_params_lower),
        k_drain=torch.tensor(k_drain_val_f, device=device, dtype=dtype, requires_grad='k_drain' in learnable_params_lower)
    )
        
    return element_props, soil_p