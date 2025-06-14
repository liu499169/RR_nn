# src/io/config_saver.py
import json
import torch
from src.data_structures import ElementProperties, SoilPropertiesIntermediate

def _tensor_to_json_serializable(value: torch.Tensor):
    """Converts a PyTorch tensor to a JSON serializable format (list or scalar)."""
    if value.numel() == 1:
        return value.item()
    return value.cpu().tolist() # Ensure it's on CPU before converting to list

def _namedtuple_to_dict_for_json(nt):
    """Converts a namedtuple (with potential tensor fields) to a JSON serializable dict."""
    if not hasattr(nt, '_asdict'):
        return nt # Not a namedtuple we can convert easily
    
    d = nt._asdict()
    for key, value in d.items():
        if isinstance(value, torch.Tensor):
            d[key] = _tensor_to_json_serializable(value)
        elif hasattr(value, '_asdict'): # Nested namedtuple
            d[key] = _namedtuple_to_dict_for_json(value)
        # Handle lists of namedtuples (e.g., if props/params were lists within a dict)
        elif isinstance(value, list) and value and hasattr(value[0], '_asdict'):
            d[key] = [_namedtuple_to_dict_for_json(item) for item in value]
    return d

def save_simulation_configuration(
    filepath: str,
    simulation_settings: dict,
    global_param_overrides: dict,
    learnable_params_list: list[str],
    watershed_obj, # Actually, easier to pass the maps directly
    element_properties_map: dict[int, ElementProperties],
    soil_parameters_map: dict[int, SoilPropertiesIntermediate],
    simulation_order: list[int],
    connectivity_map: dict[int, int]
    ):
    """
    Saves the complete, resolved simulation configuration to a JSON file.
    """
    print(f"Saving resolved simulation configuration to: {filepath}")
    
    resolved_config = {
        "simulation_settings": simulation_settings,
        "global_parameter_overrides_applied": global_param_overrides,
        "learnable_parameters_configured": learnable_params_list,
        "simulation_order": simulation_order,
        "group_connectivity_map": connectivity_map,
        "resolved_elements": {}
    }

    for eid, props in element_properties_map.items():
        soil_p = soil_parameters_map.get(eid)
        resolved_config["resolved_elements"][str(eid)] = { # Use string keys for JSON
            "properties": _namedtuple_to_dict_for_json(props),
            "soil_parameters": _namedtuple_to_dict_for_json(soil_p) if soil_p else None
        }
        # Include geom_type for easier parsing if this JSON is re-read
        if props:
             resolved_config["resolved_elements"][str(eid)]["geom_type"] = props.geom_type


    try:
        with open(filepath, 'w') as f:
            json.dump(resolved_config, f, indent=4, ensure_ascii=False)
        print(f"  Configuration saved successfully to {filepath}")
    except Exception as e:
        print(f"ERROR: Could not save configuration to {filepath}: {e}")