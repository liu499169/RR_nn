# src/utils/initializers.py
import torch
from tqdm import tqdm
from src.data_structures import ElementProperties, InfiltrationState, OverlandFlowState

def initialize_basin_states(element_props_map: dict[int, ElementProperties], 
                            element_params_map: dict[int, InfiltrationState],
                            device: torch.device, dtype: torch.dtype
                            ) -> dict[int, dict[str, OverlandFlowState | InfiltrationState]]:
    """ Initializes flow and infiltration states for all elements in the basin. """
    basin_states = {}
    print("Initializing simulation states for all elements...")
    for element_id, props in tqdm(element_props_map.items(), desc="Initializing states"):
        # Flow State
        num_nodes = props.num_nodes
        flow_state = OverlandFlowState(
            t_elapsed=torch.tensor(0.0, dtype=dtype, device=device),
            area=torch.zeros(num_nodes, dtype=dtype, device=device),
            depth=torch.zeros(num_nodes, dtype=dtype, device=device),
            discharge=torch.zeros(num_nodes, dtype=dtype, device=device)
        )
        # Infiltration State
        soil_params = element_params_map[element_id]
        infil_state = InfiltrationState(
            theta_current=soil_params.theta_init_condition.clone(), # Use init condition from params
            F_cumulative=torch.tensor(0.0, dtype=dtype, device=device)
        )
        basin_states[element_id] = {'flow': flow_state, 'infil': infil_state}
    print("State initialization complete.")
    return basin_states