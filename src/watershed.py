# src/watershed.py

from tqdm import tqdm
import torch
import json
from collections import defaultdict

# Import from our defined structures and modules
from src.data_structures import ElementProperties, SoilPropertiesIntermediate 
from src.io.basin_loader import parse_element_override # The refined parser
from src.components.plane_element import PlaneElement, PlaneElement_, \
    PlaneElement_im # Importing all variants for different use cases
from src.components.channel_element import ChannelElement, ChannelElement_, \
    ChannelElement_im # Importing all variants for different use cases

class Watershed:
    def __init__(self, 
                 basin_json_path: str, 
                 global_param_overrides: dict,
                 learnable_params_list: list[str],
                 device: torch.device, 
                 dtype: torch.dtype):
        """
        Loads and initializes the watershed, creating element modules.

        Args:
            basin_json_path (str): Path to the processed basin JSON file.
            global_param_overrides (dict): Global parameter overrides.
            learnable_params_list (list[str]): List of parameters to set requires_grad=True for.
            device (torch.device): PyTorch device for modules and tensors.
            dtype (torch.dtype): PyTorch dtype for modules and tensors.
        """
        self.basin_json_path = basin_json_path
        self.global_param_overrides = global_param_overrides
        self.learnable_params_list = learnable_params_list
        self.device = device
        self.dtype = dtype

        self.element_properties: dict[int, ElementProperties] = {}
        self.soil_parameters: dict[int, SoilPropertiesIntermediate] = {}
        self.element_modules: dict[int, torch.nn.Module] = {} # Stores PlaneElement or ChannelElement instances
        
        self.element_group_map: dict[int, int] = {} # {element_id: group_id}
        self.group_elements_map: dict[int, dict] = defaultdict(lambda: {'planes': [], 'channel': None})
        # Example: group_elements_map[group_id]['planes'] = [plane_id1, plane_id2]
        #          group_elements_map[group_id]['channel'] = channel_id
        
        self.simulation_order: list[int] = []
        self.connectivity: dict[int, int] = {} # {upstream_group_id: downstream_group_id}

        self._load_and_build_watershed()

    def _load_and_build_watershed(self):
        """
        Internal method to parse JSON, create properties, and instantiate element modules.
        """
        print(f"Loading and building watershed from: {self.basin_json_path}")
        try:
            with open(self.basin_json_path, 'r') as f:
                config_data = json.load(f)
        except FileNotFoundError:
            print(f"ERROR: Basin JSON file not found at {self.basin_json_path}")
            raise
        except json.JSONDecodeError:
            print(f"ERROR: Could not decode JSON from {self.basin_json_path}")
            raise
        
        if not config_data:
            raise ValueError("Basin JSON configuration is empty.")

        # --- Read Simulation Order and Connectivity (if defined directly) ---
        # Simulation order is critical
        simulation_order_str = config_data.get("simulation_order")
        if not simulation_order_str:
            raise ValueError("'simulation_order' not found or empty in basin JSON.")
        self.simulation_order = [int(gid) for gid in simulation_order_str]
        print(f"  Using predefined simulation order: {self.simulation_order}")

        # Connectivity can be inferred or read if explicitly present
        # For now, assuming connectivity is built by the SimulationEngine or inferred from element from/to nodes
        # If your JSON stores group connectivity directly, load it here.
        # self.connectivity = config_data.get("group_connectivity_map", {}) # Example
        # For now, we will reconstruct connectivity based on element from_node/to_node as in basin_run_dt.py
        
        node_ends_in_group = defaultdict(list)
        node_starts_in_group = {}
        all_basin_nodes = set() # For inferring connectivity

        groups_in_json = {k for k in config_data if k not in ["simulation_order", "simulation_params", "group_connectivity_map"]}

        number_processed_channel, number_processed_planes = 0, 0
        for group_id_str in tqdm(groups_in_json, desc="Processing groups and creating element modules"):
            group_id = int(group_id_str)
            group_data = config_data.get(group_id_str, {})

            # --- Process Channel ---
            channel_json_data = group_data.get('channel')
            if channel_json_data:
                props, soil_p = parse_element_override(
                    element_data_from_json=channel_json_data,
                    global_param_overrides=self.global_param_overrides,
                    device=self.device,
                    dtype=self.dtype,
                    learnable_params_list=self.learnable_params_list
                )
                if props and soil_p:
                    elem_id = props.element_id
                    self.element_properties[elem_id] = props
                    self.soil_parameters[elem_id] = soil_p
                    self.element_group_map[elem_id] = group_id
                    self.group_elements_map[group_id]['channel'] = elem_id
                    
                    channel_module = ChannelElement(props, soil_p, self.device, self.dtype)
                    self.element_modules[elem_id] = channel_module.to(self.device) # Ensure module is on device
                    number_processed_channel += 1
                    # For connectivity inference
                    from_node = channel_json_data.get('from_node'); to_node = channel_json_data.get('to_node')
                    if from_node is not None: all_basin_nodes.add(from_node); node_starts_in_group[from_node] = group_id
                    if to_node is not None: all_basin_nodes.add(to_node); node_ends_in_group[to_node].append(group_id)
                else:
                    print(f"Warning: Could not parse channel for group {group_id}.")
            
            # --- Process Planes (Head and Side) ---
            for plane_type in ['head', 'side']:
                for plane_json_data in group_data.get(plane_type, []):
                    props, soil_p = parse_element_override(
                        element_data_from_json=plane_json_data,
                        global_param_overrides=self.global_param_overrides,
                        device=self.device, 
                        dtype=self.dtype,
                        learnable_params_list=self.learnable_params_list
                    )
                    if props and soil_p:
                        elem_id = props.element_id
                        self.element_properties[elem_id] = props
                        self.soil_parameters[elem_id] = soil_p
                        self.element_group_map[elem_id] = group_id
                        self.group_elements_map[group_id]['planes'].append(elem_id)

                        plane_module = PlaneElement(props, soil_p, self.device, self.dtype)
                        self.element_modules[elem_id] = plane_module.to(self.device)
                        number_processed_planes += 1
                    else:
                        print(f"Warning: Could not parse {plane_type} plane for group {group_id}.")
        print(f"  Processed {number_processed_channel} channels and {number_processed_planes} planes.")
        # --- Infer Group Connectivity Map (same logic as in old basin_run_dt.py) ---
        print("  Inferring group connectivity map...")
        all_group_ids_parsed = set(self.element_group_map.values())
        outlet_node_id_ref = (max(all_group_ids_parsed) + 1) if all_group_ids_parsed else -1
        connections_found = 0
        for node_id_conn in all_basin_nodes:
            if node_id_conn == outlet_node_id_ref: continue
            up_grps_list = node_ends_in_group.get(node_id_conn, [])
            down_grp = node_starts_in_group.get(node_id_conn)
            if down_grp is not None and up_grps_list:
                for up_grp in up_grps_list:
                    if up_grp != down_grp:
                        if up_grp in self.connectivity and self.connectivity[up_grp] != down_grp:
                            print(f"Warning: Overwriting connectivity for upstream group {up_grp}.")
                        self.connectivity[up_grp] = down_grp
                        connections_found += 1
        print(f"    Built {connections_found} group-to-group connections: {self.connectivity}")
        print("Watershed loading and element module creation complete.")

    def get_element_module(self, element_id: int) -> torch.nn.Module | None:
        return self.element_modules.get(element_id)

    def get_element_properties(self, element_id: int) -> ElementProperties | None:
        return self.element_properties.get(element_id)

    def get_soil_parameters(self, element_id: int) -> SoilPropertiesIntermediate | None:
        return self.soil_parameters.get(element_id)
    
    def get_connectivity_map(self) -> dict[int, int]:
        return self.connectivity

    def get_group_config(self, group_id: int) -> dict | None:
        return self.group_elements_map.get(group_id)

class Watershed_:
    def __init__(self, 
                 basin_json_path: str, 
                 global_param_overrides: dict,
                 learnable_params_list: list[str],
                 device: torch.device, 
                 dtype: torch.dtype):
        """
        Loads and initializes the watershed, creating element modules.

        Args:
            basin_json_path (str): Path to the processed basin JSON file.
            global_param_overrides (dict): Global parameter overrides.
            learnable_params_list (list[str]): List of parameters to set requires_grad=True for.
            device (torch.device): PyTorch device for modules and tensors.
            dtype (torch.dtype): PyTorch dtype for modules and tensors.
        """
        self.basin_json_path = basin_json_path
        self.global_param_overrides = global_param_overrides
        self.learnable_params_list = learnable_params_list
        self.device = device
        self.dtype = dtype

        self.element_properties: dict[int, ElementProperties] = {}
        self.soil_parameters: dict[int, SoilPropertiesIntermediate] = {}
        self.element_modules: dict[int, torch.nn.Module] = {} # Stores PlaneElement or ChannelElement instances
        
        self.element_group_map: dict[int, int] = {} # {element_id: group_id}
        self.group_elements_map: dict[int, dict] = defaultdict(lambda: {'planes': [], 'channel': None})
        # Example: group_elements_map[group_id]['planes'] = [plane_id1, plane_id2]
        #          group_elements_map[group_id]['channel'] = channel_id
        
        self.simulation_order: list[int] = []
        self.connectivity: dict[int, int] = {} # {upstream_group_id: downstream_group_id}

        self._load_and_build_watershed()

    def _load_and_build_watershed(self):
        """
        Internal method to parse JSON, create properties, and instantiate element modules.
        """
        print(f"Loading and building watershed from: {self.basin_json_path}")
        try:
            with open(self.basin_json_path, 'r') as f:
                config_data = json.load(f)
        except FileNotFoundError:
            print(f"ERROR: Basin JSON file not found at {self.basin_json_path}")
            raise
        except json.JSONDecodeError:
            print(f"ERROR: Could not decode JSON from {self.basin_json_path}")
            raise
        
        if not config_data:
            raise ValueError("Basin JSON configuration is empty.")

        # --- Read Simulation Order and Connectivity (if defined directly) ---
        # Simulation order is critical
        simulation_order_str = config_data.get("simulation_order")
        if not simulation_order_str:
            raise ValueError("'simulation_order' not found or empty in basin JSON.")
        self.simulation_order = [int(gid) for gid in simulation_order_str]
        print(f"  Using predefined simulation order: {self.simulation_order}")

        # Connectivity can be inferred or read if explicitly present
        # For now, assuming connectivity is built by the SimulationEngine or inferred from element from/to nodes
        # If your JSON stores group connectivity directly, load it here.
        # self.connectivity = config_data.get("group_connectivity_map", {}) # Example
        # For now, we will reconstruct connectivity based on element from_node/to_node as in basin_run_dt.py
        
        node_ends_in_group = defaultdict(list)
        node_starts_in_group = {}
        all_basin_nodes = set() # For inferring connectivity

        groups_in_json = {k for k in config_data if k not in ["simulation_order", "simulation_params", "group_connectivity_map"]}

        number_processed_channel, number_processed_planes = 0, 0
        for group_id_str in tqdm(groups_in_json, desc="Processing groups and creating element modules"):
            group_id = int(group_id_str)
            group_data = config_data.get(group_id_str, {})

            # --- Process Channel ---
            channel_json_data = group_data.get('channel')
            if channel_json_data:
                props, soil_p = parse_element_override(
                    element_data_from_json=channel_json_data,
                    global_param_overrides=self.global_param_overrides,
                    device=self.device,
                    dtype=self.dtype,
                    learnable_params_list=self.learnable_params_list
                )
                if props and soil_p:
                    elem_id = props.element_id
                    self.element_properties[elem_id] = props
                    self.soil_parameters[elem_id] = soil_p
                    self.element_group_map[elem_id] = group_id
                    self.group_elements_map[group_id]['channel'] = elem_id
                    
                    channel_module = ChannelElement_(props, soil_p, self.device, self.dtype)
                    self.element_modules[elem_id] = channel_module.to(self.device) # Ensure module is on device
                    number_processed_channel += 1
                    # For connectivity inference
                    from_node = channel_json_data.get('from_node'); to_node = channel_json_data.get('to_node')
                    if from_node is not None: all_basin_nodes.add(from_node); node_starts_in_group[from_node] = group_id
                    if to_node is not None: all_basin_nodes.add(to_node); node_ends_in_group[to_node].append(group_id)
                else:
                    print(f"Warning: Could not parse channel for group {group_id}.")
            
            # --- Process Planes (Head and Side) ---
            for plane_type in ['head', 'side']:
                for plane_json_data in group_data.get(plane_type, []):
                    props, soil_p = parse_element_override(
                        element_data_from_json=plane_json_data,
                        global_param_overrides=self.global_param_overrides,
                        device=self.device, 
                        dtype=self.dtype,
                        learnable_params_list=self.learnable_params_list
                    )
                    if props and soil_p:
                        elem_id = props.element_id
                        self.element_properties[elem_id] = props
                        self.soil_parameters[elem_id] = soil_p
                        self.element_group_map[elem_id] = group_id
                        self.group_elements_map[group_id]['planes'].append(elem_id)

                        plane_module = PlaneElement_(props, soil_p, self.device, self.dtype)
                        self.element_modules[elem_id] = plane_module.to(self.device)
                        number_processed_planes += 1
                    else:
                        print(f"Warning: Could not parse {plane_type} plane for group {group_id}.")
        print(f"  Processed {number_processed_channel} channels and {number_processed_planes} planes.")
        # --- Infer Group Connectivity Map (same logic as in old basin_run_dt.py) ---
        print("  Inferring group connectivity map...")
        all_group_ids_parsed = set(self.element_group_map.values())
        outlet_node_id_ref = (max(all_group_ids_parsed) + 1) if all_group_ids_parsed else -1
        connections_found = 0
        for node_id_conn in all_basin_nodes:
            if node_id_conn == outlet_node_id_ref: continue
            up_grps_list = node_ends_in_group.get(node_id_conn, [])
            down_grp = node_starts_in_group.get(node_id_conn)
            if down_grp is not None and up_grps_list:
                for up_grp in up_grps_list:
                    if up_grp != down_grp:
                        if up_grp in self.connectivity and self.connectivity[up_grp] != down_grp:
                            print(f"Warning: Overwriting connectivity for upstream group {up_grp}.")
                        self.connectivity[up_grp] = down_grp
                        connections_found += 1
        print(f"    Built {connections_found} group-to-group connections: {self.connectivity}")
        print("Watershed loading and element module creation complete.")

    def get_element_module(self, element_id: int) -> torch.nn.Module | None:
        return self.element_modules.get(element_id)

    def get_element_properties(self, element_id: int) -> ElementProperties | None:
        return self.element_properties.get(element_id)

    def get_soil_parameters(self, element_id: int) -> SoilPropertiesIntermediate | None:
        return self.soil_parameters.get(element_id)
    
    def get_connectivity_map(self) -> dict[int, int]:
        return self.connectivity

    def get_group_config(self, group_id: int) -> dict | None:
        return self.group_elements_map.get(group_id)

class Watershed_im:
    def __init__(self, 
                 basin_json_path: str, 
                 global_param_overrides: dict,
                 learnable_params_list: list[str],
                 device: torch.device, 
                 dtype: torch.dtype):
        """
        Loads and initializes the watershed, creating element modules.

        Args:
            basin_json_path (str): Path to the processed basin JSON file.
            global_param_overrides (dict): Global parameter overrides.
            learnable_params_list (list[str]): List of parameters to set requires_grad=True for.
            device (torch.device): PyTorch device for modules and tensors.
            dtype (torch.dtype): PyTorch dtype for modules and tensors.
        """
        self.basin_json_path = basin_json_path
        self.global_param_overrides = global_param_overrides
        self.learnable_params_list = learnable_params_list
        self.device = device
        self.dtype = dtype

        self.element_properties: dict[int, ElementProperties] = {}
        self.soil_parameters: dict[int, SoilPropertiesIntermediate] = {}
        self.element_modules: dict[int, torch.nn.Module] = {} # Stores PlaneElement or ChannelElement instances
        
        self.element_group_map: dict[int, int] = {} # {element_id: group_id}
        self.group_elements_map: dict[int, dict] = defaultdict(lambda: {'planes': [], 'channel': None})
        # Example: group_elements_map[group_id]['planes'] = [plane_id1, plane_id2]
        #          group_elements_map[group_id]['channel'] = channel_id
        
        self.simulation_order: list[int] = []
        self.connectivity: dict[int, int] = {} # {upstream_group_id: downstream_group_id}

        self._load_and_build_watershed()

    def _load_and_build_watershed(self):
        """
        Internal method to parse JSON, create properties, and instantiate element modules.
        """
        print(f"Loading and building watershed from: {self.basin_json_path}")
        try:
            with open(self.basin_json_path, 'r') as f:
                config_data = json.load(f)
        except FileNotFoundError:
            print(f"ERROR: Basin JSON file not found at {self.basin_json_path}")
            raise
        except json.JSONDecodeError:
            print(f"ERROR: Could not decode JSON from {self.basin_json_path}")
            raise
        
        if not config_data:
            raise ValueError("Basin JSON configuration is empty.")

        # --- Read Simulation Order and Connectivity (if defined directly) ---
        # Simulation order is critical
        simulation_order_str = config_data.get("simulation_order")
        if not simulation_order_str:
            raise ValueError("'simulation_order' not found or empty in basin JSON.")
        self.simulation_order = [int(gid) for gid in simulation_order_str]
        print(f"  Using predefined simulation order: {self.simulation_order}")

        # Connectivity can be inferred or read if explicitly present
        # For now, assuming connectivity is built by the SimulationEngine or inferred from element from/to nodes
        # If your JSON stores group connectivity directly, load it here.
        # self.connectivity = config_data.get("group_connectivity_map", {}) # Example
        # For now, we will reconstruct connectivity based on element from_node/to_node as in basin_run_dt.py
        
        node_ends_in_group = defaultdict(list)
        node_starts_in_group = {}
        all_basin_nodes = set() # For inferring connectivity

        groups_in_json = {k for k in config_data if k not in ["simulation_order", "simulation_params", "group_connectivity_map"]}

        number_processed_channel, number_processed_planes = 0, 0
        for group_id_str in tqdm(groups_in_json, desc="Processing groups and creating element modules"):
            group_id = int(group_id_str)
            group_data = config_data.get(group_id_str, {})

            # --- Process Channel ---
            channel_json_data = group_data.get('channel')
            if channel_json_data:
                props, soil_p = parse_element_override(
                    element_data_from_json=channel_json_data,
                    global_param_overrides=self.global_param_overrides,
                    device=self.device,
                    dtype=self.dtype,
                    learnable_params_list=self.learnable_params_list
                )
                if props and soil_p:
                    elem_id = props.element_id
                    self.element_properties[elem_id] = props
                    self.soil_parameters[elem_id] = soil_p
                    self.element_group_map[elem_id] = group_id
                    self.group_elements_map[group_id]['channel'] = elem_id
                    
                    channel_module = ChannelElement_im(props, soil_p, self.device, self.dtype)
                    self.element_modules[elem_id] = channel_module.to(self.device) # Ensure module is on device
                    number_processed_channel += 1
                    # For connectivity inference
                    from_node = channel_json_data.get('from_node'); to_node = channel_json_data.get('to_node')
                    if from_node is not None: all_basin_nodes.add(from_node); node_starts_in_group[from_node] = group_id
                    if to_node is not None: all_basin_nodes.add(to_node); node_ends_in_group[to_node].append(group_id)
                else:
                    print(f"Warning: Could not parse channel for group {group_id}.")
            
            # --- Process Planes (Head and Side) ---
            for plane_type in ['head', 'side']:
                for plane_json_data in group_data.get(plane_type, []):
                    props, soil_p = parse_element_override(
                        element_data_from_json=plane_json_data,
                        global_param_overrides=self.global_param_overrides,
                        device=self.device, 
                        dtype=self.dtype,
                        learnable_params_list=self.learnable_params_list
                    )
                    if props and soil_p:
                        elem_id = props.element_id
                        self.element_properties[elem_id] = props
                        self.soil_parameters[elem_id] = soil_p
                        self.element_group_map[elem_id] = group_id
                        self.group_elements_map[group_id]['planes'].append(elem_id)

                        plane_module = PlaneElement_im(props, soil_p, self.device, self.dtype)
                        self.element_modules[elem_id] = plane_module.to(self.device)
                        number_processed_planes += 1
                    else:
                        print(f"Warning: Could not parse {plane_type} plane for group {group_id}.")
        print(f"  Processed {number_processed_channel} channels and {number_processed_planes} planes.")
        # --- Infer Group Connectivity Map (same logic as in old basin_run_dt.py) ---
        print("  Inferring group connectivity map...")
        all_group_ids_parsed = set(self.element_group_map.values())
        outlet_node_id_ref = (max(all_group_ids_parsed) + 1) if all_group_ids_parsed else -1
        connections_found = 0
        for node_id_conn in all_basin_nodes:
            if node_id_conn == outlet_node_id_ref: continue
            up_grps_list = node_ends_in_group.get(node_id_conn, [])
            down_grp = node_starts_in_group.get(node_id_conn)
            if down_grp is not None and up_grps_list:
                for up_grp in up_grps_list:
                    if up_grp != down_grp:
                        if up_grp in self.connectivity and self.connectivity[up_grp] != down_grp:
                            print(f"Warning: Overwriting connectivity for upstream group {up_grp}.")
                        self.connectivity[up_grp] = down_grp
                        connections_found += 1
        print(f"    Built {connections_found} group-to-group connections: {self.connectivity}")
        print("Watershed loading and element module creation complete.")

    def get_element_module(self, element_id: int) -> torch.nn.Module | None:
        return self.element_modules.get(element_id)

    def get_element_properties(self, element_id: int) -> ElementProperties | None:
        return self.element_properties.get(element_id)

    def get_soil_parameters(self, element_id: int) -> SoilPropertiesIntermediate | None:
        return self.soil_parameters.get(element_id)
    
    def get_connectivity_map(self) -> dict[int, int]:
        return self.connectivity

    def get_group_config(self, group_id: int) -> dict | None:
        return self.group_elements_map.get(group_id)
       
###### TBD
# # Example Usage (will be in run_simulation.py)
# if __name__ == '__main__':

#     import os
#     # This is just for demonstration.
#     # In a real run, these would come from config files or script arguments.
#     mock_device = torch.device('cpu')
#     mock_dtype = torch.float32
#     mock_learnable_params = ['man', 'ks']
#     mock_global_overrides = {
#         'plane': {'ks': 15.0 / (1000*3600)}, # Example override for plane Ks in m/s
#         'channel': {'man': 0.06}
#     }
    
#     # Create a dummy JSON file for testing
#     dummy_json_path = "dummy_basin.json"
#     dummy_basin_data = {
#         "simulation_order": ["1"],
#         "1": {
#             "channel": {
#                 "element_id": 101, "geom_type": "channel", "LEN": 100.0, "WID": 2.0, 
#                 "num_nodes": 11, "dx_avg": 10.0, "MAN": 0.04, "SS1": 1.0, "SS2": 1.0,
#                 "dx_segments": [10.0]*10, "node_x": [i*10.0 for i in range(11)], 
#                 "SL": [0.005]*11, "W0_nodes": [2.0]*11,
#                 "soil_params": {"Ks": 10.0, "theta_s": 0.4, "theta_r": 0.05, "theta_init_condition": 0.2, "HF_max": 50.0, "m_exponent": 0.5, "effective_depth": 100.0, "k_drain": 1e-7, "Smax": 1.0}
#             },
#             "side": [{
#                 "element_id": 102, "geom_type": "plane", "LEN": 50.0, "WID": 200.0,
#                 "num_nodes": 6, "dx_avg": 10.0, "MAN": 0.1,
#                 "dx_segments": [10.0]*5, "node_x": [i*10.0 for i in range(6)], "SL": [0.02]*6,
#                 "soil_params": {"Ks": 20.0, "theta_s": 0.42, "theta_r": 0.06, "theta_init_condition": 0.22, "HF_max": 60.0, "m_exponent": 0.5, "effective_depth": 80.0, "k_drain": 1e-7, "Smax": 1.0}
#             }]
#         }
#     }
#     with open(dummy_json_path, 'w') as f:
#         json.dump(dummy_basin_data, f, indent=4)

#     try:
#         print(f"Attempting to load watershed from: {dummy_json_path}")
#         watershed_obj = Watershed(
#             basin_json_path=dummy_json_path,
#             global_param_overrides=mock_global_overrides,
#             learnable_params_list=mock_learnable_params,
#             device=mock_device,
#             dtype=mock_dtype
#         )
#         print("\nWatershed object created successfully.")
#         print(f"Simulation order: {watershed_obj.simulation_order}")
#         print(f"Connectivity: {watershed_obj.connectivity}")
#         print(f"Number of element modules: {len(watershed_obj.element_modules)}")

#         for eid, module in watershed_obj.element_modules.items():
#             print(f"\nElement ID: {eid}, Type: {module.props.geom_type}, Module Device: {next(module.parameters()).device if list(module.parameters()) else module.area.device}")
#             print(f"  Props MAN: {module.props.MAN}, Requires Grad: {module.props.MAN.requires_grad}")
#             print(f"  Soil Ks: {module.soil_params.Ks}, Requires Grad: {module.soil_params.Ks.requires_grad}")
#             print(f"  Initial Area (buffer): {module.area}")
#             print(f"  Initial Theta_current (buffer): {module.theta_current}")

#     except Exception as e:
#         print(f"Error during watershed loading test: {e}")
#     finally:
#         if os.path.exists(dummy_json_path):
#             os.remove(dummy_json_path)