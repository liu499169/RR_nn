# src/io/results_handler.py

import torch
import h5py
import os
# from collections import deque # Not strictly needed for simple batching, but useful if an async queue was used

class ResultSaver:
    """ 
    Handles saving simulation results to an HDF5 file incrementally (element-centric),
    with an option for batching writes to reduce I/O frequency.
    """
    def __init__(self, filepath: str, 
                 element_props_map: dict, # dict {element_id: ElementProperties namedtuple}
                 sim_duration_s: float, 
                 save_interval_s: float, 
                 rainfall_mmhr_cpu: torch.Tensor, # Expected to be a CPU tensor
                 batch_save_size: int = 5, 
                 enable_batching: bool = True):
        self.filepath = filepath
        self.save_interval_s = save_interval_s if save_interval_s > 0 else float('inf')
        self.element_props_map = element_props_map 
        self.enable_batching = enable_batching
        # If batching is disabled, flush immediately (batch size = 1)
        self.batch_save_size = batch_save_size if enable_batching and batch_save_size > 0 else 1
        
        self.num_flushed_steps = 0 # Tracks save points actually written to HDF5
        # Estimate initial allocation size for HDF5 datasets
        self.max_saves_allocated = int(sim_duration_s / self.save_interval_s) + 2 if self.save_interval_s != float('inf') else 2
        
        # Buffer for batching: stores tuples of (save_idx_in_hdf5, time_seconds_float, 
        # element_states_cpu_numpy_dict, element_infil_rates_float_dict)
        self._data_buffer = [] 

        print(f"Initializing HDF5 result file: {filepath}")
        print(f"  Est. Max Saves Allocated: {self.max_saves_allocated}, Save Interval: {self.save_interval_s:.2f} s")
        if self.enable_batching and self.batch_save_size > 1:
            print(f"  Batching enabled with size: {self.batch_save_size}")
        else:
            print(f"  Batching disabled (or batch size 1). Saving on each call to save_state.")

        self.h5file = h5py.File(filepath, 'w')
        self.time_ds = self.h5file.create_dataset('time', (self.max_saves_allocated,), maxshape=(None,), dtype='f4', chunks=True)
        self.h5file.create_dataset('rainfall_intensity_mmhr', data=rainfall_mmhr_cpu.numpy(), dtype='f4', compression="gzip")

        for element_id, props in self.element_props_map.items():
            grp_name = f'element_{element_id}'
            element_h5 = self.h5file.create_group(grp_name)
            # Handle cases where num_nodes might be 0 for an element if props allow it
            num_nodes = props.num_nodes if props.num_nodes > 0 else 1 
            
            time_chunk = min(100, self.max_saves_allocated) if self.max_saves_allocated > 0 else 100
            node_chunk = min(100, num_nodes)

            element_h5.create_dataset('depth', (self.max_saves_allocated, num_nodes), dtype='f4', chunks=(time_chunk, node_chunk), compression="gzip")
            element_h5.create_dataset('discharge', (self.max_saves_allocated, num_nodes), dtype='f4', chunks=(time_chunk, node_chunk), compression="gzip")
            element_h5.create_dataset('area', (self.max_saves_allocated, num_nodes), dtype='f4', chunks=(time_chunk, node_chunk), compression="gzip")
            element_h5.create_dataset('theta_current', (self.max_saves_allocated,), dtype='f4', chunks=(time_chunk,), compression="gzip")
            element_h5.create_dataset('F_cumulative', (self.max_saves_allocated,), dtype='f4', chunks=(time_chunk,), compression="gzip")
            element_h5.create_dataset('drying_cumulative', (self.max_saves_allocated,), dtype='f4', chunks=(time_chunk,), compression="gzip")
            element_h5.create_dataset('infiltration_rate_ms', (self.max_saves_allocated,), dtype='f4', chunks=(time_chunk,), compression="gzip")
            element_h5.create_dataset('max_cfl', (self.max_saves_allocated,), dtype='f4', chunks=(time_chunk,), compression="gzip")
            # Store properties as attributes
            for field in props._fields: # Assuming props is an ElementProperties namedtuple
                value = getattr(props, field)
                # Convert PyTorch tensors (expected to be on CPU if from ElementProperties after loading)
                if isinstance(value, torch.Tensor):
                    value = value.item() if value.numel() == 1 else value.tolist()
                if value is not None:
                    try: element_h5.attrs[field] = value
                    except TypeError: element_h5.attrs[field] = str(value) # Fallback to string

    def _prepare_data_for_buffer(self, 
                                 basin_element_states_gpu: dict, # States are from nn.Module buffers (GPU)
                                 step_element_infil_rates_ms_float: dict, # Already floats
                                 ) -> tuple[dict, dict]:
        """Converts GPU tensor states to CPU NumPy arrays/Python floats for buffering."""
        states_cpu_ready = {}
        for eid, gpu_module_states in basin_element_states_gpu.items():
            # gpu_module_states is {'flow': OverlandFlowState_on_GPU, 'infil': InfiltrationState_on_GPU}
            flow_s_gpu = gpu_module_states['flow']
            infil_s_gpu = gpu_module_states['infil']
            
            # Ensure num_nodes consistency for NumPy array creation
            props = self.element_props_map.get(eid)
            num_nodes_for_np = props.num_nodes if props and props.num_nodes > 0 else 1

            states_cpu_ready[eid] = {
                'depth': flow_s_gpu.depth.detach().cpu().numpy().reshape(num_nodes_for_np),
                'discharge': flow_s_gpu.discharge.detach().cpu().numpy().reshape(num_nodes_for_np),
                'area': flow_s_gpu.area.detach().cpu().numpy().reshape(num_nodes_for_np),
                'theta_current': infil_s_gpu.theta_current.detach().cpu().item(), # Scalar
                'F_cumulative': infil_s_gpu.F_cumulative.detach().cpu().item(),    # Scalar
                'drying_cumulative': infil_s_gpu.drying_cumulative.detach().cpu().item(), # Scalar
                'max_cfl': flow_s_gpu.max_cfl.detach().cpu().item()
            }
        return states_cpu_ready, step_element_infil_rates_ms_float.copy()

    def _flush_buffer_to_hdf5(self):
        """Writes all data currently in the _data_buffer to the HDF5 file."""
        if not self._data_buffer or self.h5file is None:
            return

        # print(f"Flushing batch of {len(self._data_buffer)} save points to HDF5...")
        for save_idx_hdf5, time_val_float, elem_states_cpu_data, elem_infil_r_float_map in self._data_buffer:
            if save_idx_hdf5 >= self.max_saves_allocated:
                new_max_allocated = save_idx_hdf5 + max(100, int(self.max_saves_allocated * 0.2))
                print(f"Resizing HDF5 datasets from {self.max_saves_allocated} to {new_max_allocated} (triggered by save_idx {save_idx_hdf5})")
                self.time_ds.resize((new_max_allocated,))
                for eid_resize, props_resize in self.element_props_map.items():
                    num_nodes_r = props_resize.num_nodes if props_resize.num_nodes > 0 else 1
                    grp_path = f'element_{eid_resize}'
                    if grp_path in self.h5file:
                        for ds_name in ['depth', 'discharge', 'area']: self.h5file[grp_path][ds_name].resize((new_max_allocated, num_nodes_r))
                        for ds_name_s in ['theta_current', 'F_cumulative', 'infiltration_rate_ms', 'drying_cumulative','max_cfl']: 
                            self.h5file[grp_path][ds_name_s].resize((new_max_allocated,))
                self.max_saves_allocated = new_max_allocated
            
            try:
                self.time_ds[save_idx_hdf5] = time_val_float
                for element_id, states_data in elem_states_cpu_data.items():
                    element_h5 = self.h5file.get(f'element_{element_id}')
                    if element_h5:
                        element_h5['depth'][save_idx_hdf5, :] = states_data['depth']
                        element_h5['discharge'][save_idx_hdf5, :] = states_data['discharge']
                        element_h5['area'][save_idx_hdf5, :] = states_data['area']
                        element_h5['theta_current'][save_idx_hdf5] = states_data['theta_current']
                        element_h5['F_cumulative'][save_idx_hdf5] = states_data['F_cumulative']
                        element_h5['drying_cumulative'][save_idx_hdf5] = states_data['drying_cumulative']
                        element_h5['infiltration_rate_ms'][save_idx_hdf5] = elem_infil_r_float_map.get(element_id, 0.0)
                        element_h5['max_cfl'][save_idx_hdf5] = states_data.get('max_cfl', 0.0)
                self.num_flushed_steps += 1
            except Exception as e:
                print(f"Error writing buffered data to HDF5 at save_idx {save_idx_hdf5}, time {time_val_float:.2f}s: {e}")
        
        self._data_buffer.clear()

    def save_state(self, current_save_trigger_idx: int, 
                   time_seconds: float, 
                   basin_element_states_gpu: dict, # This now comes from module buffers
                   step_element_infil_rates_ms_float: dict # This is already float dict
                   ): 
        """Buffers the current state. If buffer is full or batching disabled, flushes to HDF5."""
        if self.h5file is None: return

        elem_states_cpu_data, elem_infil_r_float_copy = self._prepare_data_for_buffer(
            basin_element_states_gpu, step_element_infil_rates_ms_float
        )
        self._data_buffer.append((current_save_trigger_idx, time_seconds, 
                                  elem_states_cpu_data, elem_infil_r_float_copy))
        
        if len(self._data_buffer) >= self.batch_save_size:
            self._flush_buffer_to_hdf5()

    def finalize(self):
        """Flushes any remaining data, trims HDF5 datasets, and closes the file."""
        if self.h5file:
            if self._data_buffer:
                print("Finalizing ResultSaver: Flushing remaining data from buffer...")
                self._flush_buffer_to_hdf5()
            
            try:
                final_actual_size = self.num_flushed_steps
                print(f"Finalizing HDF5 file. Trimming datasets to actual flushed size: {final_actual_size}.")
                
                self.time_ds.resize((final_actual_size,))
                for eid_f, props_f in self.element_props_map.items():
                    num_nodes_f = props_f.num_nodes if props_f.num_nodes > 0 else 1
                    grp_path = f'element_{eid_f}'
                    if grp_path in self.h5file:
                        for ds_name in ['depth', 'discharge', 'area']: self.h5file[grp_path][ds_name].resize((final_actual_size, num_nodes_f))
                        for ds_name_s in ['theta_current', 'F_cumulative', 'infiltration_rate_ms', 'drying_cumulative', 'max_cfl']: 
                            self.h5file[grp_path][ds_name_s].resize((final_actual_size,))
                
                self.h5file.attrs['total_steps_written_to_hdf5'] = final_actual_size
                self.h5file.attrs['save_interval_seconds'] = self.save_interval_s
                if self.enable_batching and self.batch_save_size > 1:
                    self.h5file.attrs['batch_save_size_used'] = self.batch_save_size
                
                self.h5file.close()
                print(f"HDF5 result file closed: {self.filepath}")
            except Exception as e:
                print(f"Error during HDF5 finalization: {e}")
                if self.h5file: # Attempt to close if already open
                    try: self.h5file.close()
                    except Exception as e_close: print(f"Error during emergency close: {e_close}")