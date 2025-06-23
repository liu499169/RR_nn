import pandas as pd
import numpy as np
import h5py
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch

def load_event_from_hdf(file_path, event_name):
    """Loads event data and metrics from your specific HDF5 format."""
    with pd.HDFStore(file_path, mode='r') as store:
        try:
            event_data = store[f'{event_name}/data']
            # metrics = store[f'{event_name}/metrics'] # We don't need metrics for this script
        except KeyError:
            raise KeyError(f"Could not find event '{event_name}' in HDF5 file '{file_path}'.")
    return event_data

from abc import ABC, abstractmethod
import torch
import h5py
import json

# ==============================================================================
# The "Contract" or Abstract Base Class
# ==============================================================================
class AbstractRainfallManager(ABC):
    """
    Abstract base class for all rainfall managers.
    Defines the interface that the simulation engine will use.
    """
    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        # A pre-made zero tensor is useful for elements with no rainfall data
        self.zero_rain_tensor = torch.tensor(0.0, device=self.device, dtype=self.dtype)

    @abstractmethod
    def get_rainfall_at_time(self, time_s: float) -> dict[int, torch.Tensor]:
        """
        The core method called by the simulation engine at each timestep.

        Args:
            time_s (float): The current simulation time in seconds.

        Returns:
            dict[int, torch.Tensor]: A dictionary mapping each element_id to its
                                     current rainfall rate as a 0-dim tensor (in m/s).
                                     If an element isn't in the dict, the engine
                                     should assume its rainfall is zero.
        """
        pass

# ==============================================================================
# The "Pre-loaded Timeseries" Implementation
# ==============================================================================
class PreloadedRainfallManager(AbstractRainfallManager):
    """
    A rainfall manager that pre-loads all element rainfall timeseries into
    a single GPU tensor for maximum lookup performance during the simulation.
    
    It expects an HDF5 file created by a pre-processing script.
    """
    def __init__(self, element_rainfall_path: str, device: torch.device, dtype: torch.dtype):
        """
        Args:
            element_rainfall_path (str): Path to the pre-processed HDF5 file.
            device (torch.device): The PyTorch device to load tensors onto.
            dtype (torch.dtype): The PyTorch data type to use for tensors.
        """
        super().__init__(device, dtype)
        print(f"Initializing PreloadedRainfallManager from: {element_rainfall_path}")

        try:
            with h5py.File(element_rainfall_path, 'r') as hf:
                # Load the time vector (seconds since simulation start)
                self.times_s = torch.from_numpy(hf['time_s'][:]).to(self.device, self.dtype)
                
                # Load the rainfall rates (m/s) for all elements at all timesteps
                # Shape: (num_elements, num_timesteps)
                self.rates_ms = torch.from_numpy(hf['rainfall_rates_ms'][:]).to(self.device, self.dtype)
                
                # Load the mapping from row index to element ID from the HDF5 attribute
                if 'element_id_map' not in hf.attrs:
                    raise KeyError("HDF5 file must contain 'element_id_map' as a JSON string attribute.")
                
                element_ids = json.loads(hf.attrs['element_id_map'])
                # Create the mapping dictionary for quick lookups
                self.element_id_to_row_idx = {int(eid): i for i, eid in enumerate(element_ids)}
        
        except FileNotFoundError:
            print(f"FATAL ERROR: Pre-processed rainfall file not found at {element_rainfall_path}")
            raise
        except Exception as e:
            print(f"FATAL ERROR: Failed to load or parse pre-processed rainfall file.")
            raise e

        # Pre-calculate to avoid re-doing it in the loop
        self.max_time_idx = len(self.times_s) - 2

        print(f"Successfully pre-loaded rainfall timeseries for {self.rates_ms.shape[0]} elements.")
        vram_mb = self.rates_ms.nelement() * self.rates_ms.element_size() / (1024 * 1024)
        print(f"  - VRAM usage for rainfall data: {vram_mb:.2f} MB")

    def get_rainfall_at_time(self, time_s: float) -> dict[int, torch.Tensor]:
        """
        Performs an extremely fast, GPU-based interpolation to get the rainfall
        rate for every element at the requested simulation time.
        
        This method is called frequently and is performance-critical.
        """
        # Use torch.searchsorted for a highly optimized binary search on the GPU.
        # This finds the index of the last time point that is less than or equal to time_s.
        idx = torch.searchsorted(self.times_s, time_s, right=True) - 1
        
        # Clamp the index to ensure it and idx+1 are always valid.
        # This handles edge cases at the very beginning and end of the simulation.
        idx = torch.clamp(idx, 0, self.max_time_idx)
        
        # Get the bracketing time points
        t0, t1 = self.times_s[idx], self.times_s[idx+1]
        
        # Get the rainfall rate vectors for ALL elements at times t0 and t1.
        # This is a fast slicing operation.
        r0_vec = self.rates_ms[:, idx]
        r1_vec = self.rates_ms[:, idx+1]
        
        # Perform linear interpolation.
        # Handle the case where t0 and t1 are the same to avoid division by zero.
        time_diff = t1 - t0
        if time_diff < 1e-9:
            # If the time interval is zero, just use the rate at t0.
            current_rates_vec = r0_vec
        else:
            interp_factor = (time_s - t0) / time_diff
            current_rates_vec = r0_vec + interp_factor * (r1_vec - r0_vec)
        
        # This is the fastest way to construct the dictionary the engine needs.
        # We create a dictionary mapping every known element ID to its interpolated rate.
        return {eid: current_rates_vec[row_idx] for eid, row_idx in self.element_id_to_row_idx.items()}
    
# ==============================================================================
# Main pre-processing function (UPDATED to use HDF5 input)
# ==============================================================================
def convert_gage_data_to_element_timeseries(
    raw_event_hdf_path: str,
    event_name_in_hdf: str,
    basin_json_path: str,
    gage_locations_path: str,
    output_path: str,
    idw_power: int = 2
):
    """
    Reads raw gage data from your HDF5 event file and watershed element data,
    then creates a new HDF5 file containing the rainfall timeseries for every
    single element, calculated using Inverse Distance Weighting (IDW).
    """
    print("--- Starting Rainfall Pre-processing ---")

    # 1. Load Raw Gage Data using your provided HDF5 loading function
    print(f"Loading raw gage data from: {raw_event_hdf_path} (Event: {event_name_in_hdf})")
    df_gages_raw = load_event_from_hdf(raw_event_hdf_path, event_name_in_hdf)
    
    # Identify gage columns (all columns except 'sum' and 'Q_mm')
    gage_cols = [col for col in df_gages_raw.columns if col not in ['sum', 'Q_mm']]
    df_gages = df_gages_raw[gage_cols]
    print(f"Found {len(gage_cols)} gages: {gage_cols}")

    # 2. Convert gage rainfall depth (mm) to rainfall rate (m/s)
    timestep_s = (df_gages.index[1] - df_gages.index[0]).total_seconds()
    print(f"Detected rainfall data interval: {timestep_s / 60:.1f} minutes")
    df_rates_ms = (df_gages / 1000.0) / timestep_s
    
    # 3. Load Gage and Element Locations
    print(f"Loading locations from: {gage_locations_path} and {basin_json_path}")
    with open(gage_locations_path, 'r') as f:
        gage_locations = json.load(f)
    
    element_locations = {}
    with open(basin_json_path, 'r') as f:
        basin_data = json.load(f)
        for group_id_str, group_data in basin_data.items():
            if not group_id_str.isdigit(): continue
            for elem_type in ['channel', 'head', 'side']:
                elements = group_data.get(elem_type, []); #  print(elements)
                if elem_type == 'channel' and elements: elements = [elements]
                for elem_json in elements:
                    elem_id = elem_json['element_id']
                    # Assuming you will add x_centroid/y to your GIS pre-processing
                    if 'x_centroid' in elem_json and 'y_centroid' in elem_json:
                        element_locations[elem_id] = (elem_json['x_centroid'], elem_json['y_centroid'])
                    else:
                        # You MUST have coordinates for this to work. This is a placeholder.
                        # Consider adding a warning or error if centroids are missing.
                        print(f"Warning: Centroid not found for element {elem_id}. Using placeholder (0,0).")
                        element_locations[elem_id] = (0, 0)

    # Prepare coordinate arrays for fast computation
    gage_ids = df_rates_ms.columns.tolist()
    gage_coords = np.array([gage_locations[gid] for gid in gage_ids])
    
    element_ids = list(element_locations.keys())
    element_coords = np.array([element_locations[eid] for eid in element_ids])
    
    # 4. Perform Inverse Distance Weighting (IDW)
    print(f"Performing IDW for {len(element_ids)} elements and {len(df_rates_ms)} timesteps...")
    distances = np.linalg.norm(element_coords[:, np.newaxis, :] - gage_coords, axis=2)
    weights = 1.0 / (distances**idw_power + 1e-9)
    weights /= np.sum(weights, axis=1, keepdims=True)
    
    gage_rates_all_times = df_rates_ms.to_numpy()
    element_rates_all_times = (gage_rates_all_times @ weights.T).T
    
    # 5. Save the prepared data to a new HDF5 file
    print(f"Saving prepared element rainfall timeseries to: {output_path}")
    with h5py.File(output_path, 'w') as hf:
        sim_time_s = (df_rates_ms.index - df_rates_ms.index[0]).total_seconds()
        hf.create_dataset('time_s', data=sim_time_s)
        hf.create_dataset('rainfall_rates_ms', data=element_rates_all_times)
        hf.attrs['element_id_map'] = json.dumps(element_ids)

    print("--- Rainfall Pre-processing Complete ---")


def save_observed_discharge(
    raw_event_hdf_path: str,
    event_name_in_hdf: str,
    output_path: str,
    basin_area_m2: float = 41677010.4408
):
    """
    Reads observed discharge from the raw HDF5 event file, converts it
    to cms, and saves it to the new processed HDF5 file.
    """
    print(f"Processing and saving observed discharge for event {event_name_in_hdf}...")
    df_raw = load_event_from_hdf(raw_event_hdf_path, event_name_in_hdf)
    
    if 'Q_mm' not in df_raw.columns:
        print("Warning: 'Q_mm' column not found. Skipping discharge saving.")
        return
        
    timestep_s = (df_raw.index[1] - df_raw.index[0]).total_seconds()
    volume_m3 = (df_raw['Q_mm'] / 1000.0) * basin_area_m2
    q_cms = volume_m3 / timestep_s
    
    with h5py.File(output_path, 'a') as hf: # Append to the existing file
        sim_time_s = (df_raw.index - df_raw.index[0]).total_seconds()
        obs_group = hf.require_group("observed_data")
        obs_group.create_dataset('time_s', data=sim_time_s)
        obs_group.create_dataset('discharge_cms', data=q_cms.to_numpy())
        
    print("Observed discharge saved successfully in cms.")

def plot_simulation_results(
    prepared_rainfall_h5_path: str,
    basin_area_m2: float,
    target_element_id: int,
    output_figure_prefix: str,
    event_start_time_str: str # e.g., '2008-01-04 20:00:00'
):
    """
    Generates a suite of plots to visualize rainfall input and model output.

    Args:
        prepared_rainfall_h5_path (str): Path to the HDF5 file with pre-processed rainfall
                                         and observed discharge.
        simulation_output_h5_path (str): Path to the HDF5 file generated by the simulation.
        basin_area_m2 (float): The total basin area in square meters.
        target_element_id (int): A specific element ID to plot individually.
        output_figure_prefix (str): A prefix for the saved figure filenames.
    """
    print("--- Starting Result Visualization ---")

    # ==========================================================================
    # 1. Load All Necessary Data
    # ==========================================================================
    print("Loading data from HDF5 files...")
    event_start_time = pd.to_datetime(event_start_time_str)

    # --- Load Pre-processed Rainfall and Observed Q ---
    with h5py.File(prepared_rainfall_h5_path, 'r') as hf:
        # Load basin-average rainfall rates
        rain_rates_ms = hf['rainfall_rates_ms'][:] # Shape: (num_elements, num_timesteps)
        basin_avg_rain_rate_ms = np.mean(rain_rates_ms, axis=0)
        
        # Load time vector and convert to hours for plotting
        time_s = hf['observed_data']['time_s'][:]
        # time_hr = time_s / 3600.0
        datetimes = event_start_time + pd.to_timedelta(time_s, unit='s')
        
        # Load observed discharge in cms
        observed_q_cms = hf['observed_data']['discharge_cms'][:]
        
        # Load rainfall for the specific target element
        element_id_map = json.loads(hf.attrs['element_id_map'])
        if target_element_id not in element_id_map:
            raise ValueError(f"Target element ID {target_element_id} not found in rainfall file.")
        target_elem_idx = element_id_map.index(target_element_id)
        target_elem_rain_rate_ms = rain_rates_ms[target_elem_idx, :]

    # ==========================================================================
    # 2. Perform Unit Conversions for Plotting
    # ==========================================================================
    print("Performing unit conversions...")
    
    # --- Convert Rates (m/s) to Depths (mm) over a 5-minute interval ---
    # This is for the plots that show depth in mm.
    TIMESTEP_S = time_s[1] - time_s[0] # Assuming constant timestep
    
    # For basin average
    basin_avg_rain_mm = (basin_avg_rain_rate_ms * TIMESTEP_S) * 1000
    
    # For observed and simulated discharge
    # Q (mm) = (Q (m3/s) * timestep (s)) / Area (m2) * 1000 (mm/m)
    observed_q_mm = (observed_q_cms * TIMESTEP_S / basin_area_m2) * 1000
    
    # For target element
    target_elem_rain_mm = (target_elem_rain_rate_ms * TIMESTEP_S) * 1000

    # --- Convert Rates (m/s) to mm/hr ---
    basin_avg_rain_mmhr = basin_avg_rain_rate_ms * 3600 * 1000
    target_elem_rain_mmhr = target_elem_rain_rate_ms * 3600 * 1000

    # ==========================================================================
    # 3. Generate the Plots
    # ==========================================================================
    print("Generating plots...")

    # --- Plot 1: Basin Average - Rates (mm/hr and cms) ---
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_title('Basin Average Rainfall vs. Outlet Discharge (Rates)')
    ax1.set_xlabel('Time (hours)')
    
    # Plot rainfall on the left y-axis
    # Calculate the width of the bars for the 5-minute interval
    bar_width_days = TIMESTEP_S / (24 * 3600)
    ax1.bar(datetimes, basin_avg_rain_mmhr, width=bar_width_days, color='black', alpha=0.8, label='Precipitation (mm)')
    ax1.set_ylim(bottom=basin_avg_rain_mmhr.max() * 1.5, top=0) # Invert the axis
    ax1.tick_params(axis='y', labelcolor='black')

    # ax1.set_ylabel('Rainfall Intensity (mm/hr)', color='blue')
    # ax1.plot(datetimes, basin_avg_rain_mmhr, color='blue', alpha=0.7, label='Basin Avg. Rainfall')
    # ax1.tick_params(axis='y', labelcolor='blue')

    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Create a second y-axis for discharge
    ax2 = ax1.twinx()
    ax2.set_ylabel('Discharge (cms)', color='blue')
    ax2.plot(datetimes, observed_q_cms, 'blue', label='Observed Q')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    fig1.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    fig1.tight_layout()
    plt.savefig(f"{output_figure_prefix}_plot1_basin_rates.png", dpi=300)
    plt.close(fig1)
    print("Saved Plot 1: Basin Rates")

    # --- Plot 2: Basin Average - Depths (mm) ---
    fig2, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_title('Basin Average Rainfall vs. Outlet Discharge (Depth per 5-min Interval)')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Depth (mm)')

    # ax1.plot(datetimes, basin_avg_rain_mm, color='blue', alpha=0.7, label='Basin Avg. Rainfall (mm)')
    bar_width_days = TIMESTEP_S / (24 * 3600)
    ax1.bar(datetimes, basin_avg_rain_mm, width=bar_width_days, color='black', alpha=0.8, label='Precipitation (mm)')
    ax1.set_ylim(bottom=basin_avg_rain_mm.max() * 1.5, top=0) # Invert the axis
    ax1.tick_params(axis='y', labelcolor='black')

    # Create a second y-axis for discharge
    ax2 = ax1.twinx()
    ax2.set_ylabel('Discharge (cms)', color='blue')
    ax2.plot(datetimes, observed_q_mm, 'blue', label='Observed Q')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    fig2.tight_layout()
    plt.savefig(f"{output_figure_prefix}_plot2_basin_depths.png", dpi=300)
    plt.close(fig2)
    print("Saved Plot 2: Basin Depths")

    # --- Plot 3: Target Element - Rates (mm/hr and cms) ---
    fig3, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_title(f'Element {target_element_id} Rainfall vs. Outlet Discharge (Rates)')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Rainfall Intensity (mm/hr)', color='green')
    bar_width_days = TIMESTEP_S / (24 * 3600)
    ax1.bar(datetimes, target_elem_rain_mmhr, width=bar_width_days, color='green', alpha=0.8, label='Element {target_element_id} Rainfall (mm/hr)')
    ax1.set_ylim(bottom=target_elem_rain_mmhr.max() * 1.5, top=0) # Invert the axis
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Discharge (cms)', color='blue')
    ax2.plot(datetimes, observed_q_cms, 'blue', label='Observed Q')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    fig3.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    fig3.tight_layout()
    plt.savefig(f"{output_figure_prefix}_plot3_element_{target_element_id}_rates.png", dpi=300)
    plt.close(fig3)
    print(f"Saved Plot 3: Element {target_element_id} Rates")

    # --- Plot 4: Target Element - Depths (mm) ---
    fig4, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_title(f'Element {target_element_id} Rainfall vs. Outlet Discharge (Depth per 5-min Interval)')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Depth (mm)')
    bar_width_days = TIMESTEP_S / (24 * 3600)
    ax1.bar(datetimes, target_elem_rain_mm, width=bar_width_days, color='green', alpha=0.8, label='Element {target_element_id} (mm)')
    ax1.set_ylim(bottom=target_elem_rain_mm.max() * 1.5, top=0) # Invert the axis
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Discharge (mm)', color='blue')
    ax2.plot(datetimes, observed_q_mm, 'blue', label='Observed Q')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    fig4.tight_layout()
    plt.savefig(f"{output_figure_prefix}_plot4_element_{target_element_id}_depths.png", dpi=300)
    plt.close(fig4)
    print(f"Saved Plot 4: Element {target_element_id} Depths")

    print("--- Visualization Complete ---")


# ==============================================================================
if __name__ == '__main__':
    # --- Configuration ---
    EVENT_ID = 'E2008_1'
    EVENT_DATE = 'Jan05'
    EVENT_START_TIME = '2008-01-04 20:00:00'
    EVENT_NAME_HDF = f'{EVENT_ID}_{EVENT_DATE}'
    BASIN_AREA_M2 = 41677010.4408  # Example basin area in square meters
    TARGET_ELEMENT_ID = 21  # Example element ID to visualize
    FIGURE_PREFIX = 'EventCheck_E2008_1'

    # --- Input File Paths ---
    RAW_EVENT_HDF = f'data/Events/uAS/{EVENT_NAME_HDF}.h5'
    BASIN_JSON = 'data/watershed_processed/FromGIS_d2k2_eqdx_20m.json'
    GAGE_LOCATIONS_JSON = 'data/Events/uAS/gage_locations.json' # 

    # --- Output File Path ---
    OUTPUT_H5 = f'data/Events/uAS/{EVENT_NAME_HDF}_prepared.h5'

    # --- Step 1: Create the main element rainfall file ---
    convert_gage_data_to_element_timeseries(
        raw_event_hdf_path=RAW_EVENT_HDF,
        event_name_in_hdf=EVENT_NAME_HDF,
        basin_json_path=BASIN_JSON,
        gage_locations_path=GAGE_LOCATIONS_JSON,
        output_path=OUTPUT_H5
    )
    
    # --- Step 2: Append the observed discharge to the same file ---
    save_observed_discharge(
        raw_event_hdf_path=RAW_EVENT_HDF,
        event_name_in_hdf=EVENT_NAME_HDF,
        output_path=OUTPUT_H5
    )

    plot_simulation_results(
        prepared_rainfall_h5_path=OUTPUT_H5,
        basin_area_m2=BASIN_AREA_M2,
        target_element_id=TARGET_ELEMENT_ID,
        output_figure_prefix=FIGURE_PREFIX,
        event_start_time_str=EVENT_START_TIME
    )