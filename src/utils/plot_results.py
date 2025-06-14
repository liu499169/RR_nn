# src/utils/plot_results.py

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import argparse
import os

def plot_simulation_results(hdf5_filepath: str, 
                            elements_to_plot: list[int] | None = None, 
                            plot_all_elements: bool = False,
                            output_dir: str = "plots"):
    """
    Reads simulation results from an HDF5 file and generates plots.

    Args:
        hdf5_filepath (str): Path to the HDF5 results file.
        elements_to_plot (list[int], optional): A list of specific element IDs to plot.
                                                If None and plot_all_elements is False,
                                                plots for a few example elements might be generated.
        plot_all_elements (bool): If True, attempts to plot for all elements (can be many plots).
        output_dir (str): Directory to save the generated plots.
    """
    if not os.path.exists(hdf5_filepath):
        print(f"Error: HDF5 file not found at {hdf5_filepath}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    with h5py.File(hdf5_filepath, 'r') as hf:
        print(f"Reading data from: {hdf5_filepath}")

        time_s = hf['time'][:]  # Simulation time in seconds
        # Convert time to minutes for easier plotting
        time_min = time_s / 60.0 
        
        # Convert time to datetime objects for better x-axis formatting (optional)
        # start_datetime = datetime(2024, 1, 1) # Dummy start date
        # datetimes = [start_datetime + timedelta(seconds=sec) for sec in time_s]

        rainfall_mmhr = hf['rainfall_intensity_mmhr'][:]

        element_ids_in_file = []
        for key in hf.keys():
            if key.startswith('element_'):
                try:
                    element_ids_in_file.append(int(key.split('_')[-1]))
                except ValueError:
                    print(f"Warning: Could not parse element ID from group name: {key}")
        
        if not element_ids_in_file:
            print("No element data found in the HDF5 file.")
            return

        print(f"Found data for {len(element_ids_in_file)} elements.")

        elements_for_plotting_final = []
        if plot_all_elements:
            elements_for_plotting_final = sorted(element_ids_in_file)
        elif elements_to_plot:
            elements_for_plotting_final = [eid for eid in elements_to_plot if eid in element_ids_in_file]
            if not elements_for_plotting_final:
                print(f"Warning: None of the specified elements_to_plot found. Plotting first few available.")
                elements_for_plotting_final = sorted(element_ids_in_file)[:min(3, len(element_ids_in_file))]
        else: # Default: plot first few (e.g., 3) elements if no specific ones are requested
            elements_for_plotting_final = sorted(element_ids_in_file)[:min(3, len(element_ids_in_file))]
            if elements_for_plotting_final:
                 print(f"No specific elements requested and not plotting all. Plotting for: {elements_for_plotting_final}")
            else:
                print("No elements to plot based on current settings.")
                return


        # --- Plot 1: Basin-wide Rainfall and Outlet Hydrograph (if outlet identifiable) ---
        # Identify potential outlet(s) - this is a simple heuristic
        # A more robust way would be to have outlet IDs passed or stored in HDF5 attributes
        potential_outlet_ids = []
        # Example: Assume elements with the largest IDs or specific geom_type are outlets
        # This needs to be adapted based on how you can identify your basin outlet(s) from the HDF5.
        # For now, let's try to plot the element with the largest ID if it's a channel.
        all_props = {eid: hf[f'element_{eid}'].attrs for eid in element_ids_in_file if f'element_{eid}' in hf}
        
        channel_elements = {eid: props for eid, props in all_props.items() if props.get('geom_type') == 'channel'}
        if channel_elements:
            # Simplistic: assume largest channel ID is an outlet.
            # A better way is if your simulation_engine or watershed identified and saved outlet_ids.
            outlet_id_to_plot = max(channel_elements.keys())
            potential_outlet_ids.append(outlet_id_to_plot)
            print(f"Identified potential outlet for hydrograph: Element {outlet_id_to_plot}")


        fig_basin, ax_basin1 = plt.subplots(figsize=(12, 6))
        color = 'tab:blue'
        ax_basin1.set_xlabel('Time (minutes)')
        ax_basin1.set_ylabel('Rainfall Intensity (mm/hr)', color=color)
        ax_basin1.plot(time_min, rainfall_mmhr, color=color, linestyle='-', label='Rainfall')
        ax_basin1.tick_params(axis='y', labelcolor=color)
        ax_basin1.invert_yaxis() # Rainfall often plotted downwards

        if potential_outlet_ids:
            ax_basin2 = ax_basin1.twinx()
            color = 'tab:red'
            ax_basin2.set_ylabel('Discharge (m³/s)', color=color)
            for outlet_id in potential_outlet_ids: # Plot multiple if identified
                outlet_group = hf.get(f'element_{outlet_id}')
                if outlet_group and 'discharge' in outlet_group:
                    # For channels, discharge is per node. Plot discharge at the last node.
                    outlet_q = outlet_group['discharge'][:, -1] # Discharge at last node
                    ax_basin2.plot(time_min, outlet_q, color=color, linestyle='--', label=f'Outlet Q (Elem {outlet_id})')
            ax_basin2.tick_params(axis='y', labelcolor=color)
            ax_basin2.set_ylim(bottom=0) # Discharge should be non-negative
        
        fig_basin.suptitle('Basin Rainfall and Outlet Hydrograph(s)', fontsize=16)
        # Combine legends if both axes are used
        lines1, labels1 = ax_basin1.get_legend_handles_labels()
        if potential_outlet_ids and ax_basin2:
            lines2, labels2 = ax_basin2.get_legend_handles_labels()
            ax_basin2.legend(lines1 + lines2, labels1 + labels2, loc='best')
        else:
            ax_basin1.legend(loc='best')
            
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(output_dir, "basin_summary_plot.png"))
        plt.close(fig_basin)
        print(f"Saved basin_summary_plot.png")


        # --- Plot 2: Detailed Plots for Selected Elements ---
        for element_id in elements_for_plotting_final:
            element_group_path = f'element_{element_id}'
            if element_group_path not in hf:
                print(f"Warning: Data for element {element_id} not found in HDF5 file.")
                continue

            print(f"  Plotting for element: {element_id}")
            elem_group = hf[element_group_path]
            props = elem_group.attrs
            geom_type = props.get('geom_type', 'unknown')
            num_nodes = props.get('num_nodes', 1) # Get num_nodes from attributes

            fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
            fig.suptitle(f'Element {element_id} (Type: {geom_type}, Nodes: {num_nodes})', fontsize=16)

            # Subplot 1: Discharge and Depth (at outlet node if multiple nodes)
            discharge_data = elem_group['discharge'][:]
            depth_data = elem_group['depth'][:]
            
            # For multi-node elements, plot value at the last node (outlet of element)
            q_to_plot = discharge_data[:, -1] if num_nodes > 1 and discharge_data.ndim == 2 else discharge_data
            d_to_plot = depth_data[:, -1] if num_nodes > 1 and depth_data.ndim == 2 else depth_data

            ax1_twin = axes[0].twinx()
            axes[0].plot(time_min, q_to_plot, 'b-', label='Discharge (m³/s) at outlet node')
            axes[0].set_ylabel('Discharge (m³/s)', color='blue')
            axes[0].tick_params(axis='y', labelcolor='blue')
            axes[0].set_ylim(bottom=0)
            
            ax1_twin.plot(time_min, d_to_plot * 1000, 'r:', label='Depth (mm) at outlet node') # Depth in mm
            ax1_twin.set_ylabel('Depth (mm)', color='red')
            ax1_twin.tick_params(axis='y', labelcolor='red')
            ax1_twin.set_ylim(bottom=0)
            axes[0].grid(True, axis='y', linestyle=':')
            lines, labels = axes[0].get_legend_handles_labels()
            lines2, labels2 = ax1_twin.get_legend_handles_labels()
            axes[0].legend(lines + lines2, labels + labels2, loc='best')


            # Subplot 2: Infiltration States
            theta_current = elem_group['theta_current'][:]
            F_cumulative_mm = elem_group['F_cumulative'][:] * 1000 # Convert m to mm
            infil_rate_mmhr = elem_group['infiltration_rate_ms'][:] * 3600 * 1000 # m/s to mm/hr

            ax2_twin = axes[1].twinx()
            axes[1].plot(time_min, theta_current, 'g-', label='Theta Current (m³/m³)')
            axes[1].set_ylabel('Soil Moisture Content (-)', color='green')
            axes[1].tick_params(axis='y', labelcolor='green')
            axes[1].set_ylim(0, max(0.5, np.max(theta_current)*1.1) if theta_current.size > 0 else 0.5)


            ax2_twin.plot(time_min, infil_rate_mmhr, 'm:', label='Infiltration Rate (mm/hr)')
            ax2_twin.set_ylabel('Infiltration Rate (mm/hr)', color='purple')
            ax2_twin.tick_params(axis='y', labelcolor='purple')
            ax2_twin.set_ylim(bottom=0)
            axes[1].grid(True, axis='y', linestyle=':')
            lines, labels = axes[1].get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            axes[1].legend(lines + lines2, labels + labels2, loc='best')


            # Subplot 3: Cumulative Infiltration and Rainfall (Basin-wide for comparison)
            axes[2].plot(time_min, rainfall_mmhr, 'c--', label='Basin Rainfall (mm/hr)')
            axes[2].plot(time_min, F_cumulative_mm, 'k-', label='Element Cum. Infiltration (mm)')
            axes[2].set_ylabel('Depth (mm) or Rate (mm/hr)')
            axes[2].set_xlabel('Time (minutes)')
            axes[2].legend(loc='best')
            axes[2].grid(True)
            axes[2].set_ylim(bottom=0)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(os.path.join(output_dir, f"element_{element_id}_details.png"))
            plt.close(fig)
        
        print(f"Plots saved to directory: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot simulation results from HDF5 file.")
    parser.add_argument("hdf5_file", help="Path to the HDF5 results file.")
    parser.add_argument("-e", "--elements", nargs='+', type=int, 
                        help="List of specific element IDs to plot (e.g., -e 101 102).")
    parser.add_argument("-a", "--all", action="store_true", 
                        help="Plot details for all elements found in the file.")
    parser.add_argument("-o", "--outdir", default="simulation_plots",
                        help="Directory to save output plots (default: simulation_plots).")
    
    args = parser.parse_args()

    plot_simulation_results(args.hdf5_file, 
                            elements_to_plot=args.elements, 
                            plot_all_elements=args.all,
                            output_dir=args.outdir)