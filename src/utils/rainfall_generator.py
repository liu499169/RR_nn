# src/utils/rainfall_generator.py

import torch
import numpy as np # For linspace if preferred, though torch.linspace is also good

# --- Constants (can be moved to a global config or passed if they vary) ---
MMHR_TO_MS = 1.0 / (1000.0 * 3600.0)
MIN_TO_S = 60.0
EPSILON = 1e-9 

def generate_triangular_rainfall(
    total_sim_duration_min: float,
    rain_event_duration_min: float,
    time_to_peak_min: float,
    peak_intensity_mmhr: float,
    dt_min_for_gen: float, # High-resolution dt for generating smooth rainfall profile
    save_interval_min: float | None = None # Interval for the mm/hr save profile (optional)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Generates a triangular rainfall hyetograph.

    Args:
        total_sim_duration_min (float): Total simulation duration (minutes).
        rain_event_duration_min (float): Duration of the rainfall event (minutes).
        time_to_peak_min (float): Time from the start of the event to peak intensity (minutes).
        peak_intensity_mmhr (float): Peak rainfall intensity (mm/hr).
        dt_min_for_generation (float): Time step for generating the high-resolution
                                       rainfall profile (minutes). Used for interpolation.
        save_interval_min (float, optional): Interval for generating the mm/hr profile
                                             for saving/metadata. If None, this profile isn't generated.

    Returns:
        tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            - rain_times_s_for_interp (torch.Tensor): 
                Time points (in seconds from start of simulation) for the high-resolution
                rainfall profile. 1D tensor.
            - rain_rates_ms_for_interp (torch.Tensor): 
                Rainfall intensities (in m/s) corresponding to rain_times_s_for_interp.
                These are instantaneous rates at those time points. 1D tensor.
            - rain_rates_mmhr_for_save (Optional[torch.Tensor]): 
                Rainfall intensities (in mm/hr) at save_interval_min points,
                or None if save_interval_min is not provided. 1D tensor.
    """
    if time_to_peak_min < 0 or time_to_peak_min > rain_event_duration_min:
        raise ValueError("Time to peak must be within the rain event duration.")
    if dt_min_for_gen <= 0:
        raise ValueError("dt_min_for_generation must be positive.")

    # --- Generate High-Resolution Profile for Interpolation ---
    num_steps_gen = int(round(total_sim_duration_min / dt_min_for_gen)) + 1
    # Time points in seconds from the start of the simulation
    rain_times_s_for_interp = torch.linspace(
        0, 
        total_sim_duration_min * MIN_TO_S, 
        steps=num_steps_gen, 
        dtype=torch.float32 # Assuming DTYPE is float32, can be parameterized
    )

    rain_rates_ms_at_times = torch.zeros(num_steps_gen, dtype=torch.float32)

    # Calculate rates for each time point in the high-resolution series
    for i, t_s in enumerate(rain_times_s_for_interp):
        t_min_current = t_s / MIN_TO_S # Current time in minutes
        
        current_intensity_mmhr = 0.0
        if t_min_current < time_to_peak_min: # Rising limb
            current_intensity_mmhr = (t_min_current / time_to_peak_min) * peak_intensity_mmhr
        elif t_min_current < rain_event_duration_min: # Falling limb
            if rain_event_duration_min - time_to_peak_min > EPSILON: # Avoid division by zero if peak is at end
                current_intensity_mmhr = (1.0 - (t_min_current - time_to_peak_min) / 
                                          (rain_event_duration_min - time_to_peak_min)) * peak_intensity_mmhr
            else: # Peak is at the end of the event duration
                current_intensity_mmhr = peak_intensity_mmhr if t_min_current == time_to_peak_min else 0.0

        rain_rates_ms_at_times[i] = current_intensity_mmhr * MMHR_TO_MS
    
    rain_rates_ms_for_interp = torch.clamp(rain_rates_ms_at_times, min=0.0)


    # --- Generate Profile for Saving (Optional) ---
    rain_rates_mmhr_for_save = None
    if save_interval_min is not None and save_interval_min > 0:
        num_steps_save = int(round(total_sim_duration_min / save_interval_min)) + 1
        save_times_s = torch.linspace(
            0, 
            total_sim_duration_min * MIN_TO_S, 
            steps=num_steps_save,
            dtype=torch.float32
        )
        
        # Interpolate from the high-resolution ms profile to get mm/hr at save intervals
        # This requires rain_times_s_for_interp to be on CPU for np.interp
        # Note: torch.searchsorted can also be used for GPU-based index finding if needed for extreme performance
        # but np.interp is convenient here.
        interp_rates_ms = np.interp(
            save_times_s.cpu().numpy(), 
            rain_times_s_for_interp.cpu().numpy(), 
            rain_rates_ms_for_interp.cpu().numpy()
        )
        rain_rates_mmhr_for_save = torch.from_numpy(interp_rates_ms / MMHR_TO_MS).to(dtype=torch.float32)
        rain_rates_mmhr_for_save = torch.clamp(rain_rates_mmhr_for_save, min=0.0)

    return rain_times_s_for_interp.cpu(), rain_rates_ms_for_interp, rain_rates_mmhr_for_save

# --- Example Usage (for testing this module standalone) ---
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    total_sim_min = 120.0
    event_dur_min = 60.0
    peak_t_min = 20.0
    peak_i_mmhr = 50.0
    gen_dt_min = 0.1 # Generate a fine-grained profile
    save_int_min = 5.0  # For the mm/hr save profile

    times_s_interp, rates_ms_interp, rates_mmhr_save = generate_triangular_rainfall(
        total_sim_duration_min=total_sim_min,
        rain_event_duration_min=event_dur_min,
        time_to_peak_min=peak_t_min,
        peak_intensity_mmhr=peak_i_mmhr,
        dt_min_for_generation=gen_dt_min,
        save_interval_min=save_int_min
    )

    print(f"Generated high-res times for interpolation (s): {times_s_interp.shape}, first few: {times_s_interp[:5]}")
    print(f"Generated high-res rates for interpolation (m/s): {rates_ms_interp.shape}, first few: {rates_ms_interp[:5]}")
    if rates_mmhr_save is not None:
        print(f"Generated rates for saving (mm/hr): {rates_mmhr_save.shape}, first few: {rates_mmhr_save[:5]}")

        # Plotting for verification
        fig, ax1 = plt.subplots(figsize=(10, 5))

        color = 'tab:blue'
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Rainfall Rate (m/s)', color=color)
        ax1.plot(times_s_interp.numpy() / MIN_TO_S, rates_ms_interp.numpy(), color=color, linestyle='-', label=f'High-res (m/s) dt={gen_dt_min}min')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, axis='y', linestyle=':')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:red'
        ax2.set_ylabel('Rainfall Rate (mm/hr)', color=color)
        
        save_times_min_plot = torch.linspace(0, total_sim_min, steps=len(rates_mmhr_save))
        ax2.plot(save_times_min_plot.numpy(), rates_mmhr_save.numpy(), color=color, linestyle='--', marker='o', markersize=4, label=f'Save Profile (mm/hr) interval={save_int_min}min')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.suptitle(f'Generated Triangular Rainfall (Event: {event_dur_min}min, Peak: {peak_i_mmhr}mm/hr @ {peak_t_min}min)', fontsize=14)
        fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
        plt.show()
    else:
        print("No save profile generated as save_interval_min was None.")