import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import logging
from typing import List, Dict, Any, Optional

# --- Fixed cosmological parameters from Planck 2018 ---
# --- Using CAMB units ---
c_in_km_s = 299_792.458
h = 0.6756
H0 = 100*h/c_in_km_s
rho_cr = 3*H0**2
omega_c = 0.12/h**2
omega_b = 0.022/h**2
omega_phot = 2.47e-5/h**2
omega_massless_nu = 3*(7/8)*(4/11)**(4/3)*omega_phot
omega_r = omega_phot + omega_massless_nu
omega_de = 1 - omega_c - omega_b - omega_r
omega_de_target = omega_de
omega_c_target = omega_c

# -- DESI central values ---
w0_desi = -0.727
wa_desi = -1.05
global_alpha = 1

from HDSSolver import global_alpha, find_all_phantom_crossings, find_w_phi_critical_points, run_full_simulation

# =============================================================================
# Plotting Functions
# =============================================================================

def plot_eos_eff(results: Dict[str, Any], 
                 show_phantom_crossings=True, 
                 show_eos=True, 
                 show_critical_points=True,
                 xlim: Optional[List[float]] = None,
                 ylim: Optional[List[float]] = None,
                 use_redshift=False,
                 show_initial_parameters=True,
                 show_phantom_redshifts=True,
                 show_phantom_phase=False,
                 plot_potential_together=False,
                 plot_effective_potential=False,
                 plot_field_evolution=False,
                 show_ratio_phi_prime_over_phi=False
                 ) -> None:
    """
    Plots the effective equation of state and, optionally, other diagnostic plots.
    """
    logging.info("--- Plotting Results ---")
    
    # --- Setup Figure: Determine number of panels ---
    num_panels = 1
    if plot_potential_together:
        num_panels += 1
    if show_ratio_phi_prime_over_phi:
        num_panels += 1

    if num_panels == 1:
        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax2, ax3 = None, None
    elif num_panels == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        ax3 = None
    else: # num_panels == 3
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))

    # --- Setup X-axis for time-evolution plots ---
    if use_redshift:
        x_data = 1 / results['a'] - 1
        xlabel = "Redshift (z)"
        ax1.invert_xaxis()
        if ax2: ax2.invert_xaxis()
        if ax3: ax3.invert_xaxis()
    else:
        x_data = results['a']
        xlabel = "Scale Factor (a)"

    # --- PANEL 1: Equation of State Plot ---
    if show_eos:
        ax1.plot(x_data, results['w_phi'], label=r'$w_{\phi}$ (DE only)', linestyle=':')
    ax1.plot(x_data, results['w_eff'], label=r'$w_{eff}$ (DE + Interaction)', color='black')
    ax1.axhline(-1, color='red', linestyle='--', linewidth=1, label='Phantom Divide ($w=-1$)')

    if show_phantom_phase:
        ax1.fill_between(x_data, results['w_eff'], -1, where=results['w_eff'] < -1, 
                         color='red', alpha=0.2, label='Phantom Phase')

    if show_phantom_crossings:
        crossings = find_all_phantom_crossings(results['a'], results['w_eff'])
        for i, (crossing_a, crossing_z) in enumerate(crossings):
            x_crossing = crossing_z if use_redshift else crossing_a
            ax1.axvline(x_crossing, color='purple', linestyle='-.', alpha=0.7, 
                        label='Phantom Crossing' if i == 0 else "")
            if show_phantom_redshifts:
                ax1.text(x_crossing * 1.1, ax1.get_ylim()[0] * 0.1, f'z={crossing_z:.2f}', 
                         color='purple', rotation=90)

    if show_critical_points:
        crit_indices = find_w_phi_critical_points(results['w_phi'])
        if len(crit_indices) > 0:
            x_crit = x_data[crit_indices]
            y_crit = results['w_phi'][crit_indices]
            ax1.scatter(x_crit, y_crit, marker='o', s=80, facecolors='none', edgecolors='green',
                        zorder=5, label=r'Critical Points of $w_{\phi}$')

    if xlim: ax1.set_xlim(xlim)
    if ylim: ax1.set_ylim(ylim)
    else: ax1.set_ylim(-1.5, 1.2)
    
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Equation of State (w)")
    ax1.set_title(f"Equation of State Evolution")
    if not use_redshift: ax1.set_xscale('log')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Legend Handling ---
    handles, labels = ax1.get_legend_handles_labels()
    if show_initial_parameters:
        ic = results['initial_conditions']
        ic_text = (
            f"Initial Conditions:\n"
            fr"$a_i = {ic['a_ini']:.1e}$"
            f"\n"
            fr"$\phi_i = {ic['phi_i']}$"
            f"\n"
            fr"$\phi'_i = {ic['phi_prime_i']} [Mpc^{-1}]$"
            f"\n"
            fr"$\alpha={global_alpha}$"
        )
        extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        handles.append(extra)
        labels.append(ic_text)
    
    if num_panels > 1:
        ax1.legend(handles, labels, loc='best')
    else:
        ax1.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
        fig.subplots_adjust(right=0.7)

    # --- PANEL 2: Potential & Field Evolution Plot (Optional) ---
    if plot_potential_together and ax2:
        if potential_flag == 1:
            V_normalized = 1.0 + 0*x_data
        else:
            V_normalized = results['V_of_phi']/results['V_of_phi'][0]
        p1, = ax2.plot(x_data, V_normalized, color='darkorange', linestyle='--', label=r'$V/V_i$')
        ax2.set_ylabel(r'Normalized Potential $V/V_i$', color='darkorange')
        ax2.tick_params(axis='y', labelcolor='darkorange')
        ax2.set_xscale('log')

        lines = [p1]
        if plot_field_evolution:
            ax2b = ax2.twinx()
            phi_normalized = results['phi']/results['phi'][0] 
            p2, = ax2b.plot(x_data, phi_normalized, color='blue', linestyle=':', label=r'$\phi/\phi_i$')
            ax2b.set_ylabel(r'Normalized Field $\phi/\phi_i$', color='blue')
            ax2b.tick_params(axis='y', labelcolor='blue')
            lines.append(p2)

        if show_phantom_phase:
            is_phantom = results['w_eff'] < -1
            phantom_boundaries = np.diff(np.concatenate(([False], is_phantom, [False])).astype(int))
            start_indices = np.where(phantom_boundaries == 1)[0]
            end_indices = np.where(phantom_boundaries == -1)[0]
            
            has_phantom_label = False
            for start_idx, end_idx in zip(start_indices, end_indices):
                label = 'Phantom Phase' if not has_phantom_label else ''
                if start_idx < len(x_data) and end_idx <= len(x_data):
                    ax2.axvspan(x_data[start_idx], x_data[end_idx - 1], color='red', alpha=0.2, label=label)
                    has_phantom_label = True
        
        ax2.set_xlabel(xlabel)
        ax2.set_title('Potential & Field Evolution')
        if not use_redshift: ax2.set_xscale('log')
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax2.legend(lines, [l.get_label() for l in lines], loc='best')

    # --- PANEL 3: Ratio Plot (Optional) ---
    if show_ratio_phi_prime_over_phi and ax3:
        ratio_data = results['phi_prime_over_H_phi']
        ax3.plot(x_data, ratio_data, color='purple', label=r"$\phi' / (\mathcal{H} \phi)$")

        if show_phantom_phase:
            is_phantom = results['w_eff'] < -1
            phantom_boundaries = np.diff(np.concatenate(([False], is_phantom, [False])).astype(int))
            start_indices = np.where(phantom_boundaries == 1)[0]
            end_indices = np.where(phantom_boundaries == -1)[0]
            
            has_phantom_label = False
            for start_idx, end_idx in zip(start_indices, end_indices):
                label = 'Phantom Phase' if not has_phantom_label else ''
                if start_idx < len(x_data) and end_idx <= len(x_data):
                    ax3.axvspan(x_data[start_idx], x_data[end_idx - 1], color='red', alpha=0.2, label=label)
                    has_phantom_label = True

        ax3.set_xlabel(xlabel)
        ax3.set_ylabel(r"Ratio $\phi' / (\mathcal{H} \phi)$")
        ax3.set_title('Field Velocity Ratio')
        ax3.set_xscale('log')
        # Move ticks and label to the right
        ax3.yaxis.tick_right()
        ax3.yaxis.set_label_position("right")
        ax3.legend()
        ax3.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Final Figure Setup ---
    fig.suptitle(f"Cosmological Evolution for {results['potential_object']}", fontsize=16)

def plot_sweep_results(results_list: List[Dict[str, Any]], 
                       sweep_parameter: str, 
                       use_redshift: bool = False, 
                       x_lim=None, y_lim=None,
                       show_phantom_phase=True,
                       show_initial_parameters=True,
                       show_critical_points=False) -> None:
    """
    Plots the results from a parameter sweep on a single graph, with features
    similar to plot_eos_eff.
    """
    logging.info(f"--- Plotting Sweep Results for parameter '{sweep_parameter}' ---")
    fig, ax = plt.subplots(figsize=(12, 7))

    # --- Setup X-axis: Redshift or Scale Factor ---
    if use_redshift:
        xlabel = "Redshift (z)"
    else:
        xlabel = "Scale Factor (a)"

    # --- Get colormap for sweep ---
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_list)))
    
    all_handles = []
    all_labels = []
    
    phantom_label_added = False # Flag to ensure legend entry is added only once

    # --- Plot data from each simulation in the sweep ---
    for i, results in enumerate(results_list):
        color = colors[i]
        if use_redshift:
            x_data = 1 / results['a'] - 1
        else:
            x_data = results['a']
        
        # --- Create Label for this run ---
        sweep_val = results['initial_conditions'][sweep_parameter]
        parts = sweep_parameter.split('_')
        if len(parts) == 2 and parts[0] == 'phi' and parts[1] == 'i':
            latex_name = r"\phi_i"
        elif len(parts) == 3 and parts[0] == 'phi' and parts[1] == 'prime' and parts[2] == 'i':
            latex_name = r"\phi'_i]"
        elif len(parts) == 2:
             latex_name = fr"\{parts[0]}_{{{parts[1]}}}"
        else:
            latex_name = sweep_parameter
        
        label = fr'$w_{{eff}}$ for ${latex_name} = {sweep_val:.2f}$'
        line, = ax.plot(x_data, results['w_eff'], label=label, color=color)
        
        all_handles.append(line)
        all_labels.append(label)

        # --- Plot Critical Points (if enabled) ---
        if show_critical_points:
            crit_indices = find_w_phi_critical_points(results['w_eff'])
            if len(crit_indices) > 0:
                ax.scatter(x_data[crit_indices], results['w_eff'][crit_indices], 
                           marker='o', s=80, facecolors='none', edgecolors=color, zorder=5)

        # --- Plot Phantom Phase (if enabled) ---
        if show_phantom_phase:
            is_phantom = results['w_eff'] < -1
            if np.any(is_phantom):
                # Add label only once for the first phantom event
                label = 'Phantom Phase (color-matched)' if not phantom_label_added else ""
                ax.fill_between(x_data, results['w_eff'], -1, where=is_phantom, 
                                 color=color, alpha=0.2, label=label)
                phantom_label_added = True # Ensure label is only added once

    # --- Final plot setup ---
    line_divide = ax.axhline(-1, color='red', linestyle='--', linewidth=1, label='Phantom Divide ($w=-1$)')
    all_handles.append(line_divide)
    all_labels.append('Phantom Divide ($w=-1$)')
    
    if use_redshift:
        ax.invert_xaxis()
    else:
        ax.set_xscale('log')
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Effective Equation of State ($w_{eff}$)")
    ax.set_title(f"Parameter Sweep for '{sweep_parameter}'\n{results_list[0]['potential_object'].name} Potential")
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    
    # --- Legend Handling ---
    if show_initial_parameters:
        ic = results_list[0]['initial_conditions']
        # Build text block, skipping the swept parameter
        ic_lines = []
        if 'a_ini' != sweep_parameter: ic_lines.append(fr"$a_i = {ic['a_ini']:.1e}$")
        if 'phi_i' != sweep_parameter: ic_lines.append(fr"$\phi_i = {ic['phi_i']:.2f}$")
        if 'phi_prime_i' != sweep_parameter: ic_lines.append(fr"$\phi'_i = {ic['phi_prime_i']:.2e}$")
        ic_lines.append(fr"$\alpha={global_alpha}$")
        
        ic_text = "Fixed Conditions:\n" + "\n".join(ic_lines)
        extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        all_handles.append(extra)
        all_labels.append(ic_text)
        
    ax.legend(all_handles, all_labels, loc='lower right', fontsize='small')
    fig.subplots_adjust(right=0.75) # Make room for legend
    
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

def plot_background_species(results: Dict[str, Any], use_redshift=False, xlim=None, ylim=None, compare_with_LCDM=False) -> None:
    """
    Plots the evolution of the density parameters (Omega) for each cosmic species.
    """
    logging.info("--- Plotting Background Species Evolution ---")
    fig, ax = plt.subplots(figsize=(12, 8))

    a = results['a']
    
    # --- Calculate total density from the simulation ---
    # Note: Using the individual densities from results ensures consistency
    rho_total = results['rho_r'] + results['rho_b'] + results['rho_dm'] + results['rho_phi']

    # --- Calculate Omega for each species from the simulation ---
    omega_rad = results['rho_r'] / rho_total
    omega_baryon = results['rho_b'] / rho_total
    omega_dm = results['rho_dm'] / rho_total
    omega_de = results['rho_phi'] / rho_total

    # --- Setup X-axis ---
    if use_redshift:
        x_data = 1 / a - 1
        xlabel = "Redshift (z)"
        ax.invert_xaxis()
    else:
        x_data = a
        xlabel = "Scale Factor (a)"
        ax.set_xscale('log')

    # --- Plot the species from the simulation ---
    ax.plot(x_data, omega_rad, label=r'$\Omega_r$ (Radiation)', color='red')
    ax.plot(x_data, omega_baryon, label=r'$\Omega_b$ (Baryons)', color='blue')
    ax.plot(x_data, omega_dm, label=r'$\Omega_{dm}$ (Dark Matter)', color='green')
    ax.plot(x_data, omega_de, label=r'$\Omega_{de}$ (Dark Energy)', color='purple')

    # --- Compare with LCDM if requested ---
    if compare_with_LCDM:
        # Calculate the Hubble parameter squared for a standard LCDM model
        # The total matter density parameter is omega_b + omega_c (baryons + cold dark matter)
        H2_lcdm_over_H02 = (omega_r * a**-4 + (omega_b + omega_c_target) * a**-3 + omega_de_target)
        
        # Calculate LCDM Omegas by dividing each component's evolution by the total
        omega_r_lcdm = (omega_r * a**-4) / H2_lcdm_over_H02
        omega_m_lcdm = ((omega_b + omega_c_target) * a**-3) / H2_lcdm_over_H02
        omega_de_lcdm = omega_de_target / H2_lcdm_over_H02
        
        # Plot LCDM curves
        ax.plot(x_data, omega_r_lcdm, color='black', linestyle='--', lw=1, label='ΛCDM')
        ax.plot(x_data, omega_m_lcdm, color='black', linestyle='--', lw=1)
        ax.plot(x_data, omega_de_lcdm, color='black', linestyle='--', lw=1)


    # --- Final plot setup ---
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'Density Parameter ($\Omega$)')
    ax.set_title('Evolution of Cosmic Species')
    ax.legend(loc='best')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(0, 1.1)

    # plt.tight_layout()
    plt.show()

def plot_energies(results: Dict[str, Any], 
                  show_phantom_crossings=True, 
                  xlim: Optional[List[float]] = None,
                  ylim: Optional[List[float]] = None,
                  use_redshift=False,
                  show_initial_parameters=True,
                  show_phantom_redshifts=True,
                  show_phantom_phase=False,
                  y_scale: str = 'linear'):
    """Plots the ratio of kinetic to potential energy (K/V) and other diagnostic ratios."""
    
    logging.info("--- Plotting Energy Ratios ---")
    fig, ax1 = plt.subplots(figsize=(12, 7))
    fig.subplots_adjust(right=0.75) # Make space for multiple y-axes

    # --- Setup X-axis ---
    if use_redshift:
        x_data = 1 / results['a'] - 1
        xlabel = "Redshift (z)"
        ax1.invert_xaxis()
    else:
        x_data = results['a']
        xlabel = "Scale Factor (a)"
        ax1.set_xscale('log')
    
    # --- Create twin axes ---
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax4 = ax1.twinx()
    
    # Offset the right spine of the twin axes
    ax3.spines['right'].set_position(('outward', 60))
    ax4.spines['right'].set_position(('outward', 120))

    # --- Calculate Quantities ---
    K = results['phi_prime']**2 / (2 * results['a']**2)
    V = results['V_of_phi']
    V_safe = np.where(np.abs(V) < 1e-40, 1e-40, V)
    kv_ratio = K / V_safe
    eos_gap = results['w_phi'] + 1
    density_ratio = results['rho_dm'] / results['rho_phi']
    V_normalized = results['V_of_phi'] / results['V_of_phi'][0] if potential_flag != 1 else np.ones_like(x_data)

    # --- Plotting on different axes ---
    p1, = ax1.plot(x_data, kv_ratio, "C0--", label=r'$K/V$')
    p2, = ax2.plot(x_data, eos_gap, "C1-.", label=r'$w_{\phi}+1$')
    p3, = ax3.plot(x_data, density_ratio, "C2:", label=r'$\rho_{\chi}/\rho_{\phi}$')
    p4, = ax4.plot(x_data, V_normalized, "C3-", label=r'$V/V_i$', alpha=0.5)
    
    # --- Axis labels and scales ---
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(r"Energy Ratio $K/V$", color='C0')
    ax2.set_ylabel(r"EoS Gap $w_{\phi}+1$", color='C1')
    ax3.set_ylabel(r"Density Ratio $\rho_{\chi}/\rho_{\phi}$", color='C2')
    ax4.set_ylabel(r"Normalized Potential $V/V_i$", color='C3')
    
    ax1.tick_params(axis='y', colors='C0')
    ax2.tick_params(axis='y', colors='C1')
    ax3.tick_params(axis='y', colors='C2')
    ax4.tick_params(axis='y', colors='C3')
    
    ax1.set_yscale(y_scale)
    ax2.set_yscale(y_scale)
    ax3.set_yscale(y_scale)
    ax4.set_yscale(y_scale)
    
    # --- Phantom Phase and Crossings (on main axis) ---
    if show_phantom_phase:
        ax1.fill_between(x_data, 0, 1, where=results['w_eff'] < -1, 
                         facecolor='red', alpha=0.1, transform=ax1.get_xaxis_transform(), label='Phantom Phase')

    if show_phantom_crossings:
        crossings = find_all_phantom_crossings(results['a'], results['w_eff'])
        for i, (crossing_a, crossing_z) in enumerate(crossings):
            x_crossing = crossing_z if use_redshift else crossing_a
            ax1.axvline(x_crossing, color='purple', linestyle='-.', alpha=0.7, label='Phantom Crossing' if i == 0 else "")

    # --- Legend Handling ---
    ax1.legend(handles=[p1, p2, p3, p4], loc='best')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

def compare_potentials(potential_flags: List[int], 
                       params_to_shoot_list: List[Dict[str, Any]], 
                       initial_conditions: dict, 
                       default_guess: dict, 
                       plot_phi=True, 
                       use_redshift=False,
                       show_phantom_phase=True,
                       plot_delta=True) -> None: # New parameter
    """
    Compares multiple scalar potentials by running simulations and plotting results.
    """
    logging.info(f"--- Comparing Potentials: Flags {potential_flags} ---")

    all_results = []
    for i, flag in enumerate(potential_flags):
        logging.info(f"\n--- Running Simulation for Potential Flag {flag} ---")
        try:
            results = run_full_simulation(
                potential_flag=flag,
                params_vector_to_shoot=params_to_shoot_list[i],
                a_ini=initial_conditions['a_ini'],
                a_end=initial_conditions['a_end'],
                n_steps=initial_conditions['n_steps'],
                phi_i=initial_conditions['phi_i'],
                phi_prime_i=initial_conditions['phi_prime_i'],
                default_guesses=default_guess
            )
            all_results.append(results)
        except (RuntimeError, NotImplementedError, ValueError) as e:
            logging.error(f"Simulation failed for flag {flag}: {e}")
            # Continue to the next simulation even if one fails
    
    if not all_results:
        logging.warning("No simulations completed successfully for comparison.")
        return

    # --- Create Figure with Shared X-axis ---
    # Determine number of rows based on flags
    num_cols = 1
    if plot_delta: num_cols += 1
    num_rows = 1
    if plot_phi: num_rows += 1
    
    fig, axes = plt.subplots(num_rows, num_cols, sharex=True, figsize=(8 * num_cols, 5 * num_rows), squeeze=False)
    
    # Assign axes based on layout
    ax_pot = axes[0, 0]
    ax_phi = axes[1, 0] if plot_phi else None
    ax_delta = axes[0, 1] if plot_delta else None
    if num_rows > 1 and num_cols > 1 and axes.shape == (2,2): # If 2x2 grid
        ax_empty = axes[1, 1] 
        ax_empty.axis('off') # Turn off the unused bottom-right plot

    fig.subplots_adjust(hspace=0.1, wspace=0.3) # Adjust spacing

    # Use a colormap for distinct colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_results)))

    # --- Setup X-axis ---
    if use_redshift:
        xlabel = "Redshift (z)"
        ax_pot.invert_xaxis() # Invert shared axis applies to all
        ax_pot.set_xscale('log') # Log scale applies to all
    else:
        xlabel = "Scale Factor (a)"
        ax_pot.set_xscale('log') # Log scale applies to all

    # --- Plotting Loop ---
    pot_handles, pot_labels = [], []
    
    for i, results in enumerate(all_results):
        color = colors[i]
        pot_obj = results['potential_object']
        pot_label = f"Flag {potential_flags[i]}: {pot_obj}" # Legend entry for potential plot

        # --- Plot Potential ---
        a_data = results['a']
        x_data = 1/a_data - 1 if use_redshift else a_data
        
        if potential_flags[i] == 1:
            V_init = results['V_of_phi']
            V_normalized = [1 for c in results['a']]
        else:
            V_init = results['V_of_phi'][0]
            V_normalized = results['V_of_phi'] / V_init if abs(V_init) > 1e-40 else results['V_of_phi']
        line_pot, = ax_pot.plot(x_data, V_normalized, color=color, label=pot_label)
        pot_handles.append(line_pot)

        # --- Plot Field Evolution ---
        if plot_phi and ax_phi:
            phi_init = results['phi'][0]
            if abs(phi_init) > 1e-40:
                 phi_normalized = results['phi'] / phi_init
            else:
                 phi_normalized = results['phi']
                 logging.warning(f"Flag {potential_flags[i]}: Initial field phi_i = 0, plotting raw phi.")
                 
            line_phi, = ax_phi.plot(x_data, phi_normalized, color=color, linestyle='--') # Use same color

        # --- Plot Delta K/V ---
        if plot_delta and ax_delta:
            K = results['phi_prime']**2 / (2 * a_data**2)
            V = results['V_of_phi']
            V_safe = np.where(np.abs(V) < 1e-40, 1e-40, V)
            abs_delta_kv = np.abs(K / V_safe)
            delta_kv = K / V_safe
            line_delta, = ax_delta.plot(x_data, delta_kv, color=color, linestyle=':') # Use same color

        # --- Phantom Phase Shading (Color Matched) ---
        if show_phantom_phase:
            is_phantom = results['w_eff'] < -1
            phantom_boundaries = np.diff(np.concatenate(([False], is_phantom, [False])).astype(int))
            start_indices = np.where(phantom_boundaries == 1)[0]
            end_indices = np.where(phantom_boundaries == -1)[0]
            
            for start_idx, end_idx in zip(start_indices, end_indices):
                if start_idx < len(x_data) and end_idx <= len(x_data):
                    # Shade on potential plot
                    ax_pot.axvspan(x_data[start_idx], x_data[end_idx - 1], color=color, alpha=0.15)
                    # Shade on field plot if it exists
                    if plot_phi and ax_phi:
                        ax_phi.axvspan(x_data[start_idx], x_data[end_idx - 1], color=color, alpha=0.15)
                    # Shade on delta plot if it exists
                    if plot_delta and ax_delta:
                         ax_delta.axvspan(x_data[start_idx], x_data[end_idx - 1], color=color, alpha=0.15)

            
    # --- Configure Axes ---
    ax_pot.set_ylabel(r'Normalized Potential $V/V_i$')
    ax_pot.set_title('Comparison of Scalar Potentials')
    ax_pot.grid(True, which='both', linestyle='--', linewidth=0.5)
    #ax_pot.set_yscale('log') 
    
    # --- Legend for Potential Plot (Includes Potential Params) ---
    pot_labels = [f"Flag {potential_flags[i]}: {res['potential_object']}" for i, res in enumerate(all_results)]
    ax_pot.legend(pot_handles, pot_labels, loc='best', fontsize='small')


    # --- Configure Phi Plot ---
    if plot_phi and ax_phi:
        ax_phi.set_ylabel(r'Normalized Field $\phi/\phi_i$')
        ax_phi.grid(True, which='both', linestyle='--', linewidth=0.5)
        # Add initial conditions legend here
        ic = initial_conditions
        ic_text = (
            f"Initial Conditions (Shared):\n"
            fr"  $a_i = {ic['a_ini']:.1e}$"
            fr", $\phi_i = {ic['phi_i']:.2f}$"
            fr", $\phi'_i = {ic['phi_prime_i']:.2e}$"
            fr", $\alpha={global_alpha}$"
            f"\n"
            f"The colored regions indicate Phantom Phases."
        )
        extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        phi_handles = [extra] # Start with text block
        phi_labels = [ic_text]
        ax_phi.legend(phi_handles, phi_labels, loc='best', fontsize='small', handlelength=0, handletextpad=0) # Hide marker for text


    # --- Configure Delta Plot ---
    if plot_delta and ax_delta:
        ax_delta.set_ylabel(r'Ratio $\delta = K/V$')
        ax_delta.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax_delta.set_yscale('log')
        # Legend for delta plot (optional, could rely on color matching)
        # ax_delta.legend(loc='best', fontsize='small')

    # --- Set xlabel only on the bottom-most plot(s) ---
    if plot_phi: 
        ax_phi.set_xlabel(xlabel)
        if plot_delta and num_rows > 1 and num_cols > 1: # 2x2 case
            # ax_empty is axes[1,1] - no label needed
             pass
    elif plot_delta: # Only Pot and Delta shown (1x2)
        ax_pot.set_xlabel(xlabel)
        ax_delta.set_xlabel(xlabel)
    else: # Only Potential shown (1x1)
        ax_pot.set_xlabel(xlabel) 

    plt.tight_layout()
    plt.show()

def analysis_phantom_phase(
    phi_i_values: np.ndarray,
    phi_prime_i_values: np.ndarray,
    min_w_eff_grid: np.ndarray,
    max_w_eff_grid: np.ndarray,
    z_cross_grid: np.ndarray, # New argument
    fixed_phi_i_for_plot: float,
    fixed_phi_prime_i_for_plot: float,
    potential_name: str
) -> None:
    """
    Produces a 3-panel figure for phantom phase analysis:
    - Left: Min/Max w_eff vs. phi_i (for fixed phi_prime_i)
    - Middle: Min/Max w_eff vs. phi_prime_i (for fixed phi_i)
    - Right: Contour plot of Min w_eff with z_cross contours.
    """
    logging.info("--- Plotting Phantom Phase Analysis ---")

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)
    fig.suptitle(f"Phantom Phase Analysis for {potential_name} Potential (alpha={global_alpha})", fontsize=16)

    # --- Panel 1: Min/Max w_eff vs. phi_i (fixed phi_prime_i) ---
    ax1 = axes[0]
    # Find the index corresponding to the fixed phi_prime_i value
    fixed_phi_prime_i_idx = np.abs(phi_prime_i_values - fixed_phi_prime_i_for_plot).argmin()
    
    # Extract data for this fixed phi_prime_i
    min_w_eff_slice_phi_i = min_w_eff_grid[:, fixed_phi_prime_i_idx]
    max_w_eff_slice_phi_i = max_w_eff_grid[:, fixed_phi_prime_i_idx]

    ax1.plot(phi_i_values, min_w_eff_slice_phi_i, '-o', color='cyan', label=r'Min $w_{eff}$ ($\phi$ sweep)')
    ax1.plot(phi_i_values, max_w_eff_slice_phi_i, '-o', color='magenta', label=r'Max $w_{eff}$ ($\phi$ sweep)')
    ax1.axhline(-1, color='red', linestyle='--', linewidth=1, label='Phantom Divide ($w=-1$)')
    ax1.set_xlabel(r'$\phi_i$ [CAMB units]')
    ax1.set_ylabel(r'$w_{eff}$')
    ax1.set_title(fr"Fixed $\phi'_i = {phi_prime_i_values[fixed_phi_prime_i_idx]:.1e}$")
    ax1.grid(True)
    ax1.legend(loc='best', fontsize='small')

    # --- Panel 2: Min/Max w_eff vs. phi_prime_i (fixed phi_i) ---
    ax2 = axes[1]
    # Find the index corresponding to the fixed phi_i value
    fixed_phi_i_idx = np.abs(phi_i_values - fixed_phi_i_for_plot).argmin()

    # Extract data for this fixed phi_i
    min_w_eff_slice_phi_prime_i = min_w_eff_grid[fixed_phi_i_idx, :]
    max_w_eff_slice_phi_prime_i = max_w_eff_grid[fixed_phi_i_idx, :]

    ax2.plot(phi_prime_i_values, min_w_eff_slice_phi_prime_i, '-s', color='cyan', label=r"Min $w_{eff}$ ($\phi'$ sweep)")
    ax2.plot(phi_prime_i_values, max_w_eff_slice_phi_prime_i, '-s', color='magenta', label=r"Max $w_{eff}$ ($\phi'$ sweep)")
    ax2.axhline(-1, color='red', linestyle='--', linewidth=1, label='Phantom Divide ($w=-1$)')
    ax2.set_xlabel(r"Initial Scalar Field Derivative $\phi'_i$ [CAMB units]")
    ax2.set_ylabel(r'$w_{eff}$')
    ax2.set_xscale('log') # Log scale for phi_prime_i as in the example image
    ax2.set_title(fr'Fixed $\phi_i = {phi_i_values[fixed_phi_i_idx]:.1f}$')
    ax2.grid(True)
    ax2.legend(loc='best', fontsize='small')

    # --- Panel 3: Contour Plot of Min w_eff ---
    ax3 = axes[2]
    X, Y = np.meshgrid(phi_i_values, phi_prime_i_values) # X=phi_i, Y=phi_prime_i
    
    # We need to transpose grids to match X and Y if phi_i is rows and phi_prime_i is columns
    Z_min_w = min_w_eff_grid.T 
    Z_z_cross = z_cross_grid.T

    # Choose levels to highlight phantom region (w < -1)
    levels = np.linspace(np.nanmin(Z_min_w), -1.0, 10) # Levels in the phantom region
    levels = np.append(levels, np.linspace(-1.0, np.nanmax(Z_min_w), 10)) # Levels above phantom
    levels = sorted(list(set(np.round(levels, 2)))) # Clean up levels
    
    contour = ax3.contourf(X, Y, Z_min_w, levels=levels, cmap='viridis_r', extend='both')
    fig.colorbar(contour, ax=ax3, label=r'Minimum $w_{eff}$', shrink=0.7)

    # --- Add Contour Lines for Redshift Crossing ---
    # Filter out NaNs for contouring
    Z_z_cross_f = np.nan_to_num(Z_z_cross, nan=np.nanmax(Z_z_cross)) # Replace NaN for contouring
    z_levels = np.arange(np.nanmin(Z_z_cross), np.nanmax(Z_z_cross), 0.5) # Auto levels
    if len(z_levels) > 0:
        CS = ax3.contour(X, Y, Z_z_cross_f, levels=z_levels, colors='white', linestyles='dashed')
        ax3.clabel(CS, inline=True, fontsize=10, fmt='z=%.1f')
    
    ax3.set_xlabel(r'Initial Scalar Field Amplitude $\phi_i$')
    ax3.set_ylabel(r"Initial Scalar Field Derivative $\phi'_i$")
    ax3.set_yscale('log') # Log scale for y-axis (phi_prime_i)
    ax3.set_title(r'Contour Plot of Min $w_{eff}$ (lines = $z_{cross}$)')
    ax3.grid(True, linestyle=':', alpha=0.5)

    plt.show()

def plot_fields_and_derivatives(results: Dict[str, Any], 
                                use_redshift: bool = False,
                                show_phantom_phase: bool = True,
                                show_initial_parameters: bool = True,
                                is_log_x: bool = True,
                                is_log_y: list = [True, True, True]) -> None:
    """
    Plots the scalar field (phi), its first derivative (phi'), 
    and its second derivative (phi'') in a 3-panel plot.
    """
    logging.info("--- Plotting Field and Derivative Evolution ---")
    
    # --- Create Figure ---
    # 3 panels, stacked vertically, sharing the x-axis
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    fig.subplots_adjust(hspace=0.08) # Reduce vertical space

    # --- Setup X-axis ---
    if use_redshift:
        x_data = 1 / results['a'] - 1
        xlabel = "Redshift (z)"
        ax1.invert_xaxis() # This will apply to all shared axes
        if is_log_x:
            ax1.set_xscale('log')
    else:
        x_data = results['a']
        xlabel = "Scale Factor (a)"
        if is_log_x:
            ax1.set_xscale('log') # This will apply to all shared axes

    # --- Get Data Arrays from Results ---
    a = results['a']
    phi = results['phi']
    phi_prime = results['phi_prime']
    H_curly = results['H_curly']
    rho_dm = results['rho_dm']
    potential_object = results['potential_object']
    
    # --- Calculate phi'' (Second Conformal Time Derivative) ---
    # We use the Klein-Gordon equation:
    # phi'' = -2*H_curly*phi' - a^2*(dV/dphi) - a^2*(rho_dm*alpha/phi)
    
    # Get dV/dphi, ensuring it's an array
    dV_dphi_val = potential_object.derivative(phi)
    if np.isscalar(dV_dphi_val):
        dV_dphi_arr = np.full_like(phi, dV_dphi_val)
    else:
        dV_dphi_arr = dV_dphi_val

    # Safety checks for division by zero
    epsilon = 1e-40
    phi_safe = np.where(np.abs(phi) < epsilon, epsilon, phi)
    H_curly_safe = np.where(np.abs(H_curly) < epsilon, epsilon, H_curly)

    # Calculate the three terms of the equation
    term1 = -2 * H_curly_safe * phi_prime       # Hubble friction
    term2 = -a**2 * dV_dphi_arr                  # Potential force
    term3 = -a**2 * rho_dm * global_alpha / phi_safe # Coupling force
    
    phi_double_prime = term1 + term2 + term3

    phi_double_dot = (phi_double_prime / a**2) - (H_curly_safe * phi_prime / a**2)
    phi_dot = phi_prime / a

    # --- Panel 1: Field Evolution (phi) ---
    ax1.plot(x_data, phi, color='blue', label=r'$\phi$')
    ax1.set_ylabel(r'$\phi$')
    ax1.set_title('Scalar Field Evolution')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    if is_log_y[0]:
        ax1.set_yscale('symlog', linthresh=1e-10)

    # --- Panel 2: First Derivative (phi') ---
    ax2.plot(x_data, phi_dot, color='green', label=r"$\phi'$")
    ax2.set_ylabel(r"$\dot{\phi} ~ [Mpc^{-1}]$")
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    if is_log_y[1]:
        ax2.set_yscale('symlog', linthresh=1e-10)

    # --- Panel 3: Second Derivative (phi'') ---
    ax3.plot(x_data, phi_double_dot, color='purple', label=r"$\phi''$")
    ax3.set_ylabel(r"$\ddot{\phi} ~ [Mpc^{-2}]$")
    ax3.set_xlabel(xlabel) # Only on the bottom plot
    ax3.grid(True, which='both', linestyle='--', linewidth=0.5)
    if is_log_y[2]:
        ax3.set_yscale('symlog', linthresh=1e-9)
    
    # --- Apply Phantom Phase Shading to all panels ---
    if show_phantom_phase:
        is_phantom = results['w_eff'] < -1
        phantom_boundaries = np.diff(np.concatenate(([False], is_phantom, [False])).astype(int))
        start_indices = np.where(phantom_boundaries == 1)[0]
        end_indices = np.where(phantom_boundaries == -1)[0]
        
        has_phantom_label = False
        for start_idx, end_idx in zip(start_indices, end_indices):
            label = 'Phantom Phase' if not has_phantom_label else ''
            if start_idx < len(x_data) and end_idx <= len(x_data):
                # Shade all three axes
                ax1.axvspan(x_data[start_idx], x_data[end_idx - 1], color='red', alpha=0.2, label=label)
                ax2.axvspan(x_data[start_idx], x_data[end_idx - 1], color='red', alpha=0.2)
                ax3.axvspan(x_data[start_idx], x_data[end_idx - 1], color='red', alpha=0.2)
                has_phantom_label = True

    # --- Final Legend Handling (add to top plot) ---
    handles, labels = ax1.get_legend_handles_labels()
    if show_initial_parameters:
        ic = results['initial_conditions']
        ic_text = (
            f"Initial Conditions:\n"
            fr"  $a_i = {ic['a_ini']:.1e}$"
            fr", $\phi_i = {ic['phi_i']:.2f}$"
            fr", $\phi'_i = {ic['phi_prime_i']:.2e}$"
            fr", $\alpha={global_alpha}$"
        )
        extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        handles.append(extra)
        labels.append(ic_text)
        print(results['potential_object'])
    
    ax1.legend(handles, labels, loc='best', fontsize='small', bbox_to_anchor=(1, 0.5))
    ax2.legend(loc='best', fontsize='small')
    ax3.legend(loc='best', fontsize='small')

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    plt.show()

def plot_klein_gordon_terms(results: Dict[str, Any], 
                            use_redshift: bool = False, 
                            show_phantom_phase: bool = True,
                            show_initial_parameters: bool = True) -> None:
    """
    Plots the individual terms of the Klein-Gordon equation for the scalar field.
    """
    logging.info("--- Plotting Klein-Gordon Equation Terms ---")
    
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # --- Setup X-axis ---
    if use_redshift:
        x_data = 1 / results['a'] - 1
        xlabel = "Redshift (z)"
        ax1.invert_xaxis()
        ax1.set_xscale('log')
    else:
        x_data = results['a']
        xlabel = "Scale Factor (a)"
        ax1.set_xscale('log')

    # --- Calculate Terms ---
    a = results['a']
    phi = results['phi']
    phi_prime = results['phi_prime']
    H_curly = results['H_curly']
    rho_dm = results['rho_dm']
    potential_object = results['potential_object']

    dV_dphi_val = potential_object.derivative(phi)
    if np.isscalar(dV_dphi_val):
        dV_dphi_arr = np.full_like(phi, dV_dphi_val)
    else:
        dV_dphi_arr = dV_dphi_val

    epsilon = 1e-40
    phi_safe = np.where(np.abs(phi) < epsilon, epsilon, phi)
    H_curly_safe = np.where(np.abs(H_curly) < epsilon, epsilon, H_curly)

    term1 = -2 * H_curly_safe * phi_prime       # Hubble friction
    term2 = -a**2 * dV_dphi_arr                  # Potential force
    term3 = -a**2 * rho_dm * global_alpha / phi_safe # Coupling force
    
    phi_double_prime = term1 + term2 + term3

    phi_double_dot = (phi_double_prime / a**2) - (H_curly_safe * phi_prime / a**2)
    phi_dot = phi_prime / a
    hubble_fric = 3 * (H_curly_safe / a) * phi_dot
    pot_force = -dV_dphi_arr
    coupling_force = -global_alpha * rho_dm / phi_safe


    # --- Plot Terms ---
    ax1.plot(x_data, hubble_fric, label=r"Hubble Friction ($-3H \dot{phi}$)", color='blue', linestyle='--')
    ax1.plot(x_data, pot_force, label=r'Potential Force ($-V_{,\phi}$)', color='green', linestyle=':')
    ax1.plot(x_data, coupling_force, label=r'Coupling Force (-$\alpha \rho_{dm}/\phi$)', color='red', linestyle='-.')
    ax1.plot(x_data, phi_double_dot, label=r"Field acceleration: $\ddot{\phi}~~[Mpc^-2]$", color='black', linestyle='-')


    # --- Add Phantom Phase Shading ---
    if show_phantom_phase:
        # Use transform=ax1.get_xaxis_transform() to fill vertically
        ax1.fill_between(x_data, 0, 1, where=results['w_eff'] < -1, 
                         facecolor='red', alpha=0.1, transform=ax1.get_xaxis_transform(), 
                         label='Phantom Phase ($w_{eff} < -1$)')

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(r'Klein-Gordon Terms') # Units are complex
    ax1.set_yscale('symlog', linthresh=1e-10) # Use symlog for positive/negative values
    ax1.set_title('Klein-Gordon Equation Terms Evolution')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Legend Handling ---
    handles, labels = ax1.get_legend_handles_labels()
    if show_initial_parameters:
        ic = results['initial_conditions']
        ic_text = (
            f"Initial Conditions:\n"
            fr"  $a_i = {ic['a_ini']:.1e}$"
            fr", $\phi_i = {ic['phi_i']:.2f}$"
            fr", $\phi'_i = {ic['phi_prime_i']:.2e}$"
            fr", $\alpha={global_alpha}$"
            f"\nPotential:\n"
            f"  {results['potential_object']}"
        )
        extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        handles.append(extra)
        labels.append(ic_text)

    ax1.legend(handles, labels, loc='lower right', fontsize='small')
    fig.subplots_adjust(right=0.75) # Make room for legend

if __name__ == '__main__':

    #Flags to display simulations
    PLOT_EQUATION_OF_STATE = True
    PLOT_FIELD_EVOLUTION = False
    PLOT_BACKGROUND_EVOLUTION_SPECIES = False
    PLOT_ENERGIES_RATIO = False
    PLOT_COMPARISON_OF_POTENTIALS = False
    PLOT_PHANTOM_PHASE_ANALYSIS = False
    PLOT_FIELDS_AND_DERIVATIVES = True
    
    # --- Control Parameter Sweep ---
    # Set to a string like 'phi_i' or 'phi_prime_i' to activate sweep mode.
    # Set to None to run a single simulation.
    sweep_parameter = None
    sweep_values = [0, 1e1, 1e2, 500]
   
    # --- Set Initial Simulation Parameters ---
    potential_flag = 6

    initial_conditions = {
        'a_ini': 1e-5,
        'a_end': 1.0,
        'n_steps': 500000,
        'phi_i': 20,
        'phi_prime_i': 0.0
    }

    #Very weird stuff happening at high redshift, there is probably a mistake on the implementation of the second derivative of the field.

    default_guesses = {
        'V0': [rho_cr * omega_de_target * 0.8, rho_cr * omega_de_target * 1.2], # Standard guessses, we know it works
        'A':  [rho_cr * omega_de_target * 0.8, rho_cr * omega_de_target * 1.2], # For exponential potential: A = [1e-7, 1e-8], B = [-0.05, 0.05], these usually work for secant method
        'B':  [-1e5, 1e5],
        'C':  [1e-7, 1e-8]
    }

    a_ini = initial_conditions['a_ini']
    a_end = initial_conditions['a_end']
    n_steps = initial_conditions['n_steps']
    phi_i_base = initial_conditions['phi_i']
    phi_prime_i_base = initial_conditions['phi_prime_i']

    # params_vector_to_shoot = {'A': True, 'B': -0.09} # This one is interesting for Exponential potential
    # params_vector_to_shoot = {'A': True, 'B': 1/phi_i_base**2, 'C':phi_i_base/10} # This one is interesting for Gaussian potential
    # params_vector_to_shoot = {'V0': True} # bissection is not working properly here, need to check
    # params_vector_to_shoot = {'A': True, 'B':  4} # This one is interesting for Power law, \lambda\phi^4 seems to not work well
    params_vector_to_shoot = {'V0': True, 'A': 19.37, 'B': 19.41}

    #c=19.3
    #params_vector_to_shoot = {'A': True, 'B': 0.6, 'C':c}
    
    eps = 1
    #params_vector_to_shoot = {'V0': True, 'A': phi_i_base, 'B': phi_i_base/2, 'C': -1} # tanh
    
    # --- Main Execution Logic: Single Run vs. Parameter Sweep ---
    if PLOT_PHANTOM_PHASE_ANALYSIS:
        logging.info("--- Starting 2D Sweep for Phantom Phase Analysis ---")
        
        # Define sweep ranges for phi_i and phi_prime_i
        phi_i_values = np.linspace(1, 25, 20) # 20 points
        phi_prime_i_values = np.logspace(0, 3, 20) # 20 points (1 to 1000)

        # Initialize grids to store min/max w_eff results
        min_w_eff_grid = np.full((len(phi_i_values), len(phi_prime_i_values)), np.nan)
        max_w_eff_grid = np.full((len(phi_i_values), len(phi_prime_i_values)), np.nan)
        z_cross_grid = np.full_like(min_w_eff_grid, np.nan) # New grid for z_cross

        # Store potential name for plot title
        potential_name_for_plot = "Unknown Potential" 

        for i, phi_i_val in enumerate(phi_i_values):
            for j, phi_prime_i_val in enumerate(phi_prime_i_values):
                logging.info(f"Running simulation for phi_i = {phi_i_val:.2f}, phi_prime_i = {phi_prime_i_val:.2e}")
                
                try:
                    current_initial_conditions = initial_conditions.copy()
                    current_initial_conditions['phi_i'] = phi_i_val
                    current_initial_conditions['phi_prime_i'] = phi_prime_i_val

                    results = run_full_simulation(
                        potential_flag=potential_flag,
                        params_vector_to_shoot=params_vector_to_shoot,
                        a_ini=current_initial_conditions['a_ini'],
                        a_end=current_initial_conditions['a_end'],
                        n_steps=current_initial_conditions['n_steps'],
                        phi_i=current_initial_conditions['phi_i'],
                        phi_prime_i=current_initial_conditions['phi_prime_i'],
                        default_guesses=default_guesses
                    )
                    
                    min_w_eff_grid[i, j] = np.min(results['w_eff'])
                    max_w_eff_grid[i, j] = np.max(results['w_eff'])
                    potential_name_for_plot = results['potential_object'].name
                    
                    # Find and store the first phantom crossing redshift
                    crossings = find_all_phantom_crossings(results['a'], results['w_eff'])
                    if crossings:
                        z_cross_grid[i, j] = crossings[0][1] # [0] is first crossing, [1] is z_val

                except (RuntimeError, NotImplementedError, ValueError) as e:
                    logging.error(f"Simulation failed for phi_i={phi_i_val:.2f}, phi_prime_i={phi_prime_i_val:.2e}: {e}")
                    # Grids are already initialized with NaN, so no action needed on failure
        
        # --- Call the new plotting function ---
        # Select fixed values for the 1D slices, try to pick values close to your example
        fixed_phi_i_for_plot = 10.0 # As in your middle plot title
        fixed_phi_prime_i_for_plot = 100.0 # Pick a value within the logspace range
        
        analysis_phantom_phase(
            phi_i_values,
            phi_prime_i_values,
            min_w_eff_grid,
            max_w_eff_grid,
            z_cross_grid, # Pass the new grid
            fixed_phi_i_for_plot,
            fixed_phi_prime_i_for_plot,
            potential_name_for_plot
        )

    elif sweep_parameter is None:
        # --- Single Simulation Run ---
        if PLOT_COMPARISON_OF_POTENTIALS:

         flags_to_compare = [1, 2, 4, 6, 7]
         params_list_to_compare = [
             {'V0': True},                     # Constant
             {'A': True, 'B': -0.00001},           # Exponential
             {'A': True, 'B': 1/phi_i_base**2, 'C':phi_i_base/10},   # Gaussian
             {'V0': True, 'A': phi_i_base*0.81, 'B':phi_i_base*0.88,},
             {'V0': True, 'A': phi_i_base*0.88, 'B':phi_i_base*0.838, 'C': 8}  # Tanh
         ]

         compare_potentials(
             potential_flags=flags_to_compare,
             params_to_shoot_list=params_list_to_compare,
             initial_conditions=initial_conditions,
             default_guess=default_guesses,
             plot_phi=True, # Show field evolution comparison
             use_redshift=True # Example: Use scale factor
         )

        try:
                results = run_full_simulation(
                    potential_flag=potential_flag,
                    params_vector_to_shoot=params_vector_to_shoot,
                    a_ini=a_ini, a_end=a_end, n_steps=n_steps,
                    phi_i=phi_i_base, phi_prime_i=phi_prime_i_base,
                    default_guesses=default_guesses
                )

                print("\n--- Final Converged Solution & Results ---")
                print(f"Final Potential: {results['potential_object']}")
                print(results['phi'])
                
                if PLOT_EQUATION_OF_STATE:
                    plot_eos_eff(results, 
                                use_redshift=True, 
                                show_phantom_phase=True, 
                                xlim=[0, 6], ylim=[-1.8, 1.2], 
                                plot_potential_together=False, 
                                plot_effective_potential=False,
                                plot_field_evolution=PLOT_FIELD_EVOLUTION,
                                show_ratio_phi_prime_over_phi=False,
                                )
                
                if PLOT_BACKGROUND_EVOLUTION_SPECIES:
                    plot_background_species(results, compare_with_LCDM=True)

                if PLOT_ENERGIES_RATIO:
                    plot_energies(results, 
                                use_redshift=True, 
                                show_phantom_phase=True,
                                )
                if PLOT_FIELDS_AND_DERIVATIVES:
                    plot_klein_gordon_terms(results, use_redshift=True)
                    plot_fields_and_derivatives(results, use_redshift=True, show_phantom_phase=True, is_log_y=[False, True, True])

                plt.show()

        except (RuntimeError, NotImplementedError, ValueError) as e:
                logging.error(f"Single simulation failed: {e}")
            
    else:
            # --- Parameter Sweep Run ---
            logging.info(f"--- Starting Parameter Sweep for '{sweep_parameter}' ---")
            all_results = []
            
            # Prepare a dictionary of the base parameters
            sim_params = {
                'potential_flag': potential_flag,
                'params_vector_to_shoot': params_vector_to_shoot,
                'a_ini': a_ini, 'a_end': a_end, 'n_steps': n_steps,
                'phi_i': phi_i_base, 'phi_prime_i': phi_prime_i_base,
                'default_guesses': default_guesses
            }

            for value in sweep_values:
                logging.info(f"\n--- Running for {sweep_parameter} = {value} ---")
                # Update the sweep parameter for this specific run
                sim_params[sweep_parameter] = value
                try:
                    results = run_full_simulation(**sim_params)
                    all_results.append(results)
                except (RuntimeError, NotImplementedError, ValueError) as e:
                    logging.error(f"Simulation failed for {sweep_parameter} = {value}: {e}")
            
            # After all simulations are done, plot the combined results
            if all_results:
                plot_sweep_results(all_results, sweep_parameter, use_redshift=False, x_lim=[1, 0.0001], y_lim=[-2.0, 2.0])
                plt.show()
