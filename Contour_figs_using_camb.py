import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import camb
from scipy.interpolate import interp1d
import logging
from typing import Dict, List, Tuple
from matplotlib.patches import Rectangle # Import for legend text

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Parameters ---
# These parameters are constant across all simulations
h, omegabh2, omegach2 = 0.6756, 0.022, 0.12
As, ns = 2.215e-9, 0.962
tau = 0.0544
YHe = 0.246
TCMB = 2.7255
num_nu_massless = 3.044
num_nu_massive = 0
nu_mass_degeneracies = [0] # List required by CAMB
nu_mass_numbers = [0] # List required by CAMB

# Define scale factor array (logarithmic for contour plots)
# MODIFIED: Start at 1e-4 and increased resolution
scale_factor = np.logspace(-4, 0, 300) 

# Define parameter ranges for the contour plots (adjust ranges as needed)
# MODIFIED: New range for alpha and increased resolution
alpha_values = np.linspace(-5, 3, 200) # Range for alpha (Y-axis of plot 1)
phi_i_values = np.linspace(6, 25, 200)      # Range for phi_i (Y-axis of plot 2)
phi_prime_i_values = np.linspace(-150, 600, 300) # Range for phi_prime_i (Y-axis of plot 3)

# Define fixed parameters for each plot
fixed_params_plot1 = {'phi_i': 10.0, 'phi_prime_i': 0.0}
fixed_params_plot2 = {'alpha': 1.0, 'phi_prime_i': 0.0}
fixed_params_plot3 = {'alpha': 1.0, 'phi_i': 10.0}

# --- Helper function to find crossings (copied from HDSPlayground) ---
def find_all_phantom_crossings(a: np.ndarray, w_eff: np.ndarray) -> List[Tuple[float, float]]:
    """
    Detects all instances where w_eff crosses the -1 line (phantom crossings).
    Returns a list of (crossing_a, crossing_z) tuples.
    """
    crossings = []
    # We need to ensure w_eff is defined and not NaN
    valid_indices = np.where(~np.isnan(w_eff))
    a_valid = a[valid_indices]
    w_eff_valid = w_eff[valid_indices]

    for i in range(1, len(w_eff_valid)):
        # Check for crossing -1 in either direction
        if (w_eff_valid[i-1] > -1 and w_eff_valid[i] < -1) or \
           (w_eff_valid[i-1] < -1 and w_eff_valid[i] > -1):
            a1, a2 = a_valid[i-1], a_valid[i]
            w1, w2 = w_eff_valid[i-1], w_eff_valid[i]

            if (w2 - w1) != 0: # Avoid division by zero if w_eff is flat
                crossing_a = a1 + (-1 - w1) * (a2 - a1) / (w2 - w1)
                crossing_z = (1 / crossing_a) - 1
                crossings.append((crossing_a, crossing_z))
    return crossings

# --- CAMB Helper Function ---
def get_camb_weff(scale_factor_array: np.ndarray, camb_params: Dict) -> np.ndarray:
    """
    Runs CAMB with given parameters and calculates w_eff.
    Returns w_eff array or NaNs if calculation fails.
    """
    try:
        cosmo = camb.set_params(**camb_params)
        cosmo.NonLinear = camb.model.NonLinear_none
        results = camb.get_results(cosmo)

        rhos = results.get_background_densities(scale_factor_array, vars=['tot', 'de', 'cdm'])
        # Ensure 'hubble' matches expected units if needed, CAMB h_of_z is in Mpc^-1
        # H_conf = a * H = a * h_of_z (already conformal if h_of_z is H)
        H_conf = scale_factor_array * results.h_of_z(1/scale_factor_array - 1) 
        
        _, w_de = results.get_dark_energy_rho_w(scale_factor_array)

        rho_de = rhos['de'] # CAMB densities are rho*a^4/rho_crit_0
        rho_cdm = rhos['cdm']
        
        # Access DarkEnergy model state for phi and phi_prime
        de_model = results.Params.DarkEnergy # Access the DarkEnergy model instance
        
        # Interpolate phi and phi_prime (phidot in CAMB) onto our scale_factor array
        # Ensure sampled_a covers the requested range
        phi_func = interp1d(de_model.sampled_a, de_model.phi_a, bounds_error=False, fill_value="extrapolate")
        phi_prime_func = interp1d(de_model.sampled_a, de_model.phidot_a, bounds_error=False, fill_value="extrapolate")
        
        phi = phi_func(scale_factor_array)
        phi_prime = phi_prime_func(scale_factor_array)

        # Safety checks for division by zero
        phi_safe = np.where(np.abs(phi) < 1e-30, 1e-30, phi)
        H_conf_safe = np.where(np.abs(H_conf) < 1e-30, 1e-30, H_conf)
        rho_de_safe = np.where(np.abs(rho_de) < 1e-40, 1e-40, rho_de)

        # Calculate w_eff using the formula consistent with the Fortran code structure
        # Note: CAMB densities might need conversion depending on units. 
        # Assuming densities are correctly scaled relative to each other here.
        # Ensure alpha is correctly retrieved from params
        alpha = camb_params.get('alpha', 1.0) # Default alpha to 1 if not specified
        coupling_term = alpha * phi_prime * rho_cdm / (phi_safe * 3 * H_conf_safe * rho_de_safe)
        
        w_eff = w_de + coupling_term
        
        # Clamp extreme values if necessary for plotting
        w_eff = np.clip(w_eff, -2, 2) 
        
        return w_eff

    except Exception as e:
        logging.warning(f"CAMB calculation failed for params {camb_params}: {e}")
        return np.full_like(scale_factor_array, np.nan) # Return NaNs on failure

# --- Data Generation Function ---
def generate_contour_data(param_to_vary: str, 
                          param_values: np.ndarray, 
                          fixed_params: Dict, 
                          scale_factor_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a 2D grid of w_eff values by varying one parameter.
    """
    logging.info(f"Generating data by varying {param_to_vary}...")
    w_eff_grid = np.zeros((len(param_values), len(scale_factor_array)))

    common_params = dict(
        H0=100*h, ombh2=omegabh2, omch2=omegach2, TCMB=TCMB,
        omnuh2=0, num_nu_massless=num_nu_massless, num_nu_massive=num_nu_massive,
        nu_mass_degeneracies=nu_mass_degeneracies, nu_mass_numbers=nu_mass_numbers,
        As=As, ns=ns, tau=tau, YHe=YHe, WantTransfer=True, 
        dark_energy_model='HybridQuintessence' # Specify the model
    )

    for i, p_val in enumerate(param_values):
        current_params = common_params.copy()
        current_params.update(fixed_params)
        current_params[param_to_vary] = p_val
        
        # Ensure phi_i etc. are passed correctly if they are fixed
        if 'phi_i' in fixed_params: current_params['phi_i'] = fixed_params['phi_i']
        if 'phi_prime_i' in fixed_params: current_params['phi_prime_i'] = fixed_params['phi_prime_i']
        if 'alpha' in fixed_params: current_params['alpha'] = fixed_params['alpha']

        w_eff_grid[i, :] = get_camb_weff(scale_factor_array, current_params)
        if i % 10 == 0: # Log progress
             logging.info(f"  Completed {i}/{len(param_values)} runs for {param_to_vary}={p_val:.2f}")


    X, Y = np.meshgrid(np.log10(scale_factor_array), param_values)
    logging.info(f"Finished generating data for {param_to_vary}.")
    return X, Y, w_eff_grid

# --- Main Plotting Function ---
def plot_weff_contours(data_plot1: Tuple, fixed_params1: Dict,
                       data_plot2: Tuple, fixed_params2: Dict,
                       data_plot3: Tuple, fixed_params3: Dict):
    """
    Creates the three-panel contour plot figure.
    """
    X1, Y1, Z1 = data_plot1
    X2, Y2, Z2 = data_plot2
    X3, Y3, Z3 = data_plot3

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.subplots_adjust(bottom=0.25, wspace=0.3) # Adjust space for colorbar and between plots

    # Define common contour levels and normalization
    # Match the levels and colormap from the target figure approximately
    levels = np.linspace(-0.92, -0.01, 15) # Adjust number of levels and range
    
    # MODIFIED: Get cmap object and set 'bad' color to green
    cmap = plt.cm.get_cmap('hot_r')
    cmap.set_bad(color='green') # NaN values will be green

    norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    
    # --- X-tick setup ---
    xticks = np.log10(np.array([1e-4, 1e-3, 1e-2, 1e-1, 1e0]))
    xticklabels = [r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$']
    
    # --- Dummy handles for legends ---
    blue_line = plt.Line2D([0], [0], color='blue', lw=2)
    fixed_param_handle = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)


    # Plot 1: Varying alpha
    cf1 = axes[0].contourf(X1, Y1, Z1, levels=levels, cmap=cmap, norm=norm, extend='both')
    axes[0].contour(X1, Y1, Z1, levels=[-1.0], colors='blue', linestyles='solid', linewidths=2)
    
    axes[0].set_xlabel(r'Scale factor $a$ (log)')
    axes[0].set_ylabel(r'$\alpha$')
    axes[0].set_title(r'$w_{eff}(a)$') # MODIFIED: Title
    axes[0].set_ylim(alpha_values.min(), alpha_values.max()) 
    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels(xticklabels)
    # MODIFIED: Legend with fixed params
    fp1 = fixed_params1
    fixed_param_text1 = fr"Fixed: $\phi_i={fp1['phi_i']:.1f}, \phi'_i={fp1['phi_prime_i']:.1f}$"
    axes[0].legend([blue_line, fixed_param_handle],
                   ["Phantom Crossing ($w_{eff}=-1$)", fixed_param_text1],
                   loc='upper right', fontsize='small', handlelength=1)


    # Plot 2: Varying phi_i
    cf2 = axes[1].contourf(X2, Y2, Z2, levels=levels, cmap=cmap, norm=norm, extend='both')
    axes[1].contour(X2, Y2, Z2, levels=[-1.0], colors='blue', linestyles='solid', linewidths=2)

    axes[1].set_xlabel(r'Scale factor $a$ (log)')
    axes[1].set_ylabel(r'$\phi_i / M_{Pl}$')
    axes[1].set_title(r'$w_{eff}(a)$') # MODIFIED: Title
    axes[1].set_ylim(phi_i_values.min(), phi_i_values.max())
    axes[1].set_xticks(xticks)
    axes[1].set_xticklabels(xticklabels)
    # MODIFIED: Legend with fixed params
    fp2 = fixed_params2
    fixed_param_text2 = fr"Fixed: $\alpha={fp2['alpha']:.1f}, \phi'_i={fp2['phi_prime_i']:.1f}$"
    axes[1].legend([blue_line, fixed_param_handle],
                   ["Phantom Crossing ($w_{eff}=-1$)", fixed_param_text2],
                   loc='upper right', fontsize='small', handlelength=1)


    # Plot 3: Varying phi_prime_i
    cf3 = axes[2].contourf(X3, Y3, Z3, levels=levels, cmap=cmap, norm=norm, extend='both')
    axes[2].contour(X3, Y3, Z3, levels=[-1.0], colors='blue', linestyles='solid', linewidths=2)

    axes[2].set_xlabel(r'Scale factor $a$ (log)')
    axes[2].set_ylabel(r"$\phi'_i$ [Mpc⁻¹]")
    axes[2].set_title(r'$w_{eff}(a)$') # MODIFIED: Title
    axes[2].set_ylim(phi_prime_i_values.min(), phi_prime_i_values.max())
    axes[2].set_xticks(xticks)
    axes[2].set_xticklabels(xticklabels)
    # MODIFIED: Legend with fixed params
    fp3 = fixed_params3
    fixed_param_text3 = fr"Fixed: $\alpha={fp3['alpha']:.1f}, \phi_i={fp3['phi_i']:.1f}$"
    axes[2].legend([blue_line, fixed_param_handle],
                   ["Phantom Crossing ($w_{eff}=-1$)", fixed_param_text3],
                   loc='upper right', fontsize='small', handlelength=1)

    # Add shared color bar
    cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.05]) # Position: [left, bottom, width, height]
    cbar = fig.colorbar(cf1, cax=cbar_ax, orientation='horizontal', extend='both')
    cbar.set_label(r'Effective equation of state $w_{eff}$')

    plt.suptitle('Effective Equation of State Dependence on Parameters', fontsize=16)
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # Generate data for each plot
    data1 = generate_contour_data('alpha', alpha_values, fixed_params_plot1, scale_factor)
    data2 = generate_contour_data('phi_i', phi_i_values, fixed_params_plot2, scale_factor)
    data3 = generate_contour_data('phi_prime_i', phi_prime_i_values, fixed_params_plot3, scale_factor)
    
    # Create the plot
    # MODIFIED: Pass fixed params to plotting function
    plot_weff_contours(data1, fixed_params_plot1,
                       data2, fixed_params_plot2,
                       data3, fixed_params_plot3)

