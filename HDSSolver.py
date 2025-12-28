import numpy as np
from scipy.interpolate import interp1d
import logging
from typing import List, Tuple, Dict, Any
from scipy.integrate import odeint, cumulative_trapezoid

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

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

global_alpha=1

from HDSpotentials import ScalarPotential, potential_factory

# =============================================================================
# Main functions for HDS solving
# =============================================================================

def H_curly(y: List[float], a: float, rho_dm: float, potential_object: ScalarPotential) -> float:
    """
    Calculates the conformal Hubble parameter H*a for the given state.
    """
    phi, phi_prime = y
   
    V_phi = potential_object.value(phi)
    
    rho_phi = phi_prime**2 / (2 * (a**2)) + V_phi
    rho = rho_cr * (omega_r * a**(-4) + omega_b * a**(-3)) + rho_dm + rho_phi

    return a * np.sqrt(rho / 3) if rho >= 0 else 0.0

def equations(y: List[float], a: float, rho_dm_i: float, a_i: float, phi_i: float, potential_object: ScalarPotential) -> np.ndarray:
    """
    Defines the Klein-Gordon equation for the DE field and the analytical expression for the CDM energy density.
    The integration variable is the scale factor 'a'.
    """
    
    alpha = global_alpha # I really have to see what is going on with this boy...

    phi, phi_prime = y
    rho_dm = rho_dm_i * ((phi / phi_i)**alpha) * (a_i / a)**3

    H = H_curly(y, a, rho_dm, potential_object)
    
    epsilon_denom = 1e-30
    H_safe = H if np.abs(H) > epsilon_denom else np.sign(H) * epsilon_denom if H != 0 else epsilon_denom
    phi_safe = phi if np.abs(phi) > epsilon_denom else np.sign(phi) * epsilon_denom if phi != 0 else epsilon_denom

    # Call the .derivative() method on the passed object to get dV/dphi
    dV_dphi_val = potential_object.derivative(phi)

    # Assumes y = [phi, phi'], where phi' is the derivative wrt conformal time (eta)
    dphi_da = phi_prime / (a * H_safe)
    dphi_prime_da = -2 * phi_prime / a - (a * dV_dphi_val / H_safe) - (a * rho_dm * alpha / (phi_safe * H_safe))
    
    return np.array([dphi_da, dphi_prime_da])

def equations_loga(y: List[float], loga: float, rho_dm_i: float, a_i: float, phi_i: float, potential_object: ScalarPotential) -> np.ndarray:
    """
    Defines the system of differential equations with respect to log(a).
    It also takes a potential_object and passes it to the core equations.
    """
    a = np.exp(loga)
    # The change of variables: dy/d(loga) = a * dy/da
    derivs_wrt_a = equations(y, a, rho_dm_i, a_i, phi_i, potential_object)
    return a * derivs_wrt_a

def integrate_cosmo(ic: List[float], a_ini: float, a_end: float, n_steps: int, rho_dm_i: float, potential_object: ScalarPotential) -> Tuple[np.ndarray, np.ndarray]:
    """Integrates the cosmological equations using a hybrid log/linear scale."""
    frac, a_threshold = 0.4, 1e-3
    phi_i = ic[0]
    n_steps_log = max(2, int(frac * n_steps))
    a_log = np.logspace(np.log10(a_ini), np.log10(a_threshold), n_steps_log)
    
    args_ode = (rho_dm_i, a_ini, phi_i, potential_object)
    result_log = odeint(equations_loga, ic, np.log(a_log), args=args_ode)

    ic_normal = result_log[-1]
    n_steps_normal = max(2, n_steps - n_steps_log)
    a_normal = np.linspace(a_threshold, a_end, n_steps_normal)
    result_normal = odeint(equations, ic_normal, a_normal, args=args_ode)

    # Remove the duplicate point at the threshold
    full_a = np.concatenate((a_log, a_normal[1:]))
    full_result = np.concatenate((result_log, result_normal[1:]))
    return full_a, full_result

def find_present_day_fractions(result: np.ndarray, a_ini: float, rho_dm_i: float, potential_object: ScalarPotential) -> Tuple[float, float]:
    """Calculates the dark energy and dark matter fractions at a=1."""
    phi_0, phi_prime_0 = result[-1]
    phi_i = result[0, 0]

    V_phi_0 = potential_object.value(phi_0)
    rho_phi_0 = phi_prime_0**2 / 2 + V_phi_0 # a = 1 today
    rho_dm_0 = rho_dm_i * (phi_0 / phi_i) * (a_ini)**3
    
    rho_tot_0 = rho_cr * (omega_r + omega_b) + rho_dm_0 + rho_phi_0
    
    if rho_tot_0 == 0: return 0.0, 0.0
    return rho_phi_0 / rho_tot_0, rho_dm_0 / rho_tot_0

def integrate_cosmo(ic: List[float], a_ini: float, a_end: float, n_steps: int, rho_dm_i: float, potential_object: ScalarPotential) -> Tuple[np.ndarray, np.ndarray]:
    """Integrates the cosmological equations using a hybrid log/linear scale."""
    frac, a_threshold = 0.4, 3e-4 # JVR EDIT: changed a_threshold for compatilibity with CAMB
    phi_i = ic[0]
    n_steps_log = 2_000 # max(2, int(frac * n_steps)) # JVR EDIT: changed n_steps_log for compatibility with CAMB
    a_log = np.logspace(np.log10(a_ini), np.log10(a_threshold), n_steps_log)
    
    args_ode = (rho_dm_i, a_ini, phi_i, potential_object)
    result_log = odeint(equations_loga, ic, np.log(a_log), args=args_ode)

    ic_normal = result_log[-1]
    n_steps_normal = 10_000 # max(2, n_steps - n_steps_log) # JVR EDIT: changed n_steps_normal for compatibility with CAMB
    a_normal = np.linspace(a_threshold, a_end, n_steps_normal)
    result_normal = odeint(equations, ic_normal, a_normal, args=args_ode)

    # Remove the duplicate point at the threshold
    full_a = np.concatenate((a_log, a_normal[1:]))
    full_result = np.concatenate((result_log, result_normal[1:]))
    return full_a, full_result

def find_present_day_fractions(result: np.ndarray, a_ini: float, rho_dm_i: float, potential_object: ScalarPotential) -> Tuple[float, float]:
    """Calculates the dark energy and dark matter fractions at a=1."""
    phi_0, phi_prime_0 = result[-1]
    phi_i = result[0, 0]

    V_phi_0 = potential_object.value(phi_0)
    rho_phi_0 = phi_prime_0**2 / 2 + V_phi_0 # a = 1 today
    rho_dm_0 = rho_dm_i * (phi_0 / phi_i) * (a_ini)**3
    
    rho_tot_0 = rho_cr * (omega_r + omega_b) + rho_dm_0 + rho_phi_0
    
    if rho_tot_0 == 0: return 0.0, 0.0
    return rho_phi_0 / rho_tot_0, rho_dm_0 / rho_tot_0

def _shoot_for_rho_dm_i(ic: List[float], a_ini: float, a_end: float, n_steps: int, potential_object: ScalarPotential) -> float:
    """INNER LOOP: Finds rho_dm_i to match omega_c_target for a fixed potential."""
    phi_i = ic[0]
    rho_dm_i_guess_base = rho_cr * omega_c_target * a_ini**(-3)
    rho_dm_i_1 = rho_dm_i_guess_base * 0.95
    rho_dm_i_2 = rho_dm_i_guess_base * 1.05
    
    # --- Attempt 1: Secant Method ---
    converged = False
    for _ in range(20):
        _, result_1 = integrate_cosmo(ic, a_ini, a_end, n_steps, rho_dm_i_1, potential_object)
        _, omega_c_1 = find_present_day_fractions(result_1, a_ini, rho_dm_i_1, potential_object)
        _, result_2 = integrate_cosmo(ic, a_ini, a_end, n_steps, rho_dm_i_2, potential_object)
        _, omega_c_2 = find_present_day_fractions(result_2, a_ini, rho_dm_i_2, potential_object)

        if abs(omega_c_2 - omega_c_1) < 1e-10: break
        slope = (omega_c_2 - omega_c_1) / (rho_dm_i_2 - rho_dm_i_1)
        if abs(slope) < 1e-10: break

        rho_dm_i_next = rho_dm_i_2 - (omega_c_2 - omega_c_target) / slope
        if abs(omega_c_2 - omega_c_target) < 1e-5:
            converged = True
            break
        rho_dm_i_1, rho_dm_i_2 = rho_dm_i_2, rho_dm_i_next
    
    if converged:
        return rho_dm_i_2

    # --- Attempt 2: Bisection Method (Fallback) --- 
    logging.warning("Inner loop (rho_dm_i) secant method failed. Switching to bisection.")
    rho_dm_i_1 = rho_dm_i_guess_base * 0.5
    rho_dm_i_2 = rho_dm_i_guess_base * 1.5
    
    _, r1 = integrate_cosmo(ic, a_ini, a_end, n_steps, rho_dm_i_1, potential_object)
    _, omega_c_1 = find_present_day_fractions(r1, a_ini, rho_dm_i_1, potential_object)
    _, r2 = integrate_cosmo(ic, a_ini, a_end, n_steps, rho_dm_i_2, potential_object)
    _, omega_c_2 = find_present_day_fractions(r2, a_ini, rho_dm_i_2, potential_object)
    
    if (omega_c_1 - omega_c_target) * (omega_c_2 - omega_c_target) >= 0:
        raise RuntimeError("Inner loop bisection failed: initial guesses do not bracket the solution.")
        
    for _ in range(20):
        rho_dm_i_mid = (rho_dm_i_1 + rho_dm_i_2) / 2
        _, result_mid = integrate_cosmo(ic, a_ini, a_end, n_steps, rho_dm_i_mid, potential_object)
        _, omega_c_mid = find_present_day_fractions(result_mid, a_ini, rho_dm_i_mid, potential_object)
        
        if abs(omega_c_mid - omega_c_target) < 1e-5:
            return rho_dm_i_mid
            
        if (omega_c_mid - omega_c_target) * (omega_c_1 - omega_c_target) < 0:
            rho_dm_i_2 = rho_dm_i_mid
        else:
            rho_dm_i_1 = rho_dm_i_mid
            
    return rho_dm_i_mid

def solve_system(
    ic_phi: float, ic_phi_prime: float, a_ini: float, a_end: float, n_steps: int,
    potential_flag: int, initial_params: Dict[str, Any], param_to_shoot: str,
    guess_1: float, guess_2: float
) -> Tuple[np.ndarray, np.ndarray, ScalarPotential, float]:
    """
    OUTER LOOP: Solves the system using a nested shooting method with a bisection fallback.
    """
    logging.info(f"--- Starting Nested Shooting for Potential Flag {potential_flag} ---")
    logging.info(f"Shooting for parameter '{param_to_shoot}' to match omega_de_target = {omega_de_target:.5f}")
    
    ic = [ic_phi, ic_phi_prime]
    PotentialClass = potential_factory[potential_flag]
    
    p1, p2 = guess_1, guess_2
    params1, params2 = initial_params.copy(), initial_params.copy()
    params1[param_to_shoot], params2[param_to_shoot] = p1, p2
    potential1, potential2 = PotentialClass(**params1), PotentialClass(**params2)

    rho_dm_i1 = _shoot_for_rho_dm_i(ic, a_ini, a_end, n_steps, potential1)
    a, result1 = integrate_cosmo(ic, a_ini, a_end, n_steps, rho_dm_i1, potential1)
    omega_de1, _ = find_present_day_fractions(result1, a_ini, rho_dm_i1, potential1)

    rho_dm_i2 = _shoot_for_rho_dm_i(ic, a_ini, a_end, n_steps, potential2)
    _, result2 = integrate_cosmo(ic, a_ini, a_end, n_steps, rho_dm_i2, potential2)
    omega_de2, _ = find_present_day_fractions(result2, a_ini, rho_dm_i2, potential2)

    logging.info(f"Initial Guess 1: {param_to_shoot}={p1:.4e} -> omega_de={omega_de1:.5f}")
    logging.info(f"Initial Guess 2: {param_to_shoot}={p2:.4e} -> omega_de={omega_de2:.5f}")

    # --- Attempt 1: Secant Method ---
    converged = False
    final_result, final_potential, final_rho_dm_i = result2, potential2, rho_dm_i2
    for i in range(20):
        if abs(p2 - p1) < 1e-10: break
        slope = (omega_de2 - omega_de1) / (p2 - p1)
        if abs(slope) < 1e-10: break
        
        p_next = p2 - (omega_de2 - omega_de_target) / slope
        params_next = initial_params.copy(); params_next[param_to_shoot] = p_next
        potential_next = PotentialClass(**params_next)
        
        rho_dm_i_next = _shoot_for_rho_dm_i(ic, a_ini, a_end, n_steps, potential_next)
        a, result_next = integrate_cosmo(ic, a_ini, a_end, n_steps, rho_dm_i_next, potential_next)
        omega_de_next, omega_c_final = find_present_day_fractions(result_next, a_ini, rho_dm_i_next, potential_next)
        
        error = (omega_de_next - omega_de_target) / omega_de_target
        logging.info(f"Iter {i+1} (Secant): {param_to_shoot}={p_next:.4e} -> omega_de={omega_de_next:.5f} (err={error:.2e})")

        final_result, final_potential, final_rho_dm_i = result_next, potential_next, rho_dm_i_next
        if abs(error) < 1e-4:
            logging.info("--- Secant Method Converged Successfully! ---")
            converged = True
            break
        p1, p2, omega_de1, omega_de2 = p2, p_next, omega_de2, omega_de_next

    if converged:
        return a, final_result, final_potential, final_rho_dm_i

    # --- Attempt 2: Bisection Method (Fallback) ---
    logging.warning("Outer loop secant method failed. Switching to bisection.")
    if (omega_de1 - omega_de_target) * (omega_de2 - omega_de_target) >= 0:
        raise RuntimeError(f"Outer loop bisection failed: initial guesses do not bracket the solution for {param_to_shoot}. "
                         f"omega_de results were {omega_de1:.5f} and {omega_de2:.5f} for target {omega_de_target:.5f}")

    p1, p2 = guess_1, guess_2 # Reset to original guesses
    for i in range(40): # More iterations for bisection
        p_mid = (p1 + p2) / 2
        params_mid = initial_params.copy(); params_mid[param_to_shoot] = p_mid
        potential_mid = PotentialClass(**params_mid)
        
        rho_dm_i_mid = _shoot_for_rho_dm_i(ic, a_ini, a_end, n_steps, potential_mid)
        a, result_mid = integrate_cosmo(ic, a_ini, a_end, n_steps, rho_dm_i_mid, potential_mid)
        omega_de_mid, _ = find_present_day_fractions(result_mid, a_ini, rho_dm_i_mid, potential_mid)

        error = (omega_de_mid - omega_de_target) / omega_de_target
        logging.info(f"Iter {i+1} (Bisection): {param_to_shoot}={p_mid:.4e} -> omega_de={omega_de_mid:.5f} (err={error:.2e})")

        if abs(error) < 1e-4:
            logging.info("--- Bisection Method Converged Successfully! ---")
            return a, result_mid, potential_mid, rho_dm_i_mid

        if (omega_de_mid - omega_de_target) * (omega_de1 - omega_de_target) < 0:
            p2, omega_de2 = p_mid, omega_de_mid
        else:
            p1, omega_de1 = p_mid, omega_de_mid
            
    raise RuntimeError("Both secant and bisection methods failed to converge.")

# =============================================================================
# Auxiliary structures
# =============================================================================

def calculate_derived_quantities(
    a: np.ndarray, result: np.ndarray, a_ini: float, rho_dm_i: float, potential_object: ScalarPotential
) -> Dict[str, np.ndarray]:
    """
    Post-processes the simulation results to calculate key physical quantities.
    """
    phi_arr, phi_prime_arr = result.T
    phi_i = result[0, 0]

    # Calculate all derived quantities as arrays over the cosmic time (a)
    V_of_phi = potential_object.value(phi_arr)
    rho_dm_arr = rho_dm_i * (phi_arr / phi_i) * (a_ini / a)**3
    rho_phi_arr = 0.5 * phi_prime_arr**2 / (a**2) + V_of_phi
    rho_r_arr = rho_cr * omega_r * a**(-4)
    rho_b_arr = rho_cr * omega_b * a**(-3)
    
    H_curly_arr = np.array([H_curly(y, val_a, val_rho_dm, potential_object)
                            for y, val_a, val_rho_dm in zip(result, a, rho_dm_arr)])

    epsilon = 1e-40
    rho_phi_safe = np.where(np.abs(rho_phi_arr) < epsilon, epsilon, rho_phi_arr)
    phi_safe = np.where(np.abs(phi_arr) < epsilon, epsilon, phi_arr)
    H_curly_safe = np.where(np.abs(H_curly_arr) < epsilon, epsilon, H_curly_arr)

    w_phi = (0.5 * phi_prime_arr**2 / (a**2) - V_of_phi) / rho_phi_safe
    interaction = -rho_dm_arr / phi_safe
    coupling = (phi_prime_arr / (3 * H_curly_safe * phi_safe)) * (rho_dm_arr / rho_phi_safe)
    w_eff = w_phi + coupling
    phi_prime_over_H_phi = phi_prime_arr / (H_curly_safe * phi_safe)

    # --- Calculate Effective Potential ---
    # The total "force" on the field is dV/dphi + interaction_term
    dV_dphi_val = potential_object.derivative(phi_arr)
    if np.isscalar(dV_dphi_val):
        dV_dphi_arr = np.full_like(phi_arr, dV_dphi_val)
    else:
        dV_dphi_arr = dV_dphi_val

    interaction_force_arr = rho_dm_arr / phi_safe
    total_force_arr = dV_dphi_arr + interaction_force_arr
    
    # To integrate wrt phi, we must sort by phi as it may not be monotonic
    sort_indices = np.argsort(phi_arr)
    phi_sorted = phi_arr[sort_indices]
    force_sorted = total_force_arr[sort_indices]
    
    # Integrate and then unsort back to the original time-ordered sequence
    if potential_object.name == "Constant":
        V_eff_integrated_sorted = np.zeros_like(force_sorted)
        unsort_indices = np.argsort(sort_indices)
        V_eff_arr = V_eff_integrated_sorted[unsort_indices]
    else:
        V_eff_integrated_sorted = cumulative_trapezoid(force_sorted, phi_sorted, initial=0)
        # Set the constant of integration so V_eff = V at the start
        V_eff_integrated_sorted += V_of_phi[sort_indices][0] - V_eff_integrated_sorted[0]
        unsort_indices = np.argsort(sort_indices)
        V_eff_arr = V_eff_integrated_sorted[unsort_indices]
    
    phi_d_prime = -3*H_curly_arr * phi_prime_arr / a - a * dV_dphi_arr - a * interaction_force_arr

    delta_rho_cdm = -rho_dm_arr[0]*(a[0]/a)**3 + rho_dm_arr

    w_ds = w_phi / (1-delta_rho_cdm/rho_phi_arr) + 1/(1-rho_phi_arr/delta_rho_cdm) if np.all(delta_rho_cdm != 0) else w_phi

    return {
        "a": a,
        "phi": phi_arr,
        "phi_prime": phi_prime_arr,
        "V_of_phi": V_of_phi,
        "V_eff": V_eff_arr,
        "rho_dm": rho_dm_arr,
        "rho_phi": rho_phi_arr,
        "rho_r": rho_r_arr,
        "rho_b": rho_b_arr,
        "H_curly": H_curly_arr,
        "w_phi": w_phi,
        "interaction": interaction,
        "coupling": coupling,
        "w_eff": w_eff,
        "phi_prime_over_H_phi": phi_prime_over_H_phi,
        "w_ds": w_ds
    }

def run_full_simulation(
    potential_flag: int, params_vector_to_shoot: Dict[str, Any],
    a_ini: float, a_end: float, n_steps: int, phi_i: float, phi_prime_i: float,
    default_guesses: Dict[str, List[float]]
) -> Dict[str, Any]:
    """
    High-level wrapper that configures and runs the entire simulation.
    """
    # 1. Parse the parameter vector to find which parameter to shoot for
    param_to_shoot = None
    initial_params = {}
    for key, value in params_vector_to_shoot.items():
        if value is True:
            param_to_shoot = key
        else:
            initial_params[key] = value
    
    if param_to_shoot is None:
        raise ValueError("Shooting parameter not specified. Set one parameter to `True`.")

    # 2. Get shooting guesses from the provided dictionary
    if param_to_shoot not in default_guesses:
        raise NotImplementedError(f"No default shooting guesses configured for parameter '{param_to_shoot}'")
    guess1, guess2 = default_guesses[param_to_shoot]

    # 3. Run the solver
    a, result, potential_obj, rho_dm_i_final = solve_system(
        ic_phi=phi_i, ic_phi_prime=phi_prime_i, a_ini=a_ini, a_end=a_end, n_steps=n_steps,
        potential_flag=potential_flag, initial_params=initial_params,
        param_to_shoot=param_to_shoot, guess_1=guess1, guess_2=guess2
    )

    # 4. Post-process the results
    derived_quantities = calculate_derived_quantities(
        a, result, a_ini, rho_dm_i_final, potential_obj
    )
    
    # 5. Add initial conditions and final potential to results for plotting
    derived_quantities['potential_object'] = potential_obj
    derived_quantities['initial_conditions'] = {
        'phi_i': phi_i,
        'phi_prime_i': phi_prime_i,
        'a_ini': a_ini
    }
    return derived_quantities

def find_critical_point(scale_factor_arr: np.ndarray, phi_prime_arr: np.ndarray) -> float:
    """
    Calculates the critical point for the scalar field derivative where it changes sign.
    Returns 0.0 if no such crossing is found.
    """
    idx = np.where(np.diff(np.sign(phi_prime_arr)))[0]
    if len(idx) > 0:
        c = idx[0]
        a_c = (scale_factor_arr[c] + scale_factor_arr[c+1]) / 2
        return a_c
    return 0.0

def find_w_phi_critical_points(w_phi: np.ndarray) -> np.ndarray:
    """
    Finds the indices of critical points (local maxima and minima) of the w_phi array.
    """
    # Calculate the sign of the discrete derivative of w_phi
    deriv_sign = np.sign(np.diff(w_phi))
    # Find indices where the sign changes (e.g., from +1 to -1 or vice-versa)
    # We add 1 to the index because np.diff reduces the array length by one.
    critical_indices = np.where(np.diff(deriv_sign) != 0)[0] + 1
    return critical_indices

def find_all_phantom_crossings(a: np.ndarray, w_eff: np.ndarray) -> List[Tuple[float, float]]:
    """
    Detects all instances where w_eff crosses the -1 line (phantom crossings).
    Returns a list of (crossing_a, crossing_z) tuples.
    """
    crossings = []
    # Look for a sign change from w_eff > -1 to w_eff < -1 OR w_eff < -1 to w_eff > -1
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


# Next step is definitely to change the shooting method.
