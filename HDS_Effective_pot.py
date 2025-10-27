import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

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

from HDSSolver import global_alpha, find_all_phantom_crossings, find_w_phi_critical_points, run_full_simulation

def V_eff(results: Dict[str, Any]):
    
    phi_vals = results['phi']  # arrays of φ(a), V(φ)
    rho_cdm = results['rho_dm']  # arrays of ρ_cdm(a)
    scale_factor = results['a']  # arrays of a

    # Interpolate ρ_cdm(a) onto φ(a)
    a_of_phi = interp1d(phi_vals[::-1], scale_factor[::-1], fill_value="extrapolate")
    rho_interp = interp1d(scale_factor, rho_cdm, fill_value="extrapolate")
    rho_phi = rho_interp(a_of_phi(phi_vals))

    # Integrand and integration
    integrand = rho_phi / phi_vals
    integral = cumulative_trapezoid(integrand, phi_vals, initial=0.0)

    # Constant offset V0 (choose the first point of V_phi)
    V0 = results['potential_object'].value(results['phi'][0])
    V_eff = V0 + integral

    return phi_vals, V_eff

if __name__ == '__main__':

    potential_flag = 1

    initial_conditions = {
        'a_ini': 1e-5,
        'a_end': 1.0,
        'n_steps': 50000,
        'phi_i': 20,
        'phi_prime_i': 0
    }

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

    # params_vector_to_shoot = {'A': True, 'B': -0.23} # This one is interesting for Exponential potential
    # params_vector_to_shoot = {'A': True, 'B': 1/phi_i_base**2, 'C':phi_i_base/10} # This one is interesting for Gaussian potential
    params_vector_to_shoot = {'V0': True} # bissection is not working properly here, need to check
    # params_vector_to_shoot = {'V0': True, 'A': 8.59, 'B': phi_i_base-1.1}

    results = run_full_simulation(
                    potential_flag=potential_flag,
                    params_vector_to_shoot=params_vector_to_shoot,
                    a_ini=a_ini, a_end=a_end, n_steps=n_steps,
                    phi_i=phi_i_base, phi_prime_i=phi_prime_i_base,
                    default_guesses=default_guesses
                )

    print("\n--- Final Converged Solution & Results ---")
    print(f"Final Potential: {results['potential_object']}")
    print(results['potential_object'].value(results['phi'][0]))


    plt.plot(results['phi'], V_eff(results)[1], label='V(φ)')
    #plt.yscale('log')
    plt.show()
