import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import math
from scipy.optimize import curve_fit

#location of file
BASE = Path(__file__).resolve().parent
DATA_ROOT = BASE / "tm_aarya_rec1"

# Frame rate (s)
FRAMES_PER_SECOND = 2.0
dt = 1.0 / FRAMES_PER_SECOND

# Constants for error propagation and final calculations
c = 0.12048e-6 # Calibration factor (m/px)
dc = 0.003e-6 # Calibration uncertainty
T = 296.5 # Temperature (K)
dT = 0.5 # Temperature uncertainty
eta = 1.00e-3 # Viscosity of water (Pa*s)
deta = 0.05e-3 # Viscosity uncertainty
radius = 0.95e-6 # Bead radius in meters
dradius = 0.05e-6 # Radius uncertainty

def format_measurement(val, err):
    """Formats a value and uncertainty to 1 significant figure of error."""
    if err == 0 or not math.isfinite(err):
        return f"{val:e} ± {err:e}"
    err_order = math.floor(math.log10(abs(err)))
    err_rounded = round(err, -err_order)
    err_order = math.floor(math.log10(abs(err_rounded)))
    val_rounded = round(val, -err_order)
    if abs(val_rounded) < 1e-3 or abs(val_rounded) > 1e4:
        val_order = math.floor(math.log10(abs(val_rounded)))
        val_norm = val_rounded / 10**val_order
        err_norm = err_rounded / 10**val_order
        decimals = max(0, val_order - err_order)
        return f"({val_norm:.{decimals}f} ± {err_norm:.{decimals}f})e{val_order}"
    else:
        decimals = max(0, -err_order)
        return f"{val_rounded:.{decimals}f} ± {err_rounded:.{decimals}f}"

def load_xy(path: Path):
    '''Loads X and Y columns from the txt file, skipping the first 2 rows and dropping NaNs.'''
    df = pd.read_csv(path, sep="\t", skiprows=2, header=None, names=["X", "Y"])
    return df.dropna()["X"].to_numpy(float), df.dropna()["Y"].to_numpy(float)

# Theoretical Rayleigh PDF (Equation 18 from manual)
def rayleigh_pdf(r, D):
    '''PDF for step length r given diffusion coefficient D and time interval dt.'''
    return (r / (2 * D * dt)) * np.exp(-r**2 / (4 * D * dt))

def main():
    txt_files = sorted(DATA_ROOT.rglob("*.txt"))
    
    # 1. Gather all step lengths (r) for dt = 0.5s
    all_r_px = []
    for p in txt_files:
        x, y = load_xy(p)
        if len(x) < 2: continue
        
        # distance between consecutive frames (lag = 1)
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]
        r_px = np.sqrt(dx**2 + dy**2)
        
        # Filter out tracking glitches: keep only steps smaller than 30 pixels (within reason)
        valid_steps = r_px[r_px < 30] 
        
        all_r_px.extend(valid_steps)
        
    all_r_px = np.array(all_r_px)
    N_steps = len(all_r_px)
    
    # pixels to meters
    all_r_m = all_r_px * c

    # 2. Histogram Data
    # Use density=True so the area under the histogram equals 1, so we can fit a Probability Density Function
    hist_values, bin_edges = np.histogram(all_r_m, bins=30, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    #METHOD 1: SciPy curve fit
    popt, pcov = curve_fit(rayleigh_pdf, bin_centers, hist_values, p0=[1e-12])
    D_fit = popt[0]
    dD_fit = np.sqrt(np.diag(pcov))[0]
    
    # Get k and propagate error for the curve fit
    k_fit = (6 * np.pi * eta * radius * D_fit) / T
    rel_err_k_fit = np.sqrt( (deta/eta)**2 + (dradius/radius)**2 + (dD_fit/D_fit)**2 + (dT/T)**2 )
    dk_fit = k_fit * rel_err_k_fit

    # Format curve fit strings
    D_fit_str = format_measurement(D_fit, dD_fit)
    k_fit_str = format_measurement(k_fit, dk_fit)

    # METHOD 2:MLE
    # Equation 19 from manual
    mean_sq_r = np.mean(all_r_m**2)
    D_mle = mean_sq_r / (4 * dt)
    
    # Error propogation for MLE
    std_sq_r_px = np.std(all_r_px**2)
    err_mean_sq_r_px = std_sq_r_px / np.sqrt(N_steps) # Standard Error of the Mean for r^2
    
    rel_err_D_mle = np.sqrt( (err_mean_sq_r_px / np.mean(all_r_px**2))**2 + (2 * dc / c)**2 )
    dD_mle = D_mle * rel_err_D_mle
    
    k_mle = (6 * np.pi * eta * radius * D_mle) / T
    rel_err_k_mle = np.sqrt( (deta/eta)**2 + (dradius/radius)**2 + (dD_mle/D_mle)**2 + (dT/T)**2 )
    dk_mle = k_mle * rel_err_k_mle

    # Formatting
    k_mle_str = format_measurement(k_mle, dk_mle)
    D_mle_str = format_measurement(D_mle, dD_mle)

    # Plots
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(10, 6))
    
    # Normalized histogram
    plt.hist(all_r_m, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black', label=f'Data Histogram (N={N_steps} steps)')
    
    # Smooth x values for drawing curves
    r_smooth = np.linspace(0, max(all_r_m), 200)
    
    #  Curve fit
    plt.plot(r_smooth, rayleigh_pdf(r_smooth, D_fit), 'r--', linewidth=2, label=f'Curve Fit D: {D_fit:.2e} m²/s')
    
    #  MLE
    plt.plot(r_smooth, rayleigh_pdf(r_smooth, D_mle), 'k-', linewidth=2.5, label=f'MLE D: {D_mle:.2e} m²/s')

    plt.xlabel('Step Length $r$ (meters)', fontsize=18)
    plt.ylabel('Probability Density',   fontsize=18)
    plt.title(f'Distribution of Step Lengths ($\Delta t = {dt}$ s)', fontsize=18)
    plt.legend(loc='upper right', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(BASE / "rayleigh_histogram.png", dpi=150)
    #caption = "Histogram of step lengths (r) with fitted Rayleigh PDFs using both curve fitting and MLE methods. The MLE method is preferred for its direct use of the data's second moment, while the curve fit provides a visual confirmation of the distribution shape."
    
    #percent difference in each for k, compared to theoretical value of k_B = 1.38e-23 J/K
    k_B_theoretical = 1.38e-23
    percent_diff_k_fit = abs(k_fit - k_B_theoretical) / k_B_theoretical * 100
    percent_diff_k_mle = abs(k_mle - k_B_theoretical) / k_B_theoretical * 100



    # Terminal outputs
    print(f"\nStep Distribution Results:---")
    print(f"Total steps: {N_steps}")
    
    print(f"\nMethod 1 (Curve fit):")
    print(f"  D_fit = {D_fit_str} m²/s")
    print(f"  k_fit = {k_fit_str} J/K")
    
    print(f"\nMethod B (MLE):")
    print(f"  D_MLE = {D_mle_str} m²/s")
    print(f"  k_MLE = {k_mle_str} J/K")

    print(f"\nPercent difference from theoretical k_B = {k_B_theoretical:.2e} J/K:")
    print(f"  Curve Fit: {percent_diff_k_fit:.2f}%")
    print(f"  MLE: {percent_diff_k_mle:.2f}%")

if __name__ == "__main__":
    main()