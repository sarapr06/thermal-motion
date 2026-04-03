import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import math
from scipy.optimize import curve_fit

BASE = Path(__file__).resolve().parent
DATA_ROOT = BASE / "tm_aarya_rec1"

# Frame rate (s); ThermalMotion.pdf uses 2 frames/s
FRAMES_PER_SECOND = 2.0
dt = 1.0 / FRAMES_PER_SECOND

def format_measurement(val, err):
    """Formats a value and uncertainty to the correct significant figures."""
    if err == 0 or not math.isfinite(err):
        return f"{val:e} ± {err:e}"
    
    # Standard practice, from textbook is 1 significant figure for error
    err_order = math.floor(math.log10(abs(err)))
    err_rounded = round(err, -err_order)
    
    # Re-check order in case rounding bumped it up (e.g., 0.96 -> 1.0)
    err_order = math.floor(math.log10(abs(err_rounded)))
    val_rounded = round(val, -err_order)
    
    # Scientific notation for very small or large numbers
    if abs(val_rounded) < 1e-3 or abs(val_rounded) > 1e4:
        val_order = math.floor(math.log10(abs(val_rounded)))
        val_norm = val_rounded / 10**val_order
        err_norm = err_rounded / 10**val_order
        decimals = max(0, val_order - err_order)
        return f"({val_norm:.{decimals}f} ± {err_norm:.{decimals}f})e{val_order}"
    else:
        decimals = max(0, -err_order)
        return f"{val_rounded:.{decimals}f} ± {err_rounded:.{decimals}f}"


def load_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    '''Loads X and Y columns from the txt file, skipping the first 2 rows and dropping NaNs.'''
    df = pd.read_csv(path, sep="\t", skiprows=2, header=None, names=["X", "Y"])
    df = df.dropna()
    return df["X"].to_numpy(float), df["Y"].to_numpy(float)

def msd_lag(x: np.ndarray, y: np.ndarray, max_lag: int) -> np.ndarray:
    """MSD for lags 1..max_lag: average of (x(t+lag)-x(t))²"""
    out = np.zeros(max_lag)
    for i, lag in enumerate(range(1, max_lag + 1)):
        dx = x[lag:] - x[:-lag]
        out[i] = np.mean(dx * dx)
    return out

# Linear model for curve_fit
def linear_model(x, slope, intercept):
    return slope * x + intercept

def main() -> None: # Main function to execute the analysis
    # Load trajectories from txt files
    txt_files = sorted(DATA_ROOT.rglob("*.txt"))
    
    # Calculate MSD for each trajectory and store in a list
    trajectories: list[tuple[str, np.ndarray, np.ndarray]] = []
    for p in txt_files:
        x, y = load_xy(p)
        if len(x) < 40:
            continue
        label = str(p.relative_to(DATA_ROOT))
        trajectories.append((label, x, y))

    # Maximum lag
    max_lag = min(min(len(y) // 4 for _, x, y in trajectories), 100)
    max_lag = max(max_lag, 1)

    # MSD for each trajectory
    series = [msd_lag(x, y, max_lag) for _, x, y in trajectories]
    stack = np.vstack(series)
    msd_mean = np.mean(stack, axis=0)

    # Error bars
    msd_std = np.std(stack, axis=0)
    N_traj = stack.shape[0]
    msd_err = msd_std / np.sqrt(N_traj) 
    time_err = 0.03 # Constant time error from the lab manual

    # Time lags
    lags = np.arange(1, max_lag + 1)
    time_lags = lags * dt

    # Weighted linear fitting (curve_fit)
    # Initial fit to get approximate slope
    popt_init, _ = curve_fit(linear_model, time_lags, msd_mean)
    slope_approx = popt_init[0]
    
    # Account for both x and y errors using effective variance
    msd_err_total = np.sqrt(msd_err**2 + (slope_approx * time_err)**2)

    # Final rit
    popt, pcov = curve_fit(linear_model, time_lags, msd_mean, sigma=msd_err_total, absolute_sigma=True)
    slope_px, intercept_px = popt
    dslope_px, dintercept_px = np.sqrt(np.diag(pcov))
    fit_line = linear_model(time_lags, slope_px, intercept_px)

    # Chi-Squared and residuals
    residuals = msd_mean - fit_line
    chisq = np.sum((residuals / msd_err_total)**2)
    red_chisq = chisq / (len(time_lags) - 2)
    
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)

    # Constants for error propagation and final calculations
    c = 0.12048e-6 # Calibration factor (m/px)
    dc = 0.003e-6 # Calibration uncertainty
    T = 296.5 # Temperature (K)
    dT = 0.5 # Temperature uncertainty
    eta = 1.00e-3 # Viscosity of water (Pa*s)
    deta = 0.05e-3 # Viscosity uncertainty
    r = 0.95e-6 # Bead radius in meters
    dr = 0.05e-6  # Radius uncertainty

    # True slope and error propagation
    slope_true = slope_px * (c**2)
    rel_err_slope = np.sqrt((dslope_px / slope_px)**2 + (2 * dc / c)**2)
    dslope_true = slope_true * rel_err_slope

    # 1D diffusion coefficient 
    D_true = slope_true / 2.0
    dD_true = dslope_true / 2.0

    # Boltzmann Constant k 
    k_B = (6 * np.pi * eta * r * D_true) / T
    rel_err_k = np.sqrt(
        (deta / eta)**2 + 
        (dr / r)**2 + 
        (dD_true / D_true)**2 + 
        (dT / T)**2
    )
    dk_B = k_B * rel_err_k

    # Format output strings
    slope_str = format_measurement(slope_px, dslope_px)
    intercept_str = format_measurement(intercept_px, dintercept_px)
    D_str = format_measurement(D_true, dD_true)
    k_str = format_measurement(k_B, dk_B)

    # Plotting 
    plt.rcParams.update({'font.size': 18}) # Global font size
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    # Main plot (top)
    ax1.errorbar(time_lags, msd_mean, xerr=time_err, yerr=msd_err, fmt='ko', capsize=4, elinewidth=1.5, markersize=5, label='Mean MSD ± SEM')
    ax1.plot(time_lags, fit_line, 'r-', linewidth=2, label=f"Weighted Fit: y = ({slope_str})·τ + ({intercept_str})")
    ax1.set_ylabel(r"Mean squared distance $\langle x^2\rangle$ (pixels$^2$)")
    ax1.set_title("Mean squared distance vs. time lag (1D)")
    ax1.legend(loc="best", fontsize=16)
    ax1.grid(True, linestyle='--')

    # Residuals (bottom)
    ax2.errorbar(time_lags, residuals, yerr=msd_err_total, fmt='ko', capsize=4, elinewidth=1.5, markersize=5)
    ax2.axhline(0, color='black', lw=1.5)
    ax2.set_xlabel(r"Time lag $\tau$ (s)")
    ax2.set_ylabel('Residuals (px$^2$)')
    ax2.grid(True, linestyle='--')

    plt.tight_layout()
    plt.savefig(BASE / "msd_plot.png", dpi=150)

    # Save CSV
    out = {"Time_Lag_s": time_lags, "MSD_mean_px2": msd_mean, "MSD_fit_px2": fit_line, "Residuals_px2": residuals}
    for (label, _, _), msd in zip(trajectories, series):
        safe = label.replace("/", "_")[:80]
        out[f"MSD_{safe}"] = msd
    pd.DataFrame(out).to_csv(BASE / "msd_results.csv", index=False)

    # Outputs in terminal
    print(f"{len(trajectories)} files from {DATA_ROOT}")
    print(f"Lags 1..{max_lag}, dt = {dt} s")
    print(f"\nFits:---")
    print(f"Mean Residual: {mean_residual:.3e} px²")
    print(f"Standard Deviation of Residuals: {std_residual:.3e} px²")
    print(f"Reduced Chi-Squared: {red_chisq:.2f}")
    print(f"\nResults: ---")
    print(f"Equation: y = ({slope_str})·τ + ({intercept_str}) px²")
    print(f"Diffusion Coefficient (D): {D_str} m²/s")
    print(f"Boltzmann Constant (k_B):  {k_str} J/K")

    #percent difference from accepted value of k_B = 1.380649e-23 J/K
    k_B_accepted = 1.380649e-23
    percent_diff_k = abs(k_B - k_B_accepted) / k_B_accepted
    print(f"Percent Difference from Accepted k_B: {percent_diff_k:.2%}")

if __name__ == "__main__":
    main()