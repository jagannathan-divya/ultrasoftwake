from skimage.filters import meijering
from skimage.measure import regionprops, label
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import argparse
import numpy as np
import matplotlib
import matplotlib.colors as mc
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# ================================================================
# Program: 2D Inverse Fourier transform (IFFT) to compute the wake
#          pattern on ultrasoft solid for given Froude number (Fr),
#          Bond number (Bo), scaled shear wave speed (cs), and  
#          number of spectral modes (nk).
# Author: Divya Jagannathan (2025)
# ===============================================================

# ============================
# ---USER-DEFINED FUNCTIONS--- 
# ============================

# *** Computes integrand in eq. 14 ***
def integrand(kx, ky, Fr, Bo, cs):
    c1 = 1.0
    c2 = Fr*cs
    c3 = 1/Bo**2
    L = 1.0
    eps = 1e-2
    k = np.sqrt(kx**2 + ky**2)
    f1 = np.exp(-((k*L)/(2*np.pi))**2)
    dpplr = (Fr*kx + 1j*eps)  # (regularisation)
    f2_num = k*(dpplr**2)
    f2_den = (dpplr**2)*(c1*k + c3*k**3) - (dpplr**2 - 2*(c2*k)
                                            ** 2)**2 + 4*(c2*k)**3*np.sqrt((c2*k)**2 - dpplr**2)
    f2 = f2_num/f2_den
    fhatk = f1*f2
    return fhatk

# *** Returns locations of surface elevation extrema along diff ridges ***
def get_peaks(X, Y, W, l_ridges):
    x_peaks, y_peaks, z_peaks = [], [], []
    for label_id in np.unique(l_ridges):
        if label_id < 1:
            continue
        this_ridge_mask = (l_ridges == label_id)
        value_on_this_ridge = np.where(this_ridge_mask, abs(W), -np.inf)
        row, col = np.where(value_on_this_ridge ==
                            np.max(value_on_this_ridge))
        if len(row) < 1:
            print("Error: no peaks found?")
        x_peaks.append(X[row[0], col[0]])
        y_peaks.append(Y[row[0], col[0]])
        z_peaks.append(W[row[0], col[0]])
    return np.array(x_peaks), np.array(y_peaks), np.array(z_peaks)

# *** Identifies ridges per some thresholds, and returns elevation field (W_new) 
#     and orientation (orientation_max) of the leading ridge ***
def get_arms(X, Y, W, Lxy, cs, extra_dom):
    wake_ht_min = 1e-4
    min_strength_ridge = 1e-2
    y_dom_factor = 1.0
    x_dom_factor = 0.75

    # Isolate the second quadrant of the (center-shifted) physical domain
    mask_pos = (Y > 0) & (Y < Lxy/(y_dom_factor*2*extra_dom)) & (X > -Lxy /
                                                                 (extra_dom*x_dom_factor)) & (abs(W) > wake_ht_min)
    W_pos = np.where(mask_pos, W, 0)

    # Identify ridge-like structures
    ridge_detected = meijering(-abs(W_pos),
                               sigmas=range(1, 4), mode='mirror')
    ridge_mask = ridge_detected > min_strength_ridge

    # Identify and label the different multiply-connected ridge regions
    label_unfiltered = label(ridge_mask, connectivity=2)

    # Interpolate W on ridge and create local fine resolution data
    ridge_coords = np.column_stack((X[ridge_mask], Y[ridge_mask]))
    W_ridge_values = W_pos[ridge_mask]
    Xg = X[ridge_mask]
    Yg = Y[ridge_mask]
    W_interp = griddata(ridge_coords, W_ridge_values, (Xg, Yg), method='cubic')
    W_interp_field = np.zeros_like(W_pos)
    W_interp_field[ridge_mask] = W_interp

    # Find peaks of each labelled ridge
    xpeak, ypeak, zpeak = get_peaks(X, Y, W_interp_field, label_unfiltered)
    ridge_regions = regionprops(label_unfiltered)
    regions_with_peaks = list(zip(ridge_regions, np.abs(zpeak)))

    # Extract the leading ridge identified as the one containing the global peak
    region_top, _ = max(regions_with_peaks, key=lambda r: r[1])
    ridge_mask = (label_unfiltered == region_top.label)
    orientation_max = region_top.orientation
    W_new = np.where(ridge_mask, W_pos, 0)
    return ridge_mask, W_new, orientation_max

# *** Plots surface wake (top-view) behind the source in stationary frame ***
def plot_wake_3d(X, Y, W, Lxy, Fr, cs, Nk, extra_dom, nunits):
    Lmbda = 2*np.pi*Fr**2
    l_ridges, Wsub, wangle_r = get_arms(X, Y, W, Lxy, cs, extra_dom)

    # Straight line (xl, yl) with slope of the leading dominant ridge
    xl = np.arange(-4*Lmbda, Lmbda, Lmbda)
    yl = np.tan(wangle_r)*xl
    wangle = -np.degrees(wangle_r)
    
    # Plotting the figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    p = ax.pcolormesh(X, Y, W, cmap=mycolor(),
                      vmin=-1, vmax=1, shading="auto")
    ax.plot(xl, yl, c="k", linestyle='--', linewidth=2,
            label=f"$\\alpha={wangle:.2f}^\\circ$")
    ax.legend(loc="lower center", fontsize=24, frameon=False,
              facecolor='none', labelcolor='k')
    cbar = plt.colorbar(p, ax=ax, ticks=np.arange(-1,
                                                  1.1, 0.5))
    cbar.ax.tick_params(labelsize=20, rotation=90)
    ax.text(0.95, 0.95, rf"$Fr=${Fr:.2f}",
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=24, color='k',
            bbox=dict(facecolor='none', edgecolor='none', pad=2)
            )
    setup_axis(ax, Lxy, Lmbda, extra_dom, nunits,  with_labels=True)
    plt.savefig(f"softwake_Fr{Fr:.3f}_cs{cs:.4f}_Nk{Nk:.1f}.png",
                dpi=1000, bbox_inches='tight')
    plt.close(fig)
    return wangle

# *** auxilliary functions ***
def setup_axis(ax, Lxy, Lmbda,  extra_dom, nunits, with_labels=False):
    ax.set_xlim(-0.75*Lxy/extra_dom, 0.25*Lxy/extra_dom)
    ax.set_ylim(-Lxy/(2*extra_dom), Lxy/(2*extra_dom))
    xticks, yticks = np.arange(-(3*nunits//4), nunits //
                               4, 2), np.arange(-nunits//2+1, nunits//2, 2)
    ax.set_xticks(Lmbda*xticks)
    ax.set_yticks(Lmbda*yticks)
    if with_labels:
        ax.set_xticklabels(xticks, fontsize=20)
        ax.set_yticklabels(yticks, fontsize=20)
        ax.set_xlabel(r"$X/\Lambda_g$", fontsize=24)
        ax.set_ylabel(r"$Y/\Lambda_g$", fontsize=24)


def save_data(X, Y, W, Fr, Bo, cs, wangle, k, if_save_k):
    dinfo = np.array([[Fr, Bo, cs, wangle]])
    np.savetxt(
        f"angle_Fr{Fr:.2f}_Bo{Bo:.2f}_cs{cs:.3f}.dat", dinfo, delimiter=",")
    if if_save_k == 1:
        np.savetxt(f"wavenumbers_Fr{Fr}_cs{cs:.3f}.dat",
                   np.array(k), delimiter=",")

def mycolor():
    basic = plt.get_cmap('bwr')
    bright_red = (1.0, 0.0, 0.0, 1.0)
    bright_blue = (0.0, 0.0, 1.0, 1.0)
    colors = [bright_blue, bright_blue, basic(0.5), bright_red, bright_red]
    nodes = [0, 0.25, 0.5, 0.75, 1]
    mycmap = mc.LinearSegmentedColormap.from_list(
        'rwb_stretched', list(zip(nodes, colors)))
    return mycmap

# User-defined Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--Fr', type=float, required=True, help="Froude number")
parser.add_argument('--Bo', type=float, required=True, help="Bond number")
parser.add_argument('--cs', type=float, required=True, help="Speed factor")
parser.add_argument('--nk', type=int, default=1024,
                    help="Num of spectral space grid points (default:1024)")
args = parser.parse_args()

# ===============================
# ---MAIN PROGRAM STARTS HERE ---
# ===============================

Fr = args.Fr
Bo = args.Bo
cs = args.cs
Nk = args.nk

# Compute the physical domain size: include roughly extra_domain*10 wavelengths
extra_domain = 2
nunits = 6
Lxy = extra_domain*(nunits*2*np.pi*Fr**2)

# Construct the discrete spectral space
kx = 2*np.pi*np.fft.fftfreq(Nk, d=Lxy/Nk) # (angular wavenumers)
ky = kx
KX, KY = np.meshgrid(kx, ky, indexing='ij')

ftransform = integrand(KX, KY, Fr, Bo, cs)

# Zero-padding in SPECTRAL space for finer resolution in the REAL space
pad_factor = int((4096)//Nk)
pad_each = (pad_factor-1)*Nk//2
shifted_ftransform = np.fft.fftshift(ftransform)
padded_shift = np.pad(shifted_ftransform, ((
    pad_each, pad_each), (pad_each, pad_each)), mode='constant', constant_values=0)
fpad = np.fft.ifftshift(padded_shift)

# Compute the 2D ifft to get the surface wave height
fxy = np.fft.ifft2(fpad)
wake = np.real(np.fft.fftshift(fxy))
norm_wake = wake/np.max(np.abs(wake))

# Construct real-space domain
Nx_pad = pad_factor*Nk
x = np.linspace(-0.75*Lxy, 0.25*Lxy, Nx_pad)
y = np.linspace(-Lxy/2, Lxy/2, Nx_pad)
X, Y = np.meshgrid(x, y, indexing='ij')

shifted_wake = np.roll(norm_wake, shift=Nx_pad//2 - Nx_pad//4, axis=0)

# Plotting the wake structure
wAngle = plot_wake_3d(X, Y, shifted_wake, Lxy, Fr,
                      cs, Nk, extra_domain, nunits)

# Saving the wake data on the x-y grid
save_k = 0  # (save_k = 1 if you want to save the wavenumbers used in the ifft)
save_data(X, Y, shifted_wake, Fr, Bo, cs,  wAngle, kx, save_k)

# =============
# ---THE END---
# =============
