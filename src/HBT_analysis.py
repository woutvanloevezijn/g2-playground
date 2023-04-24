import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import sys
from src.utils import lorentzian, find_periodicity


def average_peaks(geetwo, idx_delta, plot=False):
    # Reshape into batches of idx_delta (one peak per batch)
    geetwo_reshaped = geetwo[:geetwo.size-np.mod(geetwo.size, idx_delta)].reshape((-1, idx_delta))
    
    # Iterate over the batches and find the peak
    for j, gt in enumerate(geetwo_reshaped):
        pk = int( gt.size / 2)
        
        # Pick a relative point: the peak of the first batch
        if j == 0:
            idx_rel = geetwo_reshaped[0].argmax()
        
        # Roll arrays to align peaks
        geetwo_reshaped[j] = np.roll(gt, pk - idx_rel)

        if plot:
            # plt.plot(pk, gt[idx_rel], '.')
            plt.plot(gt)
            # plt.plot(j, pk, '.')
    if plot:
        plt.xlabel("Peak")
        plt.ylabel(f"Peak time mod {idx_delta}")
        plt.show()

    # Do statistics and renormalize
    gt = geetwo_reshaped.mean(axis=0)
    gt /= gt.shape[0]
    gt_e = geetwo_reshaped.std(axis=0, ddof=1) / gt.shape[0]

    return gt, gt_e

def fit_func(x, *args):
    return lorentzian(x, *args)


def make_fit(tau, gt, gt_e, plot=False, lock_peak = 0, A_max=1, B_max=1, fit_func=None):
    if fit_func is None:
        fit_func = lorentzian
    
    peak_tau = tau[gt.argmax()]
    if lock_peak:
        fit_func_ = lambda x, *args: fit_func(x, *args, a=peak_tau)
        p0 = [1, 0, 0.01]
        bounds = (
            (0, 0, 0),
            (A_max, B_max, tau[-1]) # If you do not normalize g2, adjust A- and B_max
        )

    else:
        b_guess = np.mean(gt)
        print(b_guess)
        p0 = [1-b_guess, b_guess, 0.01, peak_tau]
        fit_func_ = fit_func

        bounds = (
            (0, 0, 0, tau[0]),
            (A_max, B_max, np.inf, tau[-1]) # If you do not normalize g2, adjust A- and B_max
        )
    if np.sum(gt_e == 0) > 0:
        gt_e = None

    popt, pcov = curve_fit(fit_func_, tau, gt, p0=p0, sigma=gt_e, bounds=bounds)
    
    if lock_peak:
        popt = np.hstack([popt, [peak_tau]])

    if plot:
        plt.errorbar(tau, gt, yerr=gt_e, label="data", fmt=".")
        plt.plot(tau, fit_func_(tau, *popt), label="fit")
        plt.title("Fit check")
        plt.ylabel("$g^{(2)}(\\tau)$")
        plt.xlabel("$\\tau [ns]$")
        plt.legend()
        plt.show()
    
    return popt, pcov

def ER(D, b, A):
    top = 1-D+np.sqrt(D**2+D*b/A)
    bottom = -D+np.sqrt(D**2+D*b/A)
    return top / bottom

def calculate_photon_ratio_error(b, b_e, geetwo, geetwo_e, tau, N_peaks):
    delta_t = tau[1] - tau[0]
    Lambda = np.sum(geetwo - b) * delta_t
    Lambda = np.max([Lambda, 0])
    Lambda_e = np.var(geetwo_e)

    N_p = np.sqrt(( Lambda  ) / N_peaks**2)
    N_p_e = np.sqrt((1 / (2*N_p) / N_peaks**2)**2 * (Lambda_e ** 2 + (tau[-1]*N_peaks)**2 * b_e**2))
    N_b = np.sqrt(Lambda+b*delta_t*geetwo.size) / N_peaks - N_p
    N_b_e = np.sqrt(1/(2*Lambda)**2 * Lambda_e **2 + N_p_e**2)

    per_e = np.sqrt(
        1/N_b**2 * N_p_e**2
        + N_p**2/N_b**4 * N_b_e**2
    )
    # print(N_p_e, N_b_e, Lambda_e, b, b_e)

    print("N_pulse: ", N_p, "\t N_bg: ", N_b, "\t PER: ", N_p/N_b, " +/- ", per_e)
    
    return N_p/N_b, per_e

def process_geetwo(geetwo, tau, plot_fit=False, period=None):
    if period is None:
    # Find periodicity to overlay the peaks
        idx_delta = find_periodicity(geetwo, plot=plot_fit)
        period = tau[:idx_delta] - tau[0]
    else:
        idx_delta = np.argmin(np.abs(period - (tau- tau[0])))
        period = tau[:idx_delta] - tau[0]

    b, b_err = estimate_background(geetwo, plot=plot_fit)

    N_peaks = np.floor(find_number_of_peaks(geetwo, period)).astype(np.int_)
    geetwo = geetwo[:N_peaks*period.size] # Cut to integer number of peaks

    geetwo_e = np.sqrt(2*geetwo**1.5)

    return b, b_err, geetwo, geetwo_e, period, N_peaks

def find_number_of_peaks(geetwo, period):
    return geetwo.size / period.size

def mode(array):
    dees = np.abs(np.diff(array))
    mean = np.mean(array)
    std = np.std(array)
    bins = np.arange(np.min(array), np.max(array), np.min(dees[dees > 0]))

    counts, bins = np.histogram(array, bins=bins)

    return bins[np.argmax(counts)]

def estimate_background(geetwo, plot=True):
    mean = np.mean(geetwo)
    std = geetwo.std(ddof=1)
    mask = geetwo <= mean+2*std
    mode_ = mode(geetwo.flatten())
    print(mode_, mean)
    mode_ = mode_

    masked_mean = geetwo[mask].mean()
    mode_e = np.abs(mode_ - masked_mean)

    if plot:
        dees = np.abs(np.diff(geetwo))
        bins = np.arange(mean-5*std, mean+5*std, np.min(dees[dees > 0]))
        counts, bins = np.histogram(geetwo, bins=bins)
        x = np.array([0, counts.max()])
        y = np.array([mode_, mode_])
        fig, ax = plt.subplots(1, 2, sharey=True)
        ax[0].plot(np.arange(geetwo.size)[mask], geetwo[mask])
        ax[0].errorbar([0, geetwo.size], y, yerr=mode_e)
        ax[1].plot(counts, bins[:-1])
        ax[1].errorbar(x, y, yerr=mode_e)
        plt.ylim(bins.min(), bins.max())
        plt.show()

    return mode_, mode_e

def process_geetwo_periodic(geetwo, tau, plot_fit=False):
    # Find periodicity to overlay the peaks
    idx_delta = find_periodicity(geetwo, plot=plot_fit)
    period = tau[:idx_delta]
    
    if idx_delta == 0:
        
        if plot_fit:
            plt.errorbar(tau[:geetwo.size], geetwo)
            plt.show()
        return geetwo, None, tau
    
    else:
        # Overlay the peaks
        gt, gt_e = average_peaks(geetwo, idx_delta,plot=plot_fit)

        if plot_fit:
            # gt_e=0
            plt.errorbar(tau[:gt.size], gt, yerr=gt_e)
            plt.show()
    
        return gt, gt_e, period-period[0]


def process_geetwos(geetwos, tau, timings=None, plot_fits=False):
    popts = []
    pcovs = []
    gts = []
    gt_es = []
    if timings is not None:
        timing_deltas = []
    periods = []

    for i, geetwo in enumerate(geetwos):
        gt, gt_e, period = process_geetwo(geetwo, tau, plot_fit=plot_fits)
        popt, pcov = make_fit(tau[:gt.size], gt, gt_e, plot=plot_fits)
        popts.append(popt)
        pcovs.append(pcov)
        periods.append(period)
        gts.append(gt)
        gt_es.append(gt_e)

        if timings is not None:
            timing_delta = float(timings[i][1]) - float(timings[i][0])
            timing_deltas.append(timing_delta)

    popts = np.array(popts)
    pcovs = np.array(pcovs)
    periods = periods
    if timings is not None:
        timing_deltas = np.array(timing_deltas)
        return popts, pcovs, periods, timing_deltas, gts, gt_es
    
    return popts, pcovs, periods

def window_tau(gt, gt_e, tau, cut_tau_start=0, cut_tau_end=99999):
    cut_idx = np.argmin(np.abs(tau - cut_tau_start))
    gt = gt[cut_idx:]
    if gt_e is None:
        pass
    else:
        gt_e = gt_e[cut_idx:]
    tau = tau[cut_idx:]

    cut_idx = np.argmin(np.abs(tau - cut_tau_end))
    gt = gt[:cut_idx]
    if gt_e is None:
        pass
    else:
        gt_e = gt_e[:cut_idx]
    tau = tau[:cut_idx]

    return gt, gt_e, tau

