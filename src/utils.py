import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt

def calc_SNR(A, B):
    return 10 * np.log10(A / B)

def calc_SNR_e(A, B, A_e, B_e):
    return np.sqrt(
    (10*A_e/(A*np.log(10)))**2
    + (10*B_e/(B*np.log(10)))**2
    )

def gaussian(x, A, b, hwhm, d):
    sigma = hwhm * 2 / (2*np.sqrt(2*np.log(2)))
    return A*np.exp(-0.5*((x-d)/sigma)**2) + b


def lorentzian(x, A, b, hwhm, a):
    # peak = np.exp(-(x-a)**2/w**2)
    peak = (1+(x-a)**2/hwhm**2)**(-1)
    return np.abs(A)*peak + np.abs(b)

def decluster_indices(array, idxs, distance=5):
    """Takes an array and indices of the array. The function returns the index where the array is maximum for groups of indices that are close together."""


    # Then find clusters (because many values are lower than the threshold)
    group_idx = np.argwhere(np.diff(idxs) > distance).flatten()
    # Because we select the elements that split the group, also split the last group from the end of the array.
    group_idx = np.hstack([group_idx, [idxs.size-1]])
    
    # In this loop, the minimum values of arr are used to replace the clusters
    peaks_idxs = []
    temp_idx = 0
    for i in range(len(group_idx)):
        # Start and stop indices of the cluster
        cluster_start_idx = temp_idx
        cluster_end_idx = group_idx[i]
        # Update temp idx
        if i == len(group_idx)-1:
            pass
        else:
            temp_idx = group_idx[i]+1 # Add one because we already used this idx
        
        # Create mask of cluster
        # We need to fill in missing values from the clusters!
        mask = np.clip(np.arange(idxs[cluster_start_idx], idxs[cluster_end_idx]+1, 1), 0, array.size-1)

        # Find minimum arg in cluster mask and add start idx (returned idx is relative to mask)
        peak_idx = np.argmax(array[mask])+idxs[cluster_start_idx]

        
        # Because we took np.diff we are off by 1 idx, correct for this!
        peaks_idxs.append(peak_idx)

    return np.array(peaks_idxs)

def peakfinder(arr, thresh=None, distance_=1, plot=False):
    ddarr = np.diff(np.diff(arr))
    if thresh is None:
        thresh = 2

    # Find an appropriate threshold
    std = np.std(ddarr, ddof=1)
    thresh= -std * thresh

    # Find the dips
    below_threshold = np.argwhere(ddarr < thresh).flatten()

    if plot:
        plt.plot(ddarr/ddarr.max())
        plt.plot(below_threshold, ddarr[below_threshold]/ddarr.max(), '.')
        # plt.show()

    # We should add one, because the greatest negative change in the second derivative happens the index before the peak!
    peak_idx = decluster_indices(arr, below_threshold+1, distance=distance_)

    if plot:
        plt.plot(arr/arr.max())
        plt.plot(peak_idx, arr[peak_idx]/arr.max(), '.')
        plt.show()

    return peak_idx

def find_periodicity(geetwo, plot=False, threshold=None):
    """This is advanced period finding
    First a Fourier transform is used to find an estimate of the period. This estimate is then refined by finding the peak of a
    correlation within the produced uncertainty window.
    """
    guess_period, delta_period = find_periodicity_(geetwo, plot=plot, method="fourier", period_error=True, threshold=threshold)
    
    geetwo_preprocessed = geetwo-geetwo.mean()
    # Correlate the signal
    corr = correlate(geetwo_preprocessed, geetwo_preprocessed, mode="same")**2
    middle_idx = np.ceil(corr.size / 2).astype(np.int_)

    lower_bound_idx = np.clip(middle_idx+guess_period-delta_period, middle_idx+int(1/10*delta_period), None)
    upper_bound_idx = np.clip(middle_idx+guess_period+delta_period, lower_bound_idx+1, corr.size)

    corr_slice = corr[lower_bound_idx:upper_bound_idx]

    
    # # Correct for slope to make it easier, removes the effect of the rectangular window.
    # mean_dcorr = np.mean(np.diff(corr[middle_idx:]))
    # corr_slice -= mean_dcorr * np.arange(corr_slice.size)

    # Find peaks
    pk_idx = np.argmax(corr_slice)
    period = pk_idx + lower_bound_idx - middle_idx
    # print("Period:", period)
    

    if plot:
        pk_idx += lower_bound_idx #  This is for the plot
        plt.plot(corr)
        plt.plot(pk_idx, corr[pk_idx], '.')
        plt.show()
    
    return period


def find_periodicity_(geetwo, plot=False, threshold=None, method="fourier", period_error=False):
    """Given an array geetwo of a periodic peaks, return the distance of the peaks in samples, idx_delta."""

    if method == "fourier":
        padded_geetwo = geetwo #* np.hamming(geetwo.size)
        # padded_geetwo = np.pad(padded_geetwo, padded_geetwo.size*0)
        signal_ft = np.abs(np.fft.rfft(padded_geetwo))**2 # Power spectrum
        signal_freq = np.fft.rfftfreq(padded_geetwo.size)

        # Add the number of idx we do not enter into peakfinder!
        lowpass_cut_freq = 1/geetwo.size

        pks = peakfinder(np.abs(signal_ft)[signal_freq >= lowpass_cut_freq], plot=plot, thresh=threshold) + np.sum(signal_freq < lowpass_cut_freq)
        # pks = pks[signal_freq[pks] > lowpass_cut_freq]
        pk_idx = pks[signal_freq[pks].argmin()]
        pk_freq = signal_freq[pk_idx]

        if plot:
            plt.plot(signal_freq, np.abs(signal_ft[:]/signal_ft.max()))
            plt.plot(signal_freq[pks], np.abs(signal_ft[pks]/signal_ft.max()), ".")
            plt.plot(pk_freq, np.abs(signal_ft[pk_idx])/signal_ft.max(), '.')
            # plt.plot(corr_freq[:], corr_ft[:]/corr_ft.max())
            # plt.plot(geetwo)
            plt.xlim(lowpass_cut_freq, 50*lowpass_cut_freq)
            plt.ylim(0, signal_ft[signal_freq > lowpass_cut_freq].max()/signal_ft.max())
            plt.title("signal ft")
            plt.show()

        # pks = pks[signal_freq[pks] > 0]

        period = 1/pk_freq
        period = np.round(period).astype(np.int_)

        if period_error:
            return period, np.ceil(np.abs(period - 1/signal_freq[pk_idx-1])).astype(np.int_)

    elif method == "correlation":
        # Correlate the signal
        corr = correlate(geetwo, geetwo, mode="same")
        # Find peaks
        pks = peakfinder(corr, thresh=threshold, plot=plot)#find_peaks(-ddcorr, threshold=threshold, distance=distance, prominence=prominance)[0]

        # For visual verification that it is correct!
        if plot:
            plt.plot(corr)
            plt.plot(pks, corr[pks], '.')

        # If there is no periodicity there is no period :)
        if len(pks) <= 1:
            return 0
        elif len(pks) == 2:
            return np.abs(pks[1] - pks[0])
        
        # Find the idx of the highest peak in pks
        highest_peak_pks_idx = corr[pks].argmax()

        # Idx of the correlated signal instead of the pks array
        highest_peak_idx = pks[highest_peak_pks_idx]
        # Find the idx of the peak right of the highest peak
        next_highest_peak_idx = pks[highest_peak_pks_idx+1]

        # For visual verification that it is correct!
        if plot:
            plt.plot(highest_peak_idx, corr[highest_peak_idx], 'o')
            plt.plot(next_highest_peak_idx, corr[next_highest_peak_idx], 'o')
            plt.show()

        period = next_highest_peak_idx - highest_peak_idx
    
    else:
        raise NotImplementedError("Only period finding methods implemented are 'fourier' and 'correlation'.")

    return period

if __name__ == "__main__":
    pass
