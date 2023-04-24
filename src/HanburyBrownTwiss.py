import numpy as np
from tqdm import tqdm
from src.photon_generator import Detector
from scipy.signal import correlate

# Basis functions needed to perform the simulation (with full cross-correlation)

def beam_splitter(emitted_times, chance1=0.5):
    '''Split the given beam into two by using probablistic chance {chance1}.
    
    Inputs:
        emitted_times (nparray): 1D array containing photon emission times
        chance1 (float): chance (0-1) for photon to go to output/detector 1
        
    Ouputs:
        output1 (nparray): 1D array containing photon emission times of photons at output/detector 1
        output2 (nparray): 1D array containing photon emission times of photons at output/detector 2
    '''
    # Generate random directions
    directions = np.random.choice([True, False], p=[chance1,1-chance1], size=len(emitted_times))
    
    # Create two arrays for the two BS outputs
    output1 = np.copy(emitted_times[directions])
    output2 = np.copy(emitted_times[np.logical_not(directions)])
    
    return output1, output2


def dead_time(stream, t_dead):
    '''Apply dead time by removing undetected photons from the stream.
    
    Inputs:
        stream (nparray): 1D array containing photon emission times in units [delta_t]
        t_dead (float): dead time of the detector in units [delta_t]

    Ouputs:
        stream (nparray): 1D array containing photon emission times in units [delta_t]
        '''
    # Remove doubles
    stream = np.unique(stream)
    
    # Calculate differences
    diffs = np.diff(stream)
    too_small = diffs <= t_dead
    
    # Remove undetectable photons untill all differences are large enough
    while np.sum(too_small) > 0:
        first_trues = np.logical_and(np.diff(too_small), np.logical_not(too_small)[:-1])
        undetected = np.append([False,too_small[0]], first_trues)
        stream = stream[np.logical_not(undetected)]
        
        diffs = np.diff(stream)
        too_small = diffs <= t_dead
    
    # print("\n", stream.size/5e-1)
    return stream

from numba import njit
# Yay I spent 3 hours making this marginally faster
# with njit:    10/10 [00:35<00:00,  3.57s/it]
# without njit: 10/10 [00:37<00:00,  3.71s/it]
@njit
def sparse_difference_matrix(stream1, stream2, max_difference):
    '''Gets all relevant differences between stream1 and stream2.
    Assumes both stream arrays are sorted!, loop is stopped when distance is greater than max_difference.
    Use same units for all inputs!
    
    Inputs:
        stream1 (nparray): 1D array containing photon emission times at output/detector 1
        stream2 (nparray): 1D array containing photon emission times at output/detector 2
        max_difference (float): time difference at which the differences are truncated

    Ouputs:
        time_differences (nparray): 1D array containing all time differences < max_difference between arrays
    '''
    # Alocate memory for output
    time_differences = np.empty(stream1.size*max_difference*4, dtype=np.int64)
    # But we will not use the whole array, so keep track of where to end
    time_differences_idx = 0

    # Memory for the inner loop
    temp_differences = np.empty(stream2.size)

    # For the inner loop, the last useful time idx of stream1
    j_start = 0

    for i in range(len(stream1)):

        for store_index, j in enumerate(range(j_start, len(stream2))):
            # Calc time differnces
            temp_differences[store_index] = stream2[j]-stream1[i]

            # Check if following calculations are useful
            if abs(temp_differences[store_index]) > max_difference: 
                if temp_differences[store_index] < 0:
                    j_start = j+1
                else:
                    break
        # Write to output array
        if time_differences_idx+store_index > time_differences.size:
            print("Time difference overflow")
            return time_differences[:time_differences_idx]
            raise OverflowError("time_differences is too small, think about it you lazy sob")
        time_differences[time_differences_idx:time_differences_idx+store_index] = temp_differences[:store_index]
        # Update index of output array
        time_differences_idx += store_index
    if time_differences_idx >= time_differences.size:
        raise OverflowError("Trying to read higher idx than exists!")

    # Return only the indices we wrote to
    return time_differences[:time_differences_idx]


def g2_tdc(output1, output2, delta_tdc, max_tau, delta_t=1, return_counts=False, method="direct"):
    '''Measure the second order correlation function from timestamps (full cross-correlation), using a 
        time-to-digital-converter with finite binsize. Calculate differences and bin as tau values to 
        deal with large arrays. 
    
    Inputs:
        output1 (nparray): 1D array containing photon emission times of photons at output/detector 1 in units [delta_t]
        output2 (nparray): 1D array containing photon emission times of photons at output/detector 2 in units [delta_t]
        delta_tdc (float): Finite time response of TDC (binsize used to bin time stamps) in [ns]
        max_tau (float): Value at which time differences are truncated, in [ns]
        delta_t (float): Time discretization of photon stream in [ns]
        return_counts (boolean): If true, return histogram of counts instead of g2
        
    Outputs:
        taus (nparray): 1D array containing values for the time delay tau in [ns]
        g2 (nparray): 1D array containing calculated g2 values
        e_g2 (nparray): 1D array containing errors on the g2 values
    '''
    # Bin the timstamps according to TDC bins (units [delta_tdc])
    bin_numbers1 = (output1//(delta_tdc/delta_t)).astype('int64')#.reshape(-1,1)
    bin_numbers2 = (output2//(delta_tdc/delta_t)).astype('int64')#.reshape(-1,1)
    
    # Get all differences between the two arrays



    if method=="convolution":
        maxbin = max(bin_numbers1.max(), bin_numbers2.max())
        bin_edges = np.arange(-0.5, maxbin+0.5, 1)
        bin_counts1 = np.histogram(bin_numbers1, bins=bin_edges)[0]
        bin_counts2 = np.histogram(bin_numbers2, bins=bin_edges)[0]
        coincidence_counts = correlate(bin_counts1, bin_counts2) 

        # Cut the array
        coincidence_counts = coincidence_counts[int(coincidence_counts.size/2-max_tau/delta_tdc+1):int(coincidence_counts.size/2+max_tau/delta_tdc-1)]   
        bin_edges = bin_edges[int(bin_edges.size/2-max_tau/delta_tdc):int(bin_edges.size/2+max_tau/delta_tdc-1)]
        bin_edges -= bin_edges[0] - 0.5 + max_tau/delta_tdc
   
    elif method=="direct":
        # Get the tau values and g2
        tau_values = sparse_difference_matrix(bin_numbers1, bin_numbers2, int(max_tau/delta_tdc))
        hist_edge = max_tau//delta_tdc
        bin_edges = np.arange(-hist_edge-0.5,hist_edge+0.5, 1)  # units [delta_tdc]
        coincidence_counts, bin_edges = np.histogram(tau_values, bin_edges)

    else:
        raise ValueError("Only 'direct' and 'convolution' methods are possible to cacluate the correlations.")
        
    taus = ((bin_edges+0.5)*delta_tdc)[:-1]
    
    g2 = coincidence_counts*max(np.max(bin_numbers1), np.max(bin_numbers2))/(len(bin_numbers1)*len(bin_numbers2))

    
    # Estimate the error using shot noise
    e_coincidence_counts = np.sqrt(coincidence_counts)
    e_g2 = e_coincidence_counts*max(np.max(bin_numbers1), np.max(bin_numbers2))/(len(bin_numbers1)*len(bin_numbers2))
    
    if return_counts: return taus, coincidence_counts, e_coincidence_counts, len(bin_numbers1), len(bin_numbers2)
    return taus, g2, e_g2


def real_detectors_hbt(stream, max_tau, eff, t_dead, t_jitter, delta_tdc, delta_t, return_counts=False):
    '''Simulate the HBT experiment with real detectors (full cross-correlation up to max_tau)

    Inputs:
        stream (nparray): 1D array containing photon emission times in units [delta_t]
        max_tau (float): Value at which time differences are truncated, in [ns]
        eff (float): Quantum efficiency of the detector (scale [0,1])
        t_dead (float): Dead time of the detectors in [ns]
        t_jitter (float): Timing jitter (FWHM of gauss) of detectors in [ns]
        delta_tdc (float): Time discretization of time-to-digital converter [ns]
        delta_t (float): Time discretization of photon stream in [ns]
        return_counts (boolean): If true, return histogram of counts instead of g2

    Ouputs:
        taus (nparray): 1D array containing values for the time delay tau in [ns]
        g2 (nparray): 1D array containing calculated g2 values
        e_g2 (nparray): 1D array containing errors on the g2 values
    '''
    stream_detected = Detector(t_jitter, eff)(stream)

    # Beam splitter
    detector1, detector2 = beam_splitter(stream_detected)
    
    # Dead time
    detector1 = dead_time(detector1, t_dead/delta_t)
    detector2 = dead_time(detector2, t_dead/delta_t)
    
    # Time-to-digital converter and g2 calculation
    if return_counts: 
        taus, CC, e_CC, SC1, SC2 = g2_tdc(detector1, detector2, delta_tdc, max_tau, delta_t, return_counts=True)
        return taus, CC, e_CC, SC1, SC2
    
    taus, g2, e_g2 = g2_tdc(detector1, detector2, delta_tdc, max_tau, delta_t)
    
    return taus, g2, e_g2

def g2_experiment(light_func, n_runs_r, max_tau, eff=0.6, t_dead=5, t_jitter=0.1, delta_tdc=0.01):
    '''Simulate the HBT experiment; compare perfect vs real detectors. (Cross-correlate all timestamps up to max_tau)

    Inputs:
        light_func (function): function which can be called to generate 1 light stream (1 run)
        n_runs_p (integer): number of runs for the perfect detector
        max_tau (float): Value at which time differences are truncated, in [ns]
        delta_t (float): Time discretization of photon stream in [ns]
        eff (float): Quantum efficiency of the detector (scale [0,1])
        t_dead (float): Dead time of the detectors in [ns]
        t_jitter (float): Timing jitter of detectors in [ns]
        delta_tdc (float): Time discretization of time-to-digital converter [ns]

    Ouputs:
        none, plots a figure
    '''

    # Get g2 data for realistic detector
    g2s = []
    e_g2s = []

    with tqdm(total=n_runs_r) as pbar:
        for i in range(n_runs_r):
            photon_stream = light_func()
            taus, g2, e_g2 = real_detectors_hbt(photon_stream, max_tau, eff, t_dead, t_jitter, delta_tdc, 1)
            g2s.append(g2)
            e_g2s.append(e_g2)
            pbar.update(1)
    
    
    # convert data to nparrays
    g2s = np.array(g2s)
    e_g2s = np.array(e_g2s)
    
    e_g2s_arr = np.sqrt(np.sum(e_g2s**2, axis=0))

    return taus, np.sum(g2s, axis=0), e_g2s_arr


if __name__ == "__main__":
    from photon_generator import PhotonGenerator

    # Generate stream
    photon_chance = 0.7   # photons / sec
    background = 0#1e8   # photons / sec
    pulse_width = 20   # in ns
    period = 20   # in ns
    delta_t = 1   # in ns

    # Detector parameters
    eff = 0.6
    t_dead = 50  # in ns
    t_jitter = 0.532  # in ns
    delta_tdc = 0.1  # in ns

    n_runs_p = 50
    n_runs_r = 50
    max_tau = 50  # ns

    # Simulation
    pulse_widths = np.array([0.05,0.6, 0.8, 0.9, 1.02])
    extinction_ratios = np.array([150, 200, 250, 280, 300])
    purities = np.array([1,])
    # background = np.logspace(7, 9, 5)
    taus = []
    g2s = []
    e_g2s = []
    for i in range(pulse_widths.size):
        pg = PhotonGenerator(photon_chance=photon_chance, purity=1, extinction_ratio=extinction_ratios[i])
        light_func = lambda: pg.photon_stream(N_pulses=200, pulse_width=pulse_widths[i], period=period, background=background)
        # light_func = lambda: initialize_pulsed_block_light(photon_chance, pulse_widths[i], period, background, delta_t, N_pulses=5000, extinction_ratio=extinction_ratios[i])
        tau, g2, e_g2_arr = g2_experiment(light_func, n_runs_p, max_tau, eff, t_dead, t_jitter, delta_tdc)
        taus.append(tau)
        g2s.append(g2)
        e_g2s.append(e_g2_arr)