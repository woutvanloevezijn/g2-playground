import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt

class PhotonGenerator():
    def __init__(self, photon_chance, purity, extinction_ratio, qd_lifetime=0.1) -> None:
        self.photon_chance = photon_chance
        self.purity = purity
        self.extinction_ratio = extinction_ratio
        self.qd_lifetime = qd_lifetime

    def photon_stream(self, N_pulses, pulse_width, period, background, pulse_shape = "square"):
        return self.photon_stream_generator(N_pulses, pulse_width, period, background, pulse_shape)()

    def photon_stream_generator(self, N_pulses, pulse_width, period, background, pulse_shape = "square"):
        self.N_pulses = N_pulses
        self.pulse_width = pulse_width
        self.period = period
        self.background = background
        self.pulse_shape = pulse_shape

        return self.generate
    
    def generate(self):
        '''Create vector containing times of photon emissions. Use Poissonian distribution for laser light.
        Create block pulses.
        Probability to emit n photons: Pn = 1/tau_p * exp(-t/tau_p)
        
        Inputs:
            N_photons (int): number of photons generated per second in the pulse, so not the measured intensity!
            pulse_width (float): width/duration of one block pulse [ns]
            period (float): duration between centers of pulses in [ns]
            background (float): number of photons per second in the background (uniformly over entire stream)
            delta_t (float): Time discretization of photon stream in [ns]
            total_time (float): total time of light stream in [s]
        
        Ouputs:
            emitted_times (nparray): 1D array containing photon emission times in units [delta_t]
        '''

        # Calculate number of photons per pulse and in background
        background_photons = int(self.background*(self.N_pulses+0.5)*self.period*1e-9)
        # print(background_photons)
        duty_cycle = (self.pulse_width / self.period)
        pulse_brightness = self.photon_chance * (self.N_pulses+0.5) 
        input_brightness = pulse_brightness / duty_cycle
        extinction_photons = int(input_brightness / self.extinction_ratio)

        if self.pulse_shape == "square":
            photon_times = self.square_pulse()
        elif self.pulse_shape == "gaussian":
            photon_times = self.gaussian_pulse()
        else:
            raise NotImplementedError("Only 'square' pulse shapes are implemented")
        
        # # Generate timestamps for background photons darkcounts!!!!
        background_times = np.random.uniform(0, (self.N_pulses+0.5)*self.period, size=background_photons)

        # # Generate timestamps for extinction photons
        extinction_times = np.sort(np.random.uniform(0, (self.N_pulses+0.5)*self.period, size=extinction_photons))

        # sort
        emitted_times = np.hstack((photon_times, extinction_times))
        emitted_times = np.sort(emitted_times)
        if self.qd_lifetime > 0:
            # They are anti bunched!
            emitted_times = emitted_times[:-1][np.diff(emitted_times) > self.qd_lifetime]
        
        if self.purity < 1:
            impure = np.random.choice([0, 1], p=[self.purity, 1-self.purity], size=emitted_times.size)
            impure_photons = emitted_times[impure == 1]
            emitted_times = np.hstack((emitted_times, impure_photons))
        
        # Darkcounts!!!
        emitted_times = np.hstack((emitted_times, background_times))

        return emitted_times
    
    def square_pulse(self):
        # Generate timestamps for pulse photons
        pulse_times = np.arange(0, self.N_pulses, 1) * self.period
        photon_exists = np.random.choice([1, 0], p=[self.photon_chance, 1-self.photon_chance], size=self.N_pulses)
        photon_bin_times = np.random.uniform(0, self.pulse_width, size=np.sum(photon_exists)) # draw random times in a pulse
        photon_times = pulse_times[photon_exists==1] + photon_bin_times    # Add pulse times plus random (poisonian) offset to make temporally spaced clusters

        return photon_times
    
    def gaussian_pulse(self):
        # Generate timestamps for pulse photons
        pulse_times = np.arange(0, self.N_pulses, 1) * self.period
        photon_exists = np.random.choice([1, 0], p=[self.photon_chance, 1-self.photon_chance], size=self.N_pulses)
        photon_bin_times = np.random.normal(self.pulse_width/2, self.pulse_width/2, size=np.sum(photon_exists)) # draw random times in a pulse
        photon_times = pulse_times[photon_exists==1] + photon_bin_times    # Add pulse times plus random (poisonian) offset to make temporally spaced clusters
        
        return photon_times
    

class LightGenerator():
    def __init__(self, stream_length, pulse_width, period, dark_counts, brightness, extinction_ratio, pulse_shape = "square") -> None:
        """set internal variables and return the generate function
        
        Inputs:
            stream_length (int): total time of light stream in [s]
            pulse_width (float): width/duration of one block pulse [ns]
            period (float): duration between centers of pulses in [ns]
            dark_counts (float): number of photons per second in the background (uniformly over entire stream)
            brightness (float): average light intensity per second of the final stream (including dark counts!) [counts]
            extinction_ratio (float): the ratio between the pulse and background intensity [-]
            pulse_shape (string): shape of the pulse, such as "square" or "gaussian"
        """
        # All in units of ns
        self.stream_length = stream_length * 1e9    # ns
        self.pulse_width = pulse_width              # ns
        self.period = period                        # ns
        self.dark_counts = dark_counts * 1e-9       # photons / ns
        self.extinction_ratio = extinction_ratio    # -
        self.brightness = brightness * 1e-9         # photons / ns
        self.pulse_shape = pulse_shape              # (string)

        # Derived quantities
        self.duty_cycle = self.pulse_width / self.period
        self.N_pulses = np.floor(self.stream_length / self.period).astype(np.int)
        # Below is derived from 'intensity = darkcounts + pulse_intensity (long time average!!!!) + background'
        self.background_brightness = (self.brightness - self.dark_counts) / (self.extinction_ratio * self.duty_cycle + 1)
        self.pulse_brightness_peak = self.extinction_ratio * self.background_brightness

        # In units of period time
        self.mean_photons_in_pulse = self.pulse_brightness_peak * self.pulse_width
        self.mean_photons_in_background = self.background_brightness * self.period
        self.mean_darkcounts = self.dark_counts * self.period
        assert np.isclose(np.sum([self.mean_darkcounts, self.mean_photons_in_pulse, self.mean_photons_in_background]), self.brightness * self.period)

        self.applychain = []

    def __call__(self):
        stream = self.generate()
        for func in self.applychain:
            stream = func(stream)
        return stream

    def clear(self):
        self.applychain = []

    def apply(self, func):
        self.applychain.append(func)

    def photon_stream(self, stream_length, pulse_width, period, dark_counts, brightness, extinction_ratio, pulse_shape = "square"):
        """Returns an array of arrival times by calling photon stream generator and then calling the result."""
        return self.photon_stream_generator(stream_length, pulse_width, period, dark_counts, brightness, extinction_ratio, pulse_shape)()

    def photon_stream_generator(self):
        return self

    
    def generate(self):
        '''Create vector containing times of photon emissions. Use Poissonian distribution for laser light.
        Create block pulses.
        Probability to emit n photons: Pn = 1/tau_p * exp(-t/tau_p)
        
        Ouputs:
            emitted_times (nparray): 1D array containing photon emission times in units [delta_t]
        '''

        
        # Generate timestamps for pulse photons
        pulse_times = np.arange(0, self.N_pulses, 1) * self.period
        pulse_photons = np.random.poisson(self.mean_photons_in_pulse, size=self.N_pulses)
        if self.pulse_shape == "square":
            photon_bin_times = np.random.uniform(0, self.pulse_width, size=np.sum(pulse_photons)) # draw random times in a pulse
        elif self.pulse_shape == "gaussian":
            photon_bin_times = np.random.normal(0, self.pulse_width/2, size=np.sum(pulse_photons)) # draw random times in a pulse
        else:
            raise NotImplementedError("Only 'square' and 'gaussian' pulse shapes are implemented")
        pulse_times = np.repeat(pulse_times, pulse_photons) + photon_bin_times    # Add pulse times plus random (poisonian) offset to make temporally spaced clusters
        
        # Remember, mean counts are in units of counts per period!
        background_times = np.random.uniform(0, self.stream_length, size=np.random.poisson(self.mean_photons_in_background*self.N_pulses))
        darkcount_times = np.random.uniform(0, self.stream_length, size=np.random.poisson(self.mean_darkcounts*self.N_pulses))

        # sort
        detector_click_times = np.hstack((pulse_times, background_times, darkcount_times))
        detector_click_times = np.sort(detector_click_times)

        return detector_click_times


from numba import njit

@njit
def passdot_jit(stream, lifetime, interaction_probability, extinction_probability):
    qd_deexcite_time = 0
    mask = np.zeros(stream.size, dtype=np.bool_)
    
    for i, photon_time in enumerate(stream):
        if photon_time > qd_deexcite_time:
            if np.random.uniform(0,1) <= interaction_probability:
                # print(qd_deexcite_time,photon_time)
                mask[i] = True
                qd_lifetime = np.random.exponential(lifetime)
                qd_deexcite_time = photon_time + qd_lifetime
        elif np.random.uniform(0,1) >= extinction_probability:
                mask[i] = True
    
    
    return mask

class QuantumDot():
    def __init__(self, lifetime, interaction_probability, extinction_probability):
        self.lifetime = lifetime
        self.interaction_probability=interaction_probability
        self.extinction_probability = extinction_probability # 

    def __call__(self, stream):
        return self.passdot(stream)

    def passdot(self, stream):
        mask = passdot_jit(stream, self.lifetime, self.interaction_probability, self.extinction_probability)
        # print(np.delete(stream, np.argwhere(~mask)).size, stream.size, stream[mask].size, np.sum(mask))
        return stream[mask]
    
    def passdot_generator(self, stream_gen):
        return lambda: self.passdot(stream_gen())
    

class Detector():
    def __init__(self, t_jitter=0, detector_efficiency=1, noise_shape="lorentzian"):
        """
        t_jitter: detector jitter FWHM in [ns]
        detector_efficiency: chance to detect a photon, in range [0,1].
        """
        self.eff = detector_efficiency
        self.noise_shape = noise_shape
        self.t_jitter = t_jitter

    def __call__(self, stream):
        if self.eff < 1.:
            # Efficiency
            detected = np.random.choice([False, True], p=[1-self.eff, self.eff], size=len(stream))
            stream = stream[detected]
        
        # Timing jitter
        if self.noise_shape == "gaussian":
            jitter = np.random.normal(0, self.t_jitter/2.355, size=len(stream))
        elif self.noise_shape == "lorentzian":
            jitter = np.random.standard_cauchy(size=len(stream))*self.t_jitter/2
        else:
            raise NotImplementedError("Only 'lorentzian' and 'gaussian' detector jitter profiles are implemented.")
        
        stream += jitter
        return stream

@njit
def sparse_difference_matrix_idx(stream1, stream2, max_difference):
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
    time_differences = np.empty((stream1.size*max(int(max_difference), 1)*3, 3), dtype=np.float64)
    # But we will not use the whole array, so keep track of where to end
    time_differences_idx = 0

    # Memory for the inner loop
    temp_differences = np.empty((stream2.size, 3))

    # For the inner loop, the last useful time idx of stream1
    j_start = 0

    for i in range(len(stream1)):

        for store_index, j in enumerate(range(j_start, len(stream2))):
            # Calc time differnces
            temp_differences[store_index, ::] = stream2[j]-stream1[i], float(i), float(j)

            # Check if following calculations are useful
            if abs(temp_differences[store_index, 0]) > max_difference: 
                if temp_differences[store_index, 0] < 0:
                    j_start = j+1
                else:
                    break
        # Write to output array
        if time_differences_idx+store_index > time_differences.size:
            print("Time difference overflow")
            return time_differences[:time_differences_idx]
            raise OverflowError("time_differences is too small, think about it you lazy sob")
        time_differences[time_differences_idx:time_differences_idx+store_index, ::] = temp_differences[:store_index, ::]
        # Update index of output array
        time_differences_idx += store_index
    if time_differences_idx >= time_differences.size:
        raise OverflowError("Trying to read higher idx than exists!")

    # Return only the indices we wrote to
    return time_differences[:time_differences_idx, ::]

# @njit
def closest_value(longstream, shortstream):
    # Alocate memory for output
    time_differences = np.empty(shape=longstream.size, dtype=np.float64)

    idx_array = np.empty(shape=longstream.size, dtype=np.int_)

    # For the inner loop, the last useful time idx of stream1
    shortstream_last_useful_idx = 0

    for longstream_idx in range(longstream.size):
        time_differences[longstream_idx] = shortstream[shortstream_last_useful_idx]-longstream[longstream_idx]
        # shortstream_last_useful_idx+=1
        for shortstream_idx in range(shortstream_last_useful_idx, shortstream.size):
            # Calc time differnces
            time_difference = shortstream[shortstream_idx]-longstream[longstream_idx]

            if np.abs(time_difference) > np.abs(time_differences[longstream_idx]):
                # print(time_differences[longstream_idx])
                idx_array[longstream_idx] = shortstream_idx - 1
                break
            else:
                time_differences[longstream_idx] = time_difference

    return time_differences, idx_array

@njit
def unique_pairs(idxs):
    seen1 = np.empty(shape=idxs.shape[0])
    seen2 = np.empty(shape=idxs.shape[0])
    output_idxs = np.empty(shape=idxs.shape[0], dtype=np.int_)
    current_idx = 0

    for i in range(idxs.shape[0]):
        if idxs[i,0] in seen1[:current_idx] or idxs[i,1] in seen2[:current_idx]:
            pass
        else:
            output_idxs[current_idx] = i
            seen1[current_idx] = idxs[i,0]
            seen2[current_idx] = idxs[i,1]
            current_idx+=1

    return output_idxs[:current_idx]

class BeamSplitter():
    def __init__(self, indistinguishability=1, R=0.5, photon_length=50.):
        assert R >=0 and R <= 1.

        self.R = R
        self.T = 1 - R
        self.indistinguishability = indistinguishability
        self.photon_length = photon_length

    def __call__(self, stream1, stream2 = None):
        '''Split the given beam into two by using probablistic chance {chance1}.
        
        Inputs:
            emitted_times (nparray): 1D array containing photon emission times
            chance1 (float): chance (0-1) for photon to go to output/detector 1
            
        Ouputs:
            output1 (nparray): 1D array containing photon emission times of photons at output/detector 1
            output2 (nparray): 1D array containing photon emission times of photons at output/detector 2
        '''

        if type(stream1) == list:
            stream1, stream2 = stream1

        # Generate random directions
        if stream2 is None:
            directions1 = np.random.choice([True, False], p=[self.T,self.R], size=len(stream1))

            # Create two arrays for the two BS outputs
            output1 = stream1[directions1]
            output2 = stream1[np.logical_not(directions1)]
        
        elif self.indistinguishability == 0:
            directions1 = np.random.choice([True, False], p=[self.T,self.R], size=len(stream1))
            directions2 = np.random.choice([True, False], p=[self.R,self.T], size=len(stream2))
            output1_1 = stream1[directions1]
            output2_1 = stream1[~directions1]
            output1_2 = stream2[directions2]
            output2_2 = stream2[~directions2]

            output1 = np.hstack([output1_1, output1_2])
            output2 = np.hstack([output2_1, output2_2])


        else:
            # Now we need to worry about HOM interference
            stream1_hom_idx, stream2_hom_idx, overlap = self.find_unique_coincidences(stream1, stream2, plot=False)

            # With HOM
            # Addition terms in classical and HOM case
            R2_T2_classical = self.R + self.T
            R2_T2_hom = np.abs(self.R - self.T)

            # Using linear calculation interpolate
            R2_T2 = self.indistinguishability * R2_T2_hom + (1-self.indistinguishability) * R2_T2_classical
            
            R2_T2 = (1-overlap/self.photon_length) * R2_T2 + (overlap/self.photon_length) * R2_T2_classical
            
            # Is there HOM interference? With chance RT two photons in one mode, with chance R2_T2 one in each mode.
            hom_interference = np.random.uniform(size=stream1_hom_idx.size) >= R2_T2
            # hom_interference = np.random.choice([True, False], p=[1-R2_T2, R2_T2], size=stream1_hom_idx.size)

            N_hom = hom_interference.sum()  # Number of instances: two photons in one mode
            # Then we pick a side
            HOM_output_1_or_2 = np.random.choice([True, False], p=[0.5,0.5], size=N_hom)

            # The HOM photons in the output are those in the time bin sdm_idx, that experience hom_interference and go to output mode stream1_hom_transmission
            # HOM photons stream 1
            HOM_stream1 = stream1[stream1_hom_idx][hom_interference]
            output1_stream1_hom = HOM_stream1[HOM_output_1_or_2]
            output2_stream1_hom = HOM_stream1[~HOM_output_1_or_2]

            # HOM photons stream 2
            HOM_stream2 = stream2[stream2_hom_idx][hom_interference]
            output1_stream2_hom = HOM_stream2[HOM_output_1_or_2]
            output2_stream2_hom = HOM_stream2[~HOM_output_1_or_2]

            # distinguishable HOM stream 1
            stream1_distinguishable_hom = stream1[stream1_hom_idx][~hom_interference]
            directions_hom1 = np.random.choice([True, False], p=[self.T,self.R], size=len(stream1_distinguishable_hom))
            outstream1_distinguishable_hom1 = stream1_distinguishable_hom[directions_hom1]
            outstream2_distinguishable_hom1 = stream1_distinguishable_hom[~directions_hom1]

            # NO HOM stream 2
            stream2_distinguishable_hom = stream2[stream2_hom_idx][~hom_interference]
            directions_hom2 = np.random.choice([True, False], p=[self.R,self.T], size=len(stream2_distinguishable_hom))
            outstream1_distinguishable_hom2 = stream2_distinguishable_hom[directions_hom2]
            outstream2_distinguishable_hom2 = stream2_distinguishable_hom[~directions_hom2]

            # Sort photons by HOM or not
            stream1_not_hom_mask = np.ones(stream1.size, dtype=np.bool_)
            stream1_not_hom_mask[stream1_hom_idx] = False

            stream2_not_hom_mask = np.ones(stream2.size, dtype=np.bool_)
            stream2_not_hom_mask[stream2_hom_idx] = False

            # No HOM stream 1
            stream1_not_hom = stream1[stream1_not_hom_mask]
            directions1 = np.random.choice([True, False], p=[self.T,self.R], size=len(stream1_not_hom))
            output1_stream1_no_hom = stream1_not_hom[directions1]
            output2_stream1_no_hom = stream1_not_hom[~directions1]
            # assert stream1_not_hom_mask.sum() + N_hom == stream1.size

            # No HOM stream 2
            stream2_not_hom = stream2[stream2_not_hom_mask]
            directions2 = np.random.choice([True, False], p=[self.R,self.T], size=len(stream2_not_hom))
            output1_stream2_no_hom = stream2_not_hom[directions2]
            output2_stream2_no_hom = stream2_not_hom[~directions2]
            # assert stream2_not_hom_mask.sum() + N_hom == stream2.size

            # Merge output
            output1 = np.hstack([
                output1_stream1_hom, 
                output1_stream2_hom, 
                output1_stream1_no_hom, 
                output1_stream2_no_hom, 
                outstream1_distinguishable_hom1, 
                outstream1_distinguishable_hom2
                ])
            output2 = np.hstack([
                output2_stream1_hom,
                output2_stream2_hom,
                output2_stream1_no_hom,
                output2_stream2_no_hom, 
                outstream2_distinguishable_hom1, 
                outstream2_distinguishable_hom2
                ])
            output1 = np.sort(output1)
            output2 = np.sort(output2)

        return [output1, output2]

    def find_unique_coincidences(self, stream1, stream2, plot=False):
        out = sparse_difference_matrix_idx(stream1, stream2, self.photon_length)
        tau, idx1, idx2 = out[::,0], out[::,1].astype(np.int_), out[::,2].astype(np.int_)

        # Mask off photons outside the photon length
        tau_mask = np.abs(tau) <= self.photon_length
        tau = tau[tau_mask]
        idxs = np.column_stack([idx1,idx2])
        idxs = idxs[tau_mask, ::]

        output_idxs = unique_pairs(idxs)
        
        return idxs[output_idxs,0], idxs[output_idxs,1], tau[output_idxs]

class DeadTime():
    def __init__(self, t_dead):
        self.t_dead=t_dead
    
    def __call__(self, stream):
        '''Apply dead time by removing undetected photons from the stream.
        
        Inputs:
            stream (nparray): 1D array containing photon emission times in units [delta_t]
            t_dead (float): dead time of the detector in units [delta_t]

        Ouputs:
            stream (nparray): 1D array containing photon emission times in units [delta_t]
            '''
        # # Remove doubles
        # stream = np.unique(stream)
        
        # Calculate differences
        diffs = np.diff(stream)
        too_small = diffs <= self.t_dead
        
        # Remove undetectable photons untill all differences are large enough
        while np.sum(too_small) > 0:
            first_trues = np.logical_and(np.diff(too_small), np.logical_not(too_small)[:-1])
            undetected = np.append([False,too_small[0]], first_trues)
            stream = stream[np.logical_not(undetected)]
            
            diffs = np.diff(stream)
            too_small = diffs <= self.t_dead
        
        # print("\n", stream.size/5e-1)
        return stream
    

class Delay():
    def __init__(self, delay):
        self.delay=delay
    
    def __call__(self, stream):
        '''Apply dead time by removing undetected photons from the stream.
        
        Inputs:
            stream (nparray): 1D array containing photon emission times in units [delta_t]
            t_dead (float): dead time of the detector in units [delta_t]

        Ouputs:
            stream (nparray): 1D array containing photon emission times in units [delta_t]
            '''
        stream += self.delay
        return stream
    

def multi_stream_wrapper(func):
    def inner(streams):
        if type(streams) is list:
            out = []
            for stream in streams:
                out.append(func(stream))
            return out
        else:
            return func(streams)
    return inner

