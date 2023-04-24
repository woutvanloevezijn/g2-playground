import matplotlib.pyplot as plt
import numpy as np

def plotstream(lg, divider=None):
    stream = lg()
    if type(stream) is list:
        #stream = np.hstack([s for s in stream])
        streams = stream
        for stream in streams:
            times = np.mod(stream, lg.period)
            bins = np.arange(0, lg.period, lg.pulse_width/5)
            counts, bins = np.histogram(times, bins=bins)
            plt.plot(bins[:-1], counts, '.')
            plt.plot([bins[0], bins[-2]], [divider,divider])
    else:
        times = np.mod(stream, lg.period)
        bins = np.arange(0, lg.period, lg.pulse_width/5)
        counts, bins = np.histogram(times, bins=bins)
        plt.plot(bins[:-1], counts, '.')
        plt.plot([bins[0], bins[-2]], [divider,divider])
    plt.yscale('log')
    # plt.xscale("log")
    plt.xlabel("Time $t$ [ns]")
    plt.ylabel("Counts [-]")
    plt.title("Histogram of photon stream mod period")
    plt.show()

def arrival_distribution(lg):
    stream = lg()
    if type(stream) is list:
        #stream = np.hstack([s for s in stream])
        streams = stream
        for stream in streams:
            dstream = np.diff(stream)
            plt.hist(dstream, bins=np.arange(0, lg.period, lg.pulse_width/5))

    else:
        dstream = np.diff(stream)
        plt.hist(dstream, bins=np.arange(0, lg.period, lg.pulse_width/5))
    plt.yscale('log')
    plt.xlabel("Time difference $\\tau$ [ns]")
    plt.ylabel("Counts [-]")
    plt.title("Histogram of photon stream arrival time difference")
    plt.show()

def statistics_test(lg):
    stream = lg()
    if type(stream) is list:
        stream = np.hstack([s for s in stream])

    times = np.mod(stream, lg.period)
    bins = np.arange(0, lg.period, lg.pulse_width/5)
    counts, bins = np.histogram(times, bins=bins)

    print("Theory")
    print("N_pulse: ", lg.mean_photons_in_pulse*lg.N_pulses, "\t N_bg: ", lg.mean_photons_in_background*lg.N_pulses, "\t PER: ", lg.mean_photons_in_pulse/lg.mean_photons_in_background)

    print("Time dividing")
    N_pulse = np.sum(counts[bins[:-1]<=lg.pulse_width])
    N_background = np.sum(counts[bins[:-1]>lg.pulse_width])
    print("N_pulse: ", N_pulse/lg.N_pulses, "\t N_bg: ", N_background/lg.N_pulses, "\t PER: ", N_pulse/N_background)
    
    divider = (lg.mean_photons_in_background + lg.mean_darkcounts)*lg.N_pulses / counts.size #counts.mean()

    print("Integration")
    # Method 1: integration
    N_background = divider * bins.size 
    N_pulse = np.sum(counts) - N_background
    print(" N_pulse: ", N_pulse/lg.N_pulses, "\t N_bg: ", N_background/lg.N_pulses, "\t PER: ", N_pulse/N_background)

    # plotstream(lg, divider)

    print("Direct counting")
    # Method 2: direct counting
    divider += 4*np.sqrt(divider) # add 4 std -> 99.9% is within this bound
    N_pulse = np.sum(counts[counts > divider]-divider)
    N_background = np.sum(counts[counts < divider]-divider) + divider*counts.size
    print(" N_pulse: ", N_pulse/lg.N_pulses, "\t N_bg: ", N_background/lg.N_pulses, "\t PER: ", N_pulse/N_background)
