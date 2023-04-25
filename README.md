# $\text{g}^2$ Playground
Using many optical components, build your virtual setup and simulate what g2 looks like. This simulation is classical, but the `BeamSplitter` element supports photon bunching up to two photons. So you can make HOM, but not a delay loop.

The three examples should give you an idea of the versitility of this code. All files start by initializing the `LightGenerator`. To this function many elements can be applied in order, all of these elements are also initiallized.
- `QuantumDot` simulates a cross polarized quantum dot
- `Detector` simulates loss and jitter
- `BeamSplitter` simultes a beamsplitter with single pass HOM interference
- `DeadTime` simultes detector deadtime
- `Delay` simulates delay in propegation

# Examples
To get you going with this code, a few examples are included that demonstrate the main features:

## `parameter_self_consistency.py`
Creates a pulsed stream of light with some photon ratio set by the user, then calculates back from the simulated g2 what the photon ratio was. This is mostly for model and calculation verification.

## `pulsed_HBT.py` 
Hanbury-Brown-Twiss experiment with pulsed light. A quantum dot can be inserted to change the statistics, extinguishing the peak at $\tau=0.$

## ` HOM.py`
This is a work in progress, currently the $\text{g}^2$ of a quantum dot HOM measurement is not accurately repreduced.
Hong-Ou-Mandel experiment with the pulser. A quantum dot can be inserted to change the statistics. Please not that the beamsplitter is not modeled using a quantum calculation, the semi-classical model assumes that all photons are indistinguishable. Continuous wave light leads to a small dip. However only the temporally nearest photons are bunched by the beamsplitter. Multiple passes do not work, you cannot build a delay loop.

# Tips for use
- Too large values of `stream_length` leads to huge arrays: expect memory issues, and long calculation times for the $\text{g}^2(\tau)$ correlations. Better repeat the experiment more times and build statistics.
- The function that returns $\text{g}^2$ uses a sparse autocorrelation algorithm by default (`method='direct'`). Using a FFT-based algorithm is also supported, by setting `method='fourier'`. This is slower in most cases.
- Set `LigthGenerator.extinction_ratio=0` for a continous wave poisonian distributed light source.
- This simulation does not simulate photons, it simulates detector _click_ times. This includes dark counts!

# Known issues
- HOM model does not reproduce 3/4 sidepeaks, instead there is behaviour of side peaks which is not understood.
- Dark counts are generated at `LightStream`. This means they experience loss and are incident on the quantum dot. Their generation should be moved to the detector.
- It would be nice if each optical element can become a subclass of `LightStream`, such that its properties are accesible along the optical chain. The `LightGenerator.apply()` method sucks and hurts my soul, but is used to circumvent this problem. The use of `BeamSplitter` really complicates how to implement the optics in a beautiful way.

# Credits
This work leans on previous work by Ilse Kuijf, the core idea and some functions were developed by Ilse. An overhaul was done by Mio Poortvliet, adding flexibility and the optical components and quantum dot. Mio currently maintains the project.
