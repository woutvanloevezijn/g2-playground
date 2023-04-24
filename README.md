# g2 Playground
Using many optical components, build your virtual setup and simulate what g2 looks like. This simulation is classical, but the `BeamSplitter` element supports photon bunching. So you can make HOM, but not a delay loop.

The three examples should give you an idea of the versitility of this code. All files start by initializing the `LightGenerator`. To this function many elements can be applied in order, all of these elements are also initiallized.
- `QuantumDot` simulates a cross polarized quantum dot
- `Detector` simulates loss and jitter
- `BeamSplitter` simultes a beamsplitter with single pass HOM interference
- `DeadTime` simultes detector deadtime
- `Delay` simulates delay in propegation


## `parameter_self_consistency.py`
Creates a pulsed stream of light with a photon ratio, then calculates back from the simulated g2 what the photon ratio was. This is mostly for model and calculation verification.

## `pulsed_HBT.py` 
Hanburry-Brown-Twiss experiment with pulsed light. A quantum dot can be inserted to change the statistics. 

## ` HOM.py`
Hong-Ou-Mandel experiment with the pulser. A quantum dot can be inserted to change the statistics. Please not that the beamsplitter is not modeled using a quantum calculation, the semi-classical model assumes that all photons are indistinguishable. Continuous wave light leads to a small dip. However only the temporally nearest photons are bunched by the beamsplitter. Multiple passes do not work, you cannot build a delay loop.