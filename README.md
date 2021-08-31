### what is in the training dataset?
atropine.csv contains 1000 episodes of training data collected by EMPC. https://quarticai.atlassian.net/wiki/spaces/AI/pages/2439086127/MPC+on+Continuous+Atropine+Manufacturing+Process contains detailed information about the challenge itself and the whole manufacturing process.
In the csv, ["USS1", "USS2", "USS3", "USS4", "ESS", "E", "KF_X1", "KF_X2", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7", "Z8", "Z9", "Z10", "Z11", "Z12", "Z13", "Z14", "Z15", "Z16", "Z17", "Z18", "Z19", "Z20", "Z21", "Z22", "Z23", "Z24", "Z25", "Z26", "Z27", "Z28", "Z29", "Z30"] are generally considered as state features, and ["U1", "U2", "U3", "U4"] are actions taken.

### what to submit?
A folder called control_model that contains your control model. This folder should contain a ```__init__.py```, a ```controller.py ```.
The ```controller.py``` should contain a function called ```get_input```, which takes ```state``` and returns ```action```. Please consult the template ```controller.py``` for detailed information.

### what result you can expect?
The returned results will contain a ```results.json```, in which ```avg_efactors``` shows the average efector of your control model (the smaller the better), and ```EMPC_avg_efactors``` shows the average efector of EMPC (baseline).
Also, it will contain 200 images; for k in range(100), efactor_fig_{k}.png contains the change of efactor through time using your controller (Real Output), the change of efactor with EMPC through time (EMPC Output), and the steady state efactor (Steady State Output); input_fig_{k}.png contains the change of 4 inputs through time using your controller (Real Output), with EMPC through time (EMPC Output), and the steady state inputs (Steady State Output).
