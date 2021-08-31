### what to submit?
A folder called control_model that contains your control model. This folder should contain a ```__init__.py```, a ```controller.py ```.
The ```controller.py``` should contain a function called ```get_input```, which takes ```state``` and returns ```action```. Please consult the template ```controller.py``` for detailed information.

### what result you can expect?
The returned results will contain a ```results.json```, in which ```avg_efactors``` shows the average efector of your control model (the smaller the better), and ```EMPC_avg_efactors``` shows the average efector of EMPC (baseline).
Also, it will contain 200 images; for k in range(100), efactor_fig_{k}.png contains the change of efactor through time using your controller (Real Output), the change of efactor with EMPC through time (EMPC Output), and the steady state efactor (Steady State Output); input_fig_{k}.png contains the change of 4 inputs through time using your controller (Real Output), with EMPC through time (EMPC Output), and the steady state inputs (Steady State Output).
