# Atropine Control Challenge

This repository contains two parts: [Environment Evaluation](./environment_evaluation) and [Sample Controller](./sample_mlp_control). The Environment Evaluation contains the [Atropine simulation environment](./environment_evaluation/quarticgym/envs/atropineenv.py) itself, along with the [evaluation script](./environment_evaluation/evaluating.py). The Sample Controller contains a [script](./sample_mlp_control/atropine_MLP/mlp_training.py) to train the controller.

## Result Reproduction

To reproduce the results, we need to train the controller algorithm, put the controller weights and controller files in the correct directory, and run the evaluation script.

### Controller Training

Run [mlp_training.py](./sample_mlp_control/atropine_MLP/mlp_training.py) at the file directory. This will train a simple multi-layer-perceptron (MLP) with a set of [offline data](#offline-dataset).

### Controller Submitting

Once the training is finished, we pick the best weights in our torchl_logs folder, rename it to best.pt and put it into the [control_model](./environment_evaluation/control_model) folder. Moreover, you should put a [controller.py](./environment_evaluation/control_model/controller.py) of your controller into the same folder.
This [controller.py](./environment_evaluation/control_model/controller.py) should contain a function called ```get_input```, which takes the  ```state``` and returns the ```action```. For the sample MLP controller, [controller.py](./environment_evaluation/control_model/controller.py) is already given.

### Evaluating

Run [evaluating.py](./environment_evaluation/evaluating.py) at the file directory. The results generated by [evaluating.py](./environment_evaluation/evaluating.py) will contain a [results.json](./environment_evaluation/results/results.json), in which ```avg_efactors``` shows the average efactor of your control model (the smaller the better), and ```EMPC_avg_efactors``` shows the average efactor of the baseline [MPC controller](https://en.wikipedia.org/wiki/Model_predictive_control).
Also, it will contain 200 images; for k in range(100), efactor_fig_{k}.png contains the change of efactor through time using your controller (New Controller Real Output), the change of efactor with MPC through time (MPC Output), and the steady state efactors for them (Steady State Output); input_fig_{k}.png contains the change of 4 inputs through time using your controller (New Controller Real Input), using MPC (MPC Input), and the steady state inputs for them (Steady State Input).

### Offline Dataset

For the data collection part, we let a baseline [MPC controller](https://en.wikipedia.org/wiki/Model_predictive_control) interact with the environment with random initialization 1000 times, for 600 minutes per time. We here by batchify the data collection process in order to control the convergence time and form into a fair comparison with the offline reinforcement learning algorithms. Also since the training should follow the "offline" setting, the MLP model is not allowed to interact with the original environment. You can find these 1000 episodes of training data at [sample_mlp_control/public_dat](./sample_mlp_control/public_dat). [Our blog post](https://quartic.blog/2021/09/11/optimizing-continuous-manufacturing-processes/) contains detailed information about the challenge itself and the whole manufacturing process.
In the csv, ["USS1", "USS2", "USS3", "USS4", "ESS", "E", "KF_X1", "KF_X2", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7", "Z8", "Z9", "Z10", "Z11", "Z12", "Z13", "Z14", "Z15", "Z16", "Z17", "Z18", "Z19", "Z20", "Z21", "Z22", "Z23", "Z24", "Z25", "Z26", "Z27", "Z28", "Z29", "Z30"] are generally considered as state features. To be more specific, USS%d are MPC Steady Inputs, KF_X%d are the ouput of the [kalman filter](https://en.wikipedia.org/wiki/Kalman_filter) used by our MPC, Z%d are the flow rates of different components as algebraic states, ESS is the MPC steady state efactor Output, E is the MPC efactor output, and ["U1", "U2", "U3", "U4"] are actions taken.
