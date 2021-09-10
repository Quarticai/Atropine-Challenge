from quarticgym.envs.atropineenv import AtropineEnvGym
import pickle
import json
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import yaml

from control_model.controller import controller


def test_on_env(max_steps, ctrl, df, x0, z0, iter, normalize):
    try:
        with open('control_model/current.yaml', 'r') as fp:
            config_dict = yaml.safe_load(fp)
        normalize = ctrl.normalize
        include_t = config_dict['include_t']
        uss_subtracted = config_dict['uss_subtracted']
        reward_on_ess_subtracted = config_dict['reward_on_ess_subtracted']
        reward_on_steady = config_dict['reward_on_steady']
        reward_on_absolute_efactor = config_dict['reward_on_absolute_efactor']
        reward_on_actions_penalty = config_dict['reward_on_actions_penalty']
        reward_on_reject_actions = False
        relaxed_max_min_actions = config_dict['relaxed_max_min_actions']
        env = AtropineEnvGym(max_steps=max_steps, x0_loc=x0, z0_loc=z0, model_loc='model.npy', normalize=normalize, observation_include_t=include_t, uss_subtracted=uss_subtracted, reward_on_ess_subtracted=reward_on_ess_subtracted, reward_on_steady=reward_on_steady, reward_on_absolute_efactor=reward_on_absolute_efactor, reward_on_actions_penalty=reward_on_actions_penalty, reward_on_reject_actions=reward_on_reject_actions, relaxed_max_min_actions=relaxed_max_min_actions)
    except Exception:
        print("env init without control_model/current.yaml. init a default env.")
        env = AtropineEnvGym(x0_loc=x0, z0_loc=z0, model_loc='model.npy', normalize=normalize)
    state = env.reset()
    efactors = []
    actions = []
    rewards = []
    for step in range(1, max_steps+1):
        action = ctrl.get_input(state)
        state, reward, done, info = env.step(action)
        efactor = info['efactor']
        efactors.append(efactor)
        actions.append(action)
        rewards.append(reward)
        if done:
            break

    # plot
    try:
        U = np.array(env.U) * 1000  # scale the solution to micro Litres
        local_t = [k * 10 for k in range(step)]
        plt.close("all")
        plt.figure(0)
        plt.plot(local_t, env.Y, label='New Controller Output')
        plt.plot(local_t, [env.Y[-1] for k in range(step)], linestyle="--", label='New Controller Steady State Output')
        plt.plot(local_t, df.E, label='MPC Output')
        plt.plot(local_t, df.ESS, linestyle="--", label='MPC Steady State Output')
        plt.xlabel('Time [min]')
        plt.ylabel('E-Factor [kg/kg]')
        plt.legend(loc="lower left", ncol=2, bbox_to_anchor=(-0.15, 1.02, 1, 0.2), borderaxespad=1)
        plt.subplots_adjust(top=0.75)
        plt.savefig(f"results/efactor_fig_{iter}")
        plt.close("all")

        fig, axs = plt.subplots(nrows=2, ncols=2)
        
        axs[0, 0].step(local_t, U[:, 0], where='post', label='New Controller Input')
        axs[0, 0].plot(local_t, [U[:, 0][-1] for k in range(step)], linestyle="--", label='New Controller Steady State Input')
        axs[0, 0].plot(local_t, df.U1 * 1000, label='MPC Input')
        axs[0, 0].plot(local_t, df.USS1 * 1000, linestyle="--", label='MPC Steady State Input')
        axs[0, 0].set_ylabel(u'$U_1$ [\u03bcL/min]')

        axs[0, 1].step(local_t, U[:, 1], where='post', label='New Controller Input')
        axs[0, 1].plot(local_t, [U[:, 1][-1] for k in range(step)], linestyle="--", label='New Controller Steady State Input')
        axs[0, 1].plot(local_t, df.U2 * 1000, label='MPC Input')
        axs[0, 1].plot(local_t, df.USS2 * 1000, linestyle="--", label='MPC Steady State Input')
        axs[0, 1].set_ylabel(u'$U_2$ [\u03bcL/min]')

        axs[1, 0].step(local_t, U[:, 2], where='post', label='New Controller Input')
        axs[1, 0].plot(local_t, [U[:, 2][-1] for k in range(step)], linestyle="--", label='New Controller Steady State Input')
        axs[1, 0].plot(local_t, df.U3 * 1000, label='MPC Input')
        axs[1, 0].plot(local_t, df.USS3 * 1000, linestyle="--", label='MPC Steady State Input')
        axs[1, 0].set_ylabel(u'$U_3$ [\u03bcL/min]')
        axs[1, 0].set_xlabel('Time [min]')

        axs[1, 1].step(local_t, U[:, 3], where='post', label='New Controller Input')
        axs[1, 1].plot(local_t, [U[:, 3][-1] for k in range(step)], linestyle="--", label='New Controller Steady State Input')
        axs[1, 1].plot(local_t, df.U4 * 1000, label='MPC Input')
        axs[1, 1].plot(local_t, df.USS4 * 1000, linestyle="--", label='MPC Steady State Input')
        axs[1, 1].set_ylabel(u'$U_4$ [\u03bcL/min]')
        axs[1, 1].set_xlabel('Time [min]')
        plt.tight_layout()
        axs[0, 0].legend(loc="lower left", ncol=2, bbox_to_anchor=(-0.45, 1.02, 3, 2), borderaxespad=1, mode='expand')
        plt.subplots_adjust(top=0.75)

        plt.savefig(f"results/input_fig_{iter}")
        plt.close()
    except Exception:
        print(f"iteration {iter} is not finished, with {step} steps.")

    return efactors, actions, rewards 


if __name__ == '__main__':
    with open('x0sz0s.pkl', 'rb') as handle:
        x0s, z0s = pickle.load(handle)
    assert len(x0s) == len(z0s)
    num_iter = len(x0s)
    max_steps = 60
    ctrl = controller()
    normalize = ctrl.normalize
    avg_efactors = 0.0
    EMPC_avg_efactors = 0.0
    df = pd.read_csv("atropine_test_100.csv")
    E = df[["E"]].fillna(0.).to_numpy()

    for iter in tqdm(range(num_iter)):
        df = pd.read_csv(f"evaluating_dat/{iter}.csv")
        efactors, actions, rewards = test_on_env(max_steps, ctrl, df, x0s[iter], z0s[iter], iter, normalize)
        avg_efactors += efactors[-1]
        EMPC_avg_efactors += E[60*(iter+1)-1]

    avg_efactors /= num_iter
    EMPC_avg_efactors /= num_iter

    results = {"avg_efactors":avg_efactors, "EMPC_avg_efactors": EMPC_avg_efactors[0]}
    with open('results/results.json', 'w') as f:
        json.dump(results, f)
    shutil.make_archive("results", 'zip', 'results/')
    print(f"The average efactor of {num_iter} runs are: {avg_efactors}.")
    print(f"The average efactor of EMPC is: {EMPC_avg_efactors}.")
