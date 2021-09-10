import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import mzutils
from mzutils import normalize_spaces, denormalize_spaces, list_of_str_to_numpy_onehot_dict
from zipfile import ZipFile
from gym import spaces, Env
import d3rlpy
import wandb
import collections
import pickle
import codecs
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


def combine_str_to_numpy_onehot_dict(lst_of_lists):
    # lst_of_lists should not contain key with the same value
    re_dict = {}
    key_dicts = {}
    for lst in lst_of_lists:
        key_dicts.update(list_of_str_to_numpy_onehot_dict(lst))
    for i in lst_of_lists[0]:
        for j in lst_of_lists[1]:
            for k in lst_of_lists[2]:
                re_dict[i+'-'+j+'-'+k] = {'one_hot': np.concatenate((key_dicts[i], key_dicts[j], key_dicts[k]))}
    additional_length = len(re_dict[next(iter(re_dict))]['one_hot'])
    return re_dict, additional_length



class SeedData:
    """
    A dictionary that aims to average the evaluated mean_episode_return accross different random seed.
    Also controls where to resume the experiments from.
    """

    def __init__(self, save_path, resume_from={}):
        self.seed_data = pd.DataFrame({
            'algo_name': pd.Series([], dtype='str'),
            'test_reward': pd.Series([], dtype='float'),
            'seed': pd.Series([], dtype='int'),
        })
        self.save_path = save_path
        mzutils.mkdir_p(save_path)
        self.load()
        # set experiment range
        self.resume_from = resume_from
        self.resume_check_passed = False

    def load(self):
        re_list = mzutils.get_things_in_loc(self.save_path)
        if not re_list:
            print("Cannot find the a seed_data.csv at", self.save_path, "initializing a new one.")
            self.seed_data.to_csv(os.path.join(self.save_path, 'seed_data.csv'), index=False)
        else:
            self.seed_data = pd.read_csv(os.path.join(self.save_path, 'seed_data.csv'))
            print("Loaded the seed_data.csv at", self.save_path)
    
    def save(self):
        self.seed_data.to_csv(os.path.join(self.save_path, 'seed_data.csv'), index=False)

    def append(self, algo_name, test_reward, seed):
        self.seed_data.loc[len(self.seed_data)] = [algo_name, test_reward, seed]

    def setter(self, algo_name, test_reward, seed):
        # average over seed makes seed==-1
        # online makes dataset_percent==0.0
        self.append(algo_name, test_reward, seed)
        averaged_reward = self.seed_data.loc[(self.seed_data['algo_name'] == algo_name)]['test_reward'].mean()
        if seed == seeds[-1]: # append the average, seed now set to -1
            self.seed_data.loc[len(self.seed_data)] = [algo_name, averaged_reward, -1]
        self.save()
        return averaged_reward

    def resume_checker(self, current_positions):
        """
        current_positions has the same shape as self.resume_from
        return True if the current loop still need to be skipped.
        """
        if self.resume_check_passed is True: # checker has already passed.
            return True

        if not self.resume_from:
            self.resume_check_passed = True
        elif all(
            self.resume_from[condition] is None for condition in self.resume_from
        ):
            self.resume_check_passed = True
        else:
            self.resume_check_passed = all(
                self.resume_from[condition] == current_positions[condition]
                for condition in self.resume_from
            )

        return self.resume_check_passed


class DataObj:
    def __init__(self, dataset_folder='public_dat', eval_size=0.1, shuffle=True, normalize=True, include_t=True, uss_subtracted=False, reward_on_ess_subtracted=False, reward_on_steady=False, reward_on_absolute_efactor=False, reward_on_actions_penalty=0.0) -> None:
        """
        if include_onehot_sys_encoding, we also append one-hot encoding of system encoding
        """
        self.dataset_folder = dataset_folder
        self.eval_size = eval_size
        self.shuffle = shuffle
        self.normalize = normalize
        self.include_t = include_t # check for MDP
        self.uss_subtracted = uss_subtracted # we assume that we can see the steady state output during steps. If true, we plus the actions with USS during steps.
        self.reward_on_ess_subtracted = reward_on_ess_subtracted
        self.reward_on_steady = reward_on_steady
        self.reward_on_absolute_efactor = reward_on_absolute_efactor
        self.reward_on_actions_penalty = reward_on_actions_penalty
        self.max_observations = None
        self.min_observations = None
        self.max_actions = None
        self.min_actions = None
        self.clamping_max_actions = None
        self.clamping_min_actions = None
        self.observation_dim = 38 # 4+1+1+2+30
        if self.include_t:
            self.observation_dim += 1
        self.action_dim = 4 #(take max, when not provided set to zero)
        self.observation_skip_columns = None
        if self.include_t:
            self.observation_skip_columns = [0]
        self.file_list = mzutils.get_things_in_loc(dataset_folder, just_files=True)
        print(f"Data found. Contains {len(self.file_list)} files.")

    def __len__(self):
        """
        how many episodes?
        """
        return self.file_list
    
    # def read_from_csv(self, file_path='public_dat/0.csv'):
    #     df = pd.read_csv(file_path)
    #     return self.read_from_df(df)
    
    def evalute_based_on_actions(self, generated_action, target_action):
        """
        generated_action and target_action are two vectors
        """
        return np.linalg.norm((generated_action-target_action), ord=2)**2

    def compute_reward(self, df):
        efactor = df[["E"]].fillna(0.).to_numpy()
        actions = df[["U1", "U2", "U3", "U4"]].fillna(0.).to_numpy()
        if self.reward_on_ess_subtracted:
            ess = df[["ESS"]].fillna(0.).to_numpy()
            reward = ess - efactor
        if self.reward_on_steady:
            ess = df[["ESS"]].fillna(0.).to_numpy()
            reward = -abs(efactor - ess)
        elif self.reward_on_absolute_efactor:
            reward = -abs(efactor)
        else:
            previous_efactor = np.concatenate([efactor[0].reshape((1,1)), efactor[:-1]])
            reward = previous_efactor - efactor
        action_norm_penalty = np.linalg.norm(actions*self.reward_on_actions_penalty, ord=2, axis=1)
        action_norm_penalty = action_norm_penalty.reshape((action_norm_penalty.shape[0], 1))
        reward += action_norm_penalty
        return reward 

    def file_list_to_dict(self, file_list):
        file_list = file_list.copy()
        if self.shuffle:
            random.shuffle(file_list)
        dataset= {}
        observations = []
        actions = []
        next_observations = []
        rewards = []
        terminals = []
        for file_path in tqdm(file_list):
            df = pd.read_csv(file_path)
            tmp_observations = df[["Unnamed: 0","USS1","USS2","USS3","USS4","ESS","E","KF_X1","KF_X2","Z1","Z2","Z3","Z4","Z5","Z6","Z7","Z8","Z9","Z10","Z11","Z12","Z13","Z14","Z15","Z16","Z17","Z18","Z19","Z20","Z21","Z22","Z23","Z24","Z25","Z26","Z27","Z28","Z29","Z30"]].fillna(0.).to_numpy()
            tmp_actions = df[['U1', 'U2', 'U3', 'U4']].fillna(0.).to_numpy()
            tmp_next_observations = np.append(tmp_observations[1:], tmp_observations[-1].reshape((1, self.observation_dim)), 0)
            tmp_rewards = self.compute_reward(df)
            tmp_terminals = np.zeros(tmp_rewards.shape[0], dtype=bool)
            tmp_terminals[-1] = True
            observations.append(tmp_observations)
            if self.uss_subtracted:
                tmp_actions = tmp_actions - df[["USS1","USS2","USS3","USS4"]].fillna(0.).to_numpy()
            actions.append(tmp_actions)
            next_observations.append(tmp_next_observations)
            rewards.append(tmp_rewards)
            terminals.append(tmp_terminals)
        observations = np.ma.concatenate(observations, axis=0)
        actions = np.ma.concatenate(actions, axis=0)
        next_observations = np.ma.concatenate(next_observations, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        terminals = np.concatenate(terminals, axis=0)
        dataset['observations'] = np.ma.array(observations, dtype=np.float32)
        dataset['actions'] = np.ma.array(actions, dtype=np.float32)
        dataset['next_observations'] = np.ma.array(next_observations, dtype=np.float32)
        dataset['rewards'] = np.array(rewards, dtype=np.float32)
        dataset['terminals'] = np.array(terminals, dtype=bool)
        # normally wont need this part, since all max and min has been set.
        tmp_ext = dataset['observations'].max(axis=0)
        if self.max_observations is not None:
            self.max_observations = np.max((self.max_observations, tmp_ext), axis=0)
        else:
            self.max_observations = tmp_ext

        tmp_ext = dataset['observations'].min(axis=0)
        if self.min_observations is not None:
            self.min_observations = np.min((self.min_observations, tmp_ext), axis=0)
        else:
            self.min_observations = tmp_ext

        tmp_ext = dataset['actions'].max(axis=0)
        if self.max_actions is not None:
            self.max_actions = np.max((self.max_actions, tmp_ext), axis=0)
        else:
            self.max_actions = tmp_ext

        tmp_ext = dataset['actions'].min(axis=0)
        if self.min_actions is not None:
            self.min_actions = np.min((self.min_actions, tmp_ext), axis=0)
        else:
            self.min_actions = tmp_ext

        if self.normalize:
            bn = normalize_spaces(dataset['observations'], self.max_observations, self.min_observations, skip_columns=self.observation_skip_columns)
            dataset['observations'] = bn[0]
            bn = normalize_spaces(dataset['next_observations'], self.max_observations, self.min_observations, skip_columns=self.observation_skip_columns)
            dataset['next_observations'] = bn[0]
            bn = normalize_spaces(dataset['actions'], self.max_actions, self.min_actions)
            dataset['actions'] = bn[0]
        else:
            dataset['observations'] = np.array(dataset['observations']) # remove masks
            dataset['next_observations'] = np.array(dataset['next_observations']) # remove masks
            dataset['actions'] = np.array(dataset['actions']) # remove masks
        tmp_ext = dataset['actions'].max(axis=0)
        if self.clamping_max_actions is not None:
            self.clamping_max_actions = np.max((self.clamping_max_actions, tmp_ext), axis=0)
        else:
            self.clamping_max_actions = tmp_ext
        tmp_ext = dataset['actions'].min(axis=0)
        if self.clamping_min_actions is not None:
            self.clamping_min_actions = np.min((self.clamping_min_actions, tmp_ext), axis=0)
        else:
            self.clamping_min_actions = tmp_ext
        return dataset
    
    # def load_file_list_to_dict(self, max_min_dir):
    #     total_dict = self.file_list_to_dict(self.file_list) # to set all max and min
    #     max_min_dict = {'max_observations': self.max_observations, 'min_observations': self.min_observations, 'max_actions': self.max_actions, 'min_actions': self.min_actions}
    #     with open(max_min_dir, 'wb') as fb:
    #         pickle.dump(max_min_dict, fb)
    #     if not self.leave_one_out_for_eval:
    #         self.training_list = self.file_list
    #         self.eval_list = []
    #         return total_dict
    #     else:
    #         if self.leave_one_out_num is None:
    #             leave_one_out_num = str(random.randint(0,19))
    #         else:
    #             leave_one_out_num = str(self.leave_one_out_num)
    #         if len(leave_one_out_num) == 1:
    #             leave_one_out_num = '0'+leave_one_out_num
    #         training_list = []
    #         eval_list = []
    #         for file_path in self.file_list:
    #             if file_path[-6:-4] == leave_one_out_num: #'.csv' should be the last 4 chars
    #                 eval_list.append(file_path)
    #             else:
    #                 training_list.append(file_path)
    #         self.training_list = training_list
    #         self.eval_list = eval_list
    #         return self.file_list_to_dict(training_list), self.file_list_to_dict(eval_list) # two set returned

    def get_dataset(self, random_state=42):
        self.file_list_to_dict(self.file_list) # set max and min. necessary.
        X_train, X_eval, y_train, y_test = train_test_split(self.file_list, self.file_list, test_size=self.eval_size, random_state=random_state)
        return self.file_list_to_dict(X_train), self.file_list_to_dict(X_eval)


class MDPTorchDataset(Dataset):
    def __init__(self, dataset_d4rl) -> None:
        # super().__init__()
        """
        dataset_d4rl should be returned by lbd_data_obj.get_dataset()
        """
        self.dataset = d3rlpy.dataset.MDPDataset(dataset_d4rl['observations'], dataset_d4rl['actions'], dataset_d4rl['rewards'], dataset_d4rl['terminals'])
        
    def __len__(self) -> int:
        return self.dataset.__len__()

    def __getitem__(self, idx):
        episode =self.dataset.__getitem__(idx)
        return {'observations': episode.observations, 'actions': episode.actions, 'rewards': episode.rewards}

    # def __iter__(self):
    #     return self.dataset.__iter__()


def average_result_csvs(csvs_loc, submission_file_name):
    file_list = mzutils.get_things_in_loc(csvs_loc)
    csv_list = [file_path for file_path in file_list if file_path[-4:] == '.csv']
    if not csv_list:
        raise Exception(csvs_loc, "contains no csv file.")
    result_df = pd.read_csv(csv_list[0])
    action_value_to_sums = [] # mean? vote for majority?
    for csv_file in csv_list:
        csv_df = pd.read_csv(csv_file)
        action_value_to_sums.append(csv_df[[f'U{k}' for k in range(1, 9)]].to_numpy())
    np.array(action_value_to_sums).mean(axis=0)
    result_df[[f'U{k}' for k in range(1, 9)]] = np.array(action_value_to_sums).mean(axis=0)
    result_df.to_csv(submission_file_name+'.csv', index=False)
    ZipFile(submission_file_name+'.zip', mode='w').write(submission_file_name+'.csv')


def d4rl_dataset_to_dt_format(d4rl_dataset, name='data/custom'):
    N = d4rl_dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    use_timeouts = 'timeouts' in d4rl_dataset
    episode_step = 0
    paths = []
    for i in range(N):
        done_bool = bool(d4rl_dataset['terminals'][i])
        if use_timeouts:
            final_timestep = d4rl_dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == 1000-1)
        for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
            data_[k].append(d4rl_dataset[k][i])
        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {k: np.array(data_[k]) for k in data_}
            paths.append(episode_data) #append for each episode
            data_ = collections.defaultdict(list)
        episode_step += 1

    returns = np.array([np.sum(p['rewards']) for p in paths])
    num_samples = np.sum([p['rewards'].shape[0] for p in paths])
    print(f'Number of samples collected: {num_samples}')
    print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}, ceil = {int(np.ceil(np.max(returns)))}')
    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(paths, f)
    states, traj_lens, returns = [], [], []
    for path in paths:
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)
    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    # print(f'states: mean = {state_mean}, std = {state_std}')
    return paths, num_samples, int(np.ceil(np.max(returns))), state_mean, state_std


if __name__ == '__main__':
    Data_obj = DataObj(shuffle=False)
    dataset_d4rl, dataset_d4rl_eval = Data_obj.get_dataset()