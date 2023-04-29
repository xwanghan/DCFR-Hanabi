import numpy as np
import torch
import pickle


class Buffer:

    def __init__(self, size, obs_max_length=30, obs_size=113, action_size=9):

        self.memory = [None] * size
        self.size = size
        self.obs_max_length = obs_max_length
        self.obs_size = obs_size
        self.action_size = action_size

        self.n_entries = 0
        self.n_entries_added = 0

    def form_entry(self, obs_h, t, arr):
        obs_h = torch.tensor(obs_h).reshape(-1, self.obs_size).float()
        length = torch.tensor(obs_h.shape[0]).int()
        t = torch.tensor(t).int()
        arr = torch.tensor(arr).float()

        return [obs_h, t, arr, length]

    def add(self, obs_h, t, arr):

        entry = self.form_entry(obs_h, t, arr)
        # print('formed entry: ', entry)

        if self.n_entries < self.size:

            self.memory[self.n_entries] = entry
            self.n_entries += 1
            self.n_entries_added += 1

        else:

            self.n_entries_added += 1

            acceptance_prob = self.n_entries_added / self.size

            if np.random.random() < acceptance_prob:
                idx = np.random.randint(0, self.size)
                self.memory[idx] = entry

    def add_buffer(self, local_buffer):

        for i in range(len(local_buffer)):

            if self.n_entries < self.size:
                self.memory[self.n_entries] = local_buffer.memory[i]
                self.n_entries += 1
                self.n_entries_added += 1

            else:

                self.n_entries_added += 1

                acceptance_prob = self.n_entries_added / self.size

                if np.random.random() < acceptance_prob:
                    idx = np.random.randint(0, self.size)
                    self.memory[idx] = local_buffer.memory[i]

    def __len__(self):
        return self.n_entries

    def recast(self, n_samples):

        n_samples = min(n_samples, self.n_entries)

        sample_idx = np.random.randint(0, self.n_entries, n_samples)

        x_train = torch.zeros((n_samples, self.obs_max_length, self.obs_size)).float()
        t_arr = torch.zeros(n_samples).int()
        action_train = torch.zeros((n_samples, self.action_size)).float()
        lengths = torch.zeros(n_samples).int()

        for i in range(n_samples):
            lengths[i] = self.memory[sample_idx[i]][3]
            x_train[i][:lengths[i]] = self.memory[sample_idx[i]][0]
            t_arr[i] = self.memory[sample_idx[i]][1]
            action_train[i] = self.memory[sample_idx[i]][2]

        return x_train.float().cuda(), t_arr.int().cuda(), action_train.float().cuda(), lengths.int().cuda()

    def reset(self):

        self.memory = [None] * self.size

        self.n_entries = 0
        self.n_entries_added = 0

    def save(self, prefix_str, t):

        filename = './Buffer/' + prefix_str + '_'

        for i in range(self.__len__()):
            x_train = self.memory[i][0]
            t_arr = self.memory[i][1]
            action_train = self.memory[i][2]
            length = self.memory[i][3]

            torch.save(x_train, filename + 'x' + '_' + str(t + i))
            torch.save(t_arr, filename + 't' + '_' + str(t + i))
            torch.save(action_train, filename + 'y' + '_' + str(t + i))
            torch.save(length, filename + 'xl' + '_' + str(t + i))

        self.memory = [None] * self.size

        self.n_entries = 0
        self.n_entries_added = 0
