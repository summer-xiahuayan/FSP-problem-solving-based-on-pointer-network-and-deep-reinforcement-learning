# Generate sorting data and store in .txt
# Define the reward function
import random

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from tqdm import trange, tqdm
import os
import pandas as pd
import sys

def GetProcessTime(num_machines,num_jobs,processing_times):
    # 机器数量
    #num_machines = num_machines

    # 工件数量
    #num_jobs = num_jobs

    # 随机生成每个工件在每台机器上的加工时间（假设范围是1到10）
    #processing_times = [[random.randint(1, 10) for _ in range(num_machines)] for _ in range(num_jobs)]
    #df = pd.read_csv('加工时间s.csv')
    #processing_times=df.values.tolist()
    # processing_times=processing_times
    #print(processing_times)
    # 初始化每台机器的可用时间
    machine_available_times = [0] * num_machines

    # 计算每个工件完成的时间
    job_completion_times = [0] * num_jobs

    # 计算流水线总时间
    for job in range(num_jobs):
        start_time = 0
        for machine in range(num_machines):
            start_time = max(start_time, machine_available_times[machine])
            machine_available_times[machine] = start_time + processing_times[job][machine]
            start_time = machine_available_times[machine]
        job_completion_times[job] = start_time

    total_time = job_completion_times[-1]

    #print(f"加工完"+str(num_jobs)+"个工件的总时间："+str(total_time))
    return total_time


def reward(sample_solution, USE_CUDA=False):
    batch_size = sample_solution[0].size(0)
    num_jobs = len(sample_solution)
    num_machines=sample_solution[0].size(1)
    processing_times = Variable(torch.zeros([batch_size]))

    if USE_CUDA:
        processing_times = processing_times.cuda()
    # 计算流水线总时间
    for i in range(batch_size):
        machine_available_times = [0] * num_machines
        job_completion_times = [0] * num_jobs
        for job in range(num_jobs):
            start_time = 0
            for machine in range(num_machines):
                start_time = max(start_time, machine_available_times[machine])
                time=sample_solution[job][i]
                machine_available_times[machine] = start_time + time[machine]
                start_time = machine_available_times[machine]
            job_completion_times[job] = start_time
        total_time = job_completion_times[-1]
        processing_times[i] = total_time
    # For TSP_20 - map to a number between 0 and 1
    # min_len = 3.5
    # max_len = 10.
    # TODO: generalize this for any TSP size
    #tour_len = -0.1538*tour_len + 1.538
    #tour_len[tour_len < 0.] = 0.
    return processing_times

def create_dataset(
        train_size,
        val_size,
        #test_size,
        data_dir,
        data_len,
        seed=None):

    if seed is not None:
        torch.manual_seed(seed)

    train_task = 'fsp-size-{}-len-{}-train-random.txt'.format(train_size, data_len)
    val_task = 'fsp-size-{}-len-{}-val-random.txt'.format(val_size, data_len)
    #test_task = 'sorting-size-{}-len-{}-test.txt'.format(test_size, data_len)

    train_fname = os.path.join(data_dir, train_task)
    val_fname = os.path.join(data_dir, val_task)


    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    else:
        if os.path.exists(train_fname) and os.path.exists(val_fname):
            return train_fname, val_fname

    train_set = open(os.path.join(data_dir, train_task), 'w')
    val_set = open(os.path.join(data_dir, val_task), 'w')
    #test_set = open(os.path.join(data_dir, test_task), 'w')

    def to_string(tensor):
        """
        Convert a a torch.LongTensor
        of size data_len to a string
        of integers separated by whitespace
        and ending in a newline character
        """
        line = ''
        for i in range(len(tensor)):
            for j in range(len(tensor[i])):
                if j==len(tensor[i])-1:
                    line += '{} '.format(tensor[i][j])
                else:
                    line += '{},'.format(tensor[i][j])
        line +=  '\n'
        # line += str(tensor[-1]) + '\n'
        return line

    print('Creating training data set for {}...'.format(train_task))
    if(data_len==20):
        df = pd.read_csv('data/fsp/processtime-20s.csv')
    elif(data_len==100):
        df = pd.read_csv('data/fsp/processtime-100s.csv')
    else:
        df = pd.read_csv('data/fsp/processtime-50s.csv')
    # seeds=df.values.tolist()
    # seeds_time=[]
    # for x in seeds:
    #     seeds_time.append(x[0:4])
    # Generate a training set of size train_size
    for i in trange(train_size):
        # train_shuffled_list = random.sample(seeds_time, len(seeds_time))
       # print(train_shuffled_list)
        train_shuffled_list=[[random.randint(1, 100) for _ in range(4)] for _ in range(20)]
        train_set.write(to_string(train_shuffled_list))

    print('Creating validation data set for {}...'.format(val_task))

    for i in trange(val_size):
        val_shuffled_list = [[random.randint(1, 100) for _ in range(4)] for _ in range(20)]
        val_set.write(to_string(val_shuffled_list))

    #    print('Creating test data set for {}...'.format(test_task))
    #
    #    for i in trange(test_size):
    #        x = torch.randperm(data_len)
    #        test_set.write(to_string(x))

    train_set.close()
    val_set.close()
    #    test_set.close()
    return train_fname, val_fname

class FspingDataset(Dataset):

    def __init__(self, dataset_fname):
        super(FspingDataset, self).__init__()
        print('Loading training data into memory')
        self.data_set = []
        with open(dataset_fname, 'r') as dset:
            lines = dset.readlines()
            for next_line in tqdm(lines):
                toks = next_line.strip().split(" ")
                items=toks[0].strip().split(",")
                sample = torch.zeros(len(items),len(toks)).long()
                for idx in range(len(items)):
                    for idy, tok in enumerate(toks):
                        time=tok.split(",")[idx]
                        sample[idx, idy] = int(time.split(".")[0])
                self.data_set.append(sample)

        self.size = len(self.data_set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data_set[idx]

if __name__ == '__main__':
    create_dataset(128, 128, 'data\\fsp', 100, 123)
    # training_dataset = FspingDataset('data/fsp/fsp-size-1000-len-20-train.txt')
    # val_dataset = FspingDataset('data/fsp/fsp-size-100-len-20-val.txt')
    # A = torch.tensor([3.0, 4.0])
    # norm = torch.norm(A)
    # print(norm)

