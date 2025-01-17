#!/usr/bin/env python

import argparse
import os
from tqdm import tqdm 

import pprint as pp
import numpy as np

import torch
print(torch.__version__)
import torch.optim as optim

from torch.optim import lr_scheduler

from torch.utils.data import DataLoader
from tensorboard_logger import configure, log_value

from neural_combinatorial_rl import NeuralCombOptRL



def str2bool(v):
      return v.lower() in ('true', '1')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Neural Combinatorial Optimization with RL")

    # Data
    parser.add_argument('--task', default='hfsp_20', help="The task to solve, in the form {COP}_{size}, e.g., tsp_20")
    parser.add_argument('--batch_size', default=128, help='')
    parser.add_argument('--train_size', default=128, help='')
    parser.add_argument('--val_size', default=128, help='')
    # Network
    parser.add_argument('--embedding_dim', default=256, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', default=256, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_process_blocks', default=3, help='Number of process block iters to run in the Critic network')
    parser.add_argument('--n_glimpses', default=2, help='No. of glimpses to use in the pointer network')
    parser.add_argument('--use_tanh', type=str2bool, default=True)
    parser.add_argument('--tanh_exploration', default=10, help='Hyperparam controlling exploration in the pointer net by scaling the tanh in the softmax')
    parser.add_argument('--dropout', default=0., help='')
    parser.add_argument('--terminating_symbol', default='<0>', help='')
    parser.add_argument('--beam_size', default=1, help='Beam width for beam search')

    # Training
    parser.add_argument('--actor_net_lr', default=1e-4, help="Set the learning rate for the actor network")
    parser.add_argument('--critic_net_lr', default=1e-4, help="Set the learning rate for the critic network")
    parser.add_argument('--actor_lr_decay_step', default=5000, help='')
    parser.add_argument('--critic_lr_decay_step', default=5000, help='')
    parser.add_argument('--actor_lr_decay_rate', default=0.96, help='')
    parser.add_argument('--critic_lr_decay_rate', default=0.96, help='')
    parser.add_argument('--reward_scale', default=2, type=float,  help='')
    parser.add_argument('--is_train', type=str2bool, default=True, help='')
    parser.add_argument('--n_epochs', default=1, help='')
    parser.add_argument('--random_seed', default=24601, help='')
    parser.add_argument('--max_grad_norm', default=2.0, help='Gradient clipping')
    parser.add_argument('--use_cuda', type=str2bool, default=True, help='')
    parser.add_argument('--critic_beta', type=float, default=0.9, help='Exp mvg average decay')

    # Misc
    parser.add_argument('--log_step', default=50, help='Log info every log_step steps')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--run_name', type=str, default='train_size_128_')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--epoch_start', type=int, default=0, help='Restart at epoch #')
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--disable_tensorboard', type=str2bool, default=False)
    parser.add_argument('--plot_attention', type=str2bool, default=False)
    parser.add_argument('--disable_progress_bar', type=str2bool, default=False)

    args = vars(parser.parse_args())

    # Pretty print the run args
    pp.pprint(args)

    # Set the random seed
    torch.manual_seed(int(args['random_seed']))

    # Optionally configure tensorboard
    if not args['disable_tensorboard']:
        configure(os.path.join(args['log_dir'], args['task'], args['run_name']))

    # Task specific configuration - generate dataset if needed
    task = args['task'].split('_')
    COP = task[0]
    size = int(task[1])
    data_dir = 'data/' + COP

    if COP == 'sort':
        import sorting_task

        input_dim = 1
        reward_fn = sorting_task.reward
        train_fname, val_fname = sorting_task.create_dataset(
            int(args['train_size']),
            int(args['val_size']),
            data_dir,
            data_len=size)
        training_dataset = sorting_task.SortingDataset(train_fname)
        val_dataset = sorting_task.SortingDataset(val_fname)
    elif COP == 'tsp':
        import tsp_task
        input_dim = 2
        reward_fn = tsp_task.reward
        val_fname = tsp_task.create_dataset(
            problem_size=str(size),
            data_dir=data_dir)
        training_dataset = tsp_task.TSPDataset(train=True, size=size,
             num_samples=int(args['train_size']))
        val_dataset = tsp_task.TSPDataset(train=True, size=size,
                num_samples=int(args['val_size']))
    elif COP == 'fsp':
        import fsp_task
        input_dim = 4
        reward_fn = fsp_task.reward
        train_fname, val_fname = fsp_task.create_dataset(
            int(args['train_size']),
            int(args['val_size']),
            data_dir,
            data_len=size)
        training_dataset = fsp_task.FspingDataset(train_fname)
        val_dataset = fsp_task.FspingDataset(val_fname)
    elif COP == 'hfsp':
        import hfsp_task
        input_dim = sum(hfsp_task.Machine)
        reward_fn = hfsp_task.reward
        train_fname, val_fname = hfsp_task.create_dataset(
            int(args['train_size']),
            int(args['val_size']),
            data_dir,
            data_len=size)
        training_dataset = hfsp_task.HfspDataset(train_fname)
        val_dataset = hfsp_task.HfspDataset(val_fname)
    else:
        print('Currently unsupported task!')
        exit(1)

    # Load the model parameters from a saved state
    if args['load_path'] != '':
        print('  [*] Loading model from {}'.format(args['load_path']))

        model = torch.load(
            os.path.join(
                os.getcwd(),
                args['load_path']
            ))
        model.actor_net.decoder.max_length = size
        model.is_train = args['is_train']
    else:
        # Instantiate the Neural Combinatorial Opt with RL module
        model = NeuralCombOptRL(
            input_dim,
            int(args['embedding_dim']),
            int(args['hidden_dim']),
            size, # decoder len
            args['terminating_symbol'],
            int(args['n_glimpses']),
            int(args['n_process_blocks']),
            float(args['tanh_exploration']),
            args['use_tanh'],
            int(args['beam_size']),
            reward_fn,
            args['is_train'],
            args['use_cuda'])


    save_dir = os.path.join(os.getcwd(),
               args['output_dir'],
               args['task'],
               args['run_name'])

    try:
        os.makedirs(save_dir)
    except:
        pass

    #critic_mse = torch.nn.MSELoss()
    #critic_optim = optim.Adam(model.critic_net.parameters(), lr=float(args['critic_net_lr']))
    actor_optim = optim.Adam(model.actor_net.parameters(), lr=float(args['actor_net_lr']))

    actor_scheduler = lr_scheduler.MultiStepLR(actor_optim,
            range(int(args['actor_lr_decay_step']), int(args['actor_lr_decay_step']) * 1000,
                int(args['actor_lr_decay_step'])), gamma=float(args['actor_lr_decay_rate']))

    #critic_scheduler = lr_scheduler.MultiStepLR(critic_optim,
    #        range(int(args['critic_lr_decay_step']), int(args['critic_lr_decay_step']) * 1000,
    #            int(args['critic_lr_decay_step'])), gamma=float(args['critic_lr_decay_rate']))

    training_dataloader = DataLoader(training_dataset, batch_size=int(args['batch_size']),
        shuffle=True, num_workers=8)

    validation_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1)

    critic_exp_mvg_avg = torch.zeros(1)
    beta = args['critic_beta']

    if args['use_cuda']:
        model = model.cuda()
        #critic_mse = critic_mse.cuda()
        critic_exp_mvg_avg = critic_exp_mvg_avg.cuda()

    step = 0
    val_step = 0

    if not args['is_train']:
        args['n_epochs'] = '1'


    epoch = int(args['epoch_start'])
    for i in range(epoch, epoch + int(args['n_epochs'])):

        if args['is_train']:

            # for batch_id, sample_batch in enumerate(validation_dataloader):
            #     print(sample_batch)
            # put in train mode!
            model.train()

            # sample_batch is [batch_size x input_dim x sourceL]
            for batch_id, sample_batch in enumerate(tqdm(training_dataloader,
                                                         disable=args['disable_progress_bar'])):
                #print(sample_batch)
                # 直接使用 Tensor 而不是 Variable
                batch_tensor = sample_batch
                # 根据配置选择使用 CPU 或 GPU
                device = torch.device("cuda" if args['use_cuda'] else "cpu")

                # 将数据移动到 GPU 或 CPU
                batch_tensor = batch_tensor.to(device)

                # 接下来的代码中，使用 batch_tensor 进行计算...


                R, probs, actions, actions_idxs = model(batch_tensor)

                if batch_id == 0:
                    critic_exp_mvg_avg = R.mean()
                else:
                    critic_exp_mvg_avg = (critic_exp_mvg_avg * beta) + ((1. - beta) * R.mean())

                advantage = R - critic_exp_mvg_avg

                logprobs = 0
                nll = 0
                for prob in probs:
                    # compute the sum of the log probs
                    # for each tour in the batch
                    logprob = torch.log(prob)
                    nll += -logprob
                    logprobs += logprob

                # guard against nan
                nll[(nll != nll).detach()] = 0.
                # clamp any -inf's to 0 to throw away this tour
                logprobs[(logprobs < -1000).detach()] = 0.

                # multiply each time step by the advanrate
                reinforce = advantage * logprobs
                actor_loss = reinforce.mean()

                actor_optim.zero_grad()

                actor_loss.backward()

                # clip gradient norms
                torch.nn.utils.clip_grad_norm(model.actor_net.parameters(),
                        float(args['max_grad_norm']), norm_type=2)

                actor_optim.step()
                actor_scheduler.step()

                critic_exp_mvg_avg = critic_exp_mvg_avg.detach()

                #critic_scheduler.step()

                #R = R.detach()
                #critic_loss = critic_mse(v.squeeze(1), R)
                #critic_optim.zero_grad()
                #critic_loss.backward()

                #torch.nn.utils.clip_grad_norm(model.critic_net.parameters(),
                #        float(args['max_grad_norm']), norm_type=2)

                #critic_optim.step()

                step += 1

                if not args['disable_tensorboard']:
                    #log_value('avg_reward', R.mean().data[0], step)
                    log_value('avg_reward', R.mean().item(), step)
                    log_value('actor_loss', actor_loss.item(), step)
                    #log_value('critic_loss', critic_loss.data[0], step)
                    log_value('critic_exp_mvg_avg', critic_exp_mvg_avg.item(), step)
                    log_value('nll', nll.mean().item(), step)

                if step % int(args['log_step']) == 0:
                    print('epoch: {}, train_batch_id: {}, avg_reward: {}'.format(
                        i, batch_id, R.mean().item()))
                    example_output = []
                    example_input = []
                    for idx, action in enumerate(actions):
                        if task[0] == 'tsp':
                            example_output.append(actions_idxs[idx][0].item())
                        elif task[0] == 'fsp':
                            example_output.append(action[0].cpu())
                        else:
                            example_output.append(action[0].item())  # <-- ??
                        example_input.append(sample_batch[0, :, idx][0])
                    #print('Example train input: {}'.format(example_input))
                    print('Example train output: {}'.format(example_output))

        # Use beam search decoding for validation
        #model.actor_net.decoder.decode_type = "beam_search"

        print('\n~Validating~\n')

        example_input = []
        example_output = []
        avg_reward = []

        # put in test mode!
        model.eval()

        for batch_id, val_batch in enumerate(tqdm(validation_dataloader,
                disable=args['disable_progress_bar'])):
           # bat = Variable(val_batch)

            val_batch_tensor = val_batch
            # 根据配置选择使用 CPU 或 GPU
            device = torch.device("cuda" if args['use_cuda'] else "cpu")

            # 将数据移动到 GPU 或 CPU
            val_batch_tensor = val_batch_tensor.to(device)


            R, probs, actions, action_idxs = model(val_batch_tensor)

            avg_reward.append(R[0].item())
            val_step += 1.

            if not args['disable_tensorboard']:
                log_value('val_avg_reward', R[0].item(), int(val_step))

            if val_step % int(args['log_step']) == 0:
                example_output = []
                example_input = []
                # for idx, action in enumerate(actions):
                #     if task[0] == 'tsp':
                #         example_output.append(action_idxs[idx][0].item())
                #     elif task[0] == 'fsp':
                #         example_output.append(action[0].cpu())
                #     elif task[0]=='hfsp':
                #         example_output.append(action[0].cpu())
                #     else:
                #         example_output.append(action[0].item())
                #         #print(action)
                #     example_input.append(val_batch[0, :, idx])

                for idx, action in enumerate(action_idxs):
                    if task[0] == 'tsp':
                        example_output.append(action[0].cpu())
                    elif task[0] == 'fsp':
                        example_output.append(action[0].cpu())
                    elif task[0]=='hfsp':
                        example_output.append(action[0].cpu())
                    else:
                        example_output.append(action[0].cpu())
                        #print(action)
                    example_input.append(val_batch[0, :, idx])

                #print(fsp_task.GetProcessTime(4,20,example_output))
                print('Step: {}'.format(batch_id))
                print('Example test input: {}'.format(example_input))
                print('Example test output: {}'.format(example_output))
                print('Example test reward: {}'.format(R[0].item()))


                # if args['plot_attention']:
                #     probs = torch.cat(probs, 0)
                #     plot_attention(example_input,
                #             example_output, probs.data.cpu().numpy())
        print('Validation overall avg_reward: {}'.format(np.mean(avg_reward)))
        print('Validation overall reward var: {}'.format(np.var(avg_reward)))

        if args['is_train']:
            model.actor_net.decoder.decode_type = "stochastic"

            print('Saving model...')

            torch.save(model, os.path.join(save_dir, 'epoch-{}.pt'.format(i)))

            # If the task requires generating new data after each epoch, do that here!
            # if COP == 'tsp':
            #     training_dataset = tsp_task.TSPDataset(train=True, size=size,
            #         num_samples=int(args['train_size']))
            #     training_dataloader = DataLoader(training_dataset, batch_size=int(args['batch_size']),
            #         shuffle=True, num_workers=1)
            # if COP == 'sort':
            #     train_fname, _ = sorting_task.create_dataset(
            #         int(args['train_size']),
            #         int(args['val_size']),
            #         data_dir,
            #         data_len=size)
            #     training_dataset = sorting_task.SortingDataset(train_fname)
            #     training_dataloader = DataLoader(training_dataset, batch_size=int(args['batch_size']),
            #             shuffle=True, num_workers=1)
