import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import fsp_task


def gantt(processing_times,num_machines,num_jobs):
    # 示例调用
    num_machines = num_machines
    num_jobs = num_jobs
    #processing_times = [[3, 2], [2, 1], [1, 3]]  # 假设有3台机器，2个工件，每个工件在每台机器上的加工时间

    total_time, machine_start_times, machine_end_times = fsp_task.GetMachineTime(num_machines, num_jobs, processing_times)

    print("总时间：", total_time)
    print("每台机器每个工件的加工起始时间：", machine_start_times)
    print("每台机器每个工件的加工结束时间：", machine_end_times)








if __name__ == '__main__':

    # 假设你的模型保存在 'path_to_saved_model.pt'
    model_path = r"E:\PYCHARM\pycharm project\FSP-problem-solving-based-on-pointer-network-and-deep-reinforcement-learning\output\fsp_20\train_size_1000000\DRL_FSP_20.pt"
    # 加载模型
    model = torch.load(model_path)

    # 设置为评估模式
    model.eval()

    # 准备输入数据，这里需要根据你的具体任务来准备
    # 假设你的输入数据是一个张量，你需要确保它与训练时的格式一致
    # 例如，如果输入是一批20个点的坐标，你需要创建一个形状为[1, 2, 20]的张量
    train_fname=r"C:\Users\Administrator\PycharmProjects\FSP-problem-solving-based-on-pointer-network-and-deep-reinforcement-learning\data\fsp\fsp-size-10-len-20-train.txt"
    training_dataset = fsp_task.FspingDataset(train_fname)

    # 进行预测
    # put in test mode!
    model.eval()
    validation_dataloader = DataLoader(training_dataset, batch_size=1, shuffle=True, num_workers=0)


    for batch_id, val_batch in enumerate(tqdm(validation_dataloader,disable=False)):
        # bat = Variable(val_batch)
        val_batch_tensor = val_batch
        # 根据配置选择使用 CPU 或 GPU
        device = torch.device("cuda")
        # 将数据移动到 GPU 或 CPU
        val_batch_tensor = val_batch_tensor.to(device)
        R, probs, actions, action_idxs = model(val_batch_tensor)
        example_output = []
        example_input = []
        for idx, action in enumerate(actions):
            example_output.append(action[0].cpu())
                #print(action)
            example_input.append(val_batch[0, :, idx])
        #print(fsp_task.GetProcessTime(4,20,example_output))
        print('Step: {}'.format(batch_id))
        print('Example test input: {}'.format(example_input))
        print('Example test output: {}'.format(example_output))
        print('Example test reward: {}'.format(R[0].item()))

processing_times=[x.tolist() for x in example_output]

gantt(processing_times,4,20)