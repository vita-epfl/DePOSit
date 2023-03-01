import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from model import ModelMain
from utils.amass_3dpw import AMASS, D3DPW

parser = argparse.ArgumentParser(description='Arguments for running the scripts')
parser.add_argument("--miss_rate", type=int, default=20)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--skip_rate_train", type=int, default=5)
parser.add_argument("--skip_rate_val", type=int, default=15)
parser.add_argument("--joints", type=int, default=32)
parser.add_argument("--input_n", type=int, default=25)
parser.add_argument("--output_n", type=int, default=25)
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                    help='Choose to train or test from the model')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--dataset', type=str, default='AMASS', choices=['AMASS', '3DPW'])
parser.add_argument('--data', type=str, default='all', choices=['one', 'all'],
                    help='Choose to train on one subject or all')
parser.add_argument('--output_dir', type=str, default='default')
parser.add_argument('--data_dir', type=str, default='/datasets/')

args = parser.parse_args()
print(args)

config = {
    'train':
        {
            'epochs': 100,
            'batch_size': 32,
            'batch_size_test': 32,
            'lr': 1.0e-3
        },
    'diffusion':
        {
            'layers': 4,
            'channels': 64,
            'nheads': 8,
            'diffusion_embedding_dim': 128,
            'beta_start': 0.0001,
            'beta_end': 0.5,
            'num_steps': 50,
            'schedule': "cosine"
        },
    'model':
        {
            'is_unconditional': 0,
            'timeemb': 128,
            'featureemb': 16
        }
}


def save_csv_log(head, value, is_create=False, file_name='test'):
    if len(value.shape) < 2:
        value = np.expand_dims(value, axis=0)
    df = pd.DataFrame(value)
    file_path = f'{output_dir}/{file_name}.csv'
    if not os.path.exists(file_path) or is_create:
        df.to_csv(file_path, header=head, index=False)
    else:
        with open(file_path, 'a') as f:
            df.to_csv(f, header=False, index=False)


def save_state(model, optimizer, scheduler, epoch_no, foldername):
    params = {'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch_no}
    torch.save(model.state_dict(), foldername + "/model.pth")
    torch.save(params, foldername + "/params.pth")


def train(
        model,
        config,
        train_loader,
        valid_loader=None,
        valid_epoch_interval=5,
        foldername="",
        load_state=False
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if load_state:
        optimizer.load_state_dict(torch.load(f'{output_dir}/params.pth')['optimizer'])

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )
    if load_state:
        lr_scheduler.load_state_dict(torch.load(f'{output_dir}/params.pth')['scheduler'])

    train_loss = []
    valid_loss = []
    train_loss_epoch = []
    valid_loss_epoch = []

    best_valid_loss = 1e10
    start_epoch = 0
    if load_state:
        start_epoch = torch.load(f'{output_dir}/params.pth')['epoch']
    for epoch_no in range(start_epoch, config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                batch = train_batch

                optimizer.zero_grad()

                loss = model(batch).mean()
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            lr_scheduler.step()
        train_loss.append(avg_loss / batch_no)
        train_loss_epoch.append(epoch_no)
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        batch = valid_batch
                        loss = model(batch, is_train=0).mean()
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            valid_loss.append(avg_loss_valid / batch_no)
            valid_loss_epoch.append(epoch_no)
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )
                save_state(model, optimizer, lr_scheduler, epoch_no, foldername)

            if (epoch_no + 1) == config["epochs"]:
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.plot(train_loss_epoch, train_loss)
                ax.plot(valid_loss_epoch, valid_loss)
                ax.grid(True)
                plt.show()
                fig.savefig(f"{foldername}/loss.png")

    save_state(model, optimizer, lr_scheduler, config["epochs"], foldername)
    np.save(f'{foldername}/train_loss.npy', np.array(train_loss))
    np.save(f'{foldername}/valid_loss.npy', np.array(valid_loss))


def mpjpe_error(batch_imp, batch_gt):
    batch_imp = batch_imp.contiguous().view(-1, 3)
    batch_gt = batch_gt.contiguous().view(-1, 3)

    return torch.mean(torch.norm(batch_gt - batch_imp, 2, 1))


def evaluate(model, loader, nsample=5, scaler=1, sample_strategy='best'):
    with torch.no_grad():
        model.eval()
        mpjpe_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []

        titles = np.array(range(output_n)) + 1
        m_p3d_h36 = np.zeros([output_n])
        n = 0

        with tqdm(loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                batch = test_batch

                batch_size = batch["pose"].shape[0]
                n += batch_size

                output = model.module.evaluate(batch, nsample)
                samples, c_target, eval_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)

                samples_mean = np.mean(samples.cpu().numpy(), axis=1)

                renorm_pose = []
                renorm_c_target = []

                for i in range(len(samples_mean)):
                    renorm_c_target_i = c_target.cpu().data.numpy()[i][input_n:input_n + output_n] * 1000

                    if sample_strategy == 'best':
                        best_renorm_pose = None
                        best_error = float('inf')

                        for j in range(nsample):
                            renorm_pose_j = samples.cpu().numpy()[i][j][input_n:input_n + output_n] * 1000
                            error = mpjpe_error(torch.from_numpy(renorm_pose_j).view(output_n, 18, 3),
                                                torch.from_numpy(renorm_c_target_i).view(output_n, 18, 3))
                            if error.item() < best_error:
                                best_error = error.item()
                                best_renorm_pose = renorm_pose_j
                    else:
                        best_renorm_pose = samples_mean[i][input_n:input_n + output_n] * 1000
                    renorm_pose.append(best_renorm_pose)
                    renorm_c_target.append(renorm_c_target_i)

                renorm_pose = torch.from_numpy(np.array(renorm_pose))
                renorm_c_target = torch.from_numpy(np.array(renorm_c_target))

                eval_points = eval_points[:, input_n:input_n + output_n, :]

                mpjpe_p3d_h36 = torch.sum(torch.mean(
                    torch.norm(renorm_c_target.view(-1, output_n, 18, 3)
                               - renorm_pose.view(-1, output_n, 18, 3), dim=3),
                    dim=2), dim=0)
                m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()

                all_target.append(renorm_c_target)
                all_evalpoint.append(eval_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(renorm_pose)


                mpjpe_current = mpjpe_error(renorm_pose.view(-1, output_n, 18, 3),
                                            renorm_c_target.view(-1, output_n, 18, 3))

                mpjpe_total += mpjpe_current.item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "mpjpe_total": mpjpe_total / batch_no,
                        "batch_no": batch_no
                    },
                    refresh=True,
                )

            print("Average MPJPE:", mpjpe_total / batch_no)

            ret = {}
            m_p3d_h36 = m_p3d_h36 / n
            for j in range(output_n):
                ret["#{:d}".format(titles[j])] = m_p3d_h36[j]

            return all_generated_samples, all_target, all_evalpoint, ret


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: %s' % device)

    data_dir = args.data_dir
    output_dir = f'{args.output_dir}'
    input_n = args.input_n
    output_n = args.output_n
    skip_rate = args.skip_rate_train
    config['train']['epochs'] = args.epochs

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = ModelMain(config, device, target_dim=(args.joints * 3))

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    data = AMASS if args.dataset == 'AMASS' else D3DPW

    if args.mode == 'train':
        all_data = True if args.data == 'all' else False
        dataset = data(data_dir, input_n, output_n, args.skip_rate_train, split=0, miss_rate=(args.miss_rate / 100),
                       all_data=all_data)
        print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
        train_loader = DataLoader(dataset, batch_size=config["train"]["batch_size"], shuffle=True, num_workers=0,
                                  pin_memory=True)

        valid_dataset = data(data_dir, input_n, output_n, args.skip_rate_val, split=1, miss_rate=(args.miss_rate / 100),
                             all_data=all_data)
        print('>>> Validation dataset length: {:d}'.format(valid_dataset.__len__()))
        valid_loader = DataLoader(valid_dataset, batch_size=config["train"]["batch_size"], shuffle=True, num_workers=0,
                                  pin_memory=True)

        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=output_dir,
            load_state=args.resume
        )
    elif args.mode == 'test':
        test_dataset = data(data_dir, input_n, output_n, skip_rate, split=2, miss_rate=(args.miss_rate / 100))
        print('>>> Test dataset length: {:d}'.format(test_dataset.__len__()))
        test_loader = DataLoader(test_dataset, batch_size=config["train"]["batch_size_test"], shuffle=False,
                                 num_workers=0, pin_memory=True)

        model.load_state_dict(torch.load(f'{output_dir}/model.pth'))
        pose, target, mask, ret = evaluate(
            model,
            test_loader,
            nsample=5,
            scaler=1,
            sample_strategy='best'
        )

        ret_log = np.array([])
        head = np.array([])
        for k in range(1, output_n + 1):
            head = np.append(head, [f'#{k}'])
        for k in ret.keys():
            ret_log = np.append(ret_log, [ret[k]])
        save_csv_log(head, ret_log, is_create=True, file_name=f'fde_{args.dataset}')
