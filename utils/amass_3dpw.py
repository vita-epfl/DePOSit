import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import ang2joint
import pickle as pkl
from os import walk


class AMASS(Dataset):

    def __init__(self, data_dir, input_n, output_n, skip_rate, split=0, miss_rate=0.2, all_data=False):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        self.path_to_data = os.path.join(data_dir, 'AMASS') + '/'
        self.split = split
        self.in_n = input_n
        self.out_n = output_n
        self.miss_rate = miss_rate
        # self.sample_rate = opt.sample_rate
        self.p3d = []
        self.keys = []
        self.data_idx = []
        self.joint_used = np.arange(4, 22)  # start from 4 for 17 joints, removing the non moving ones
        seq_len = self.in_n + self.out_n

        if all_data:
            amass_splits = [
                ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'EKUT', 'TCD_handMocap', 'ACCAD'],
                # ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
                ['SFU'],
                ['BioMotionLab_NTroje'],
            ]
        else:
            amass_splits = [
                ['MPI_Limits', 'TotalCapture', 'EKUT'],
                ['SFU'],
                ['BioMotionLab_NTroje'],
            ]
        # amass_splits = [['BioMotionLab_NTroje'], ['HumanEva'], ['SSM_synced']]
        # amass_splits = [['HumanEva'], ['HumanEva'], ['HumanEva']]
        # amass_splits[0] = list(
        #     set(amass_splits[0]).difference(set(amass_splits[1] + amass_splits[2])))

        # from human_body_prior.body_model.body_model import BodyModel
        # from smplx import lbs
        # root_path = os.path.dirname(__file__)
        # bm_path = root_path[:-6] + '/body_models/smplh/neutral/model.npz'
        # bm = BodyModel(bm_path=bm_path, num_betas=16, batch_size=1, model_type='smplh')
        # beta_mean = np.array([0.41771687, 0.25984767, 0.20500051, 0.13503872, 0.25965645, -2.10198147, -0.11915666,
        #                       -0.5498772, 0.30885323, 1.4813145, -0.60987528, 1.42565269, 2.45862726, 0.23001716,
        #                       -0.64180912, 0.30231911])
        # beta_mean = torch.from_numpy(beta_mean).unsqueeze(0).float()
        # # Add shape contribution
        # v_shaped = bm.v_template + lbs.blend_shapes(beta_mean, bm.shapedirs)
        # # Get the joints
        # # NxJx3 array
        # p3d0 = lbs.vertices2joints(bm.J_regressor, v_shaped)  # [1,52,3]
        # p3d0 = (p3d0 - p3d0[:, 0:1, :]).float().cuda().cpu().data.numpy()
        # parents = bm.kintree_table.data.numpy()[0, :]
        # np.savez_compressed('smpl_skeleton.npz', p3d0=p3d0, parents=parents)

        # load mean skeleton
        skel = np.load('./utils/body_models/smpl_skeleton.npz')
        p3d0 = torch.from_numpy(skel['p3d0']).float().cuda()
        parents = skel['parents']
        parent = {}
        for i in range(len(parents)):
            parent[i] = parents[i]
        n = 0
        for ds in amass_splits[split]:
            if not os.path.isdir(self.path_to_data + ds):
                print(f'{ds} not found!')
                continue
            print('>>> loading {}'.format(ds))
            for sub in os.listdir(self.path_to_data + ds):
                if not os.path.isdir(self.path_to_data + ds + '/' + sub):
                    continue
                for act in os.listdir(self.path_to_data + ds + '/' + sub):
                    if not act.endswith('.npz'):
                        continue
                    # if not ('walk' in act or 'jog' in act or 'run' in act or 'treadmill' in act):
                    #     continue
                    pose_all = np.load(self.path_to_data + ds + '/' + sub + '/' + act)
                    try:
                        poses = pose_all['poses']
                    except:
                        print('no poses at {}/{}/{}'.format(ds, sub, act))
                        continue
                    frame_rate = pose_all['mocap_framerate']
                    # gender = pose_all['gender']
                    # dmpls = pose_all['dmpls']
                    # betas = pose_all['betas']
                    # trans = pose_all['trans']
                    fn = poses.shape[0]
                    sample_rate = int(frame_rate // 25)
                    fidxs = range(0, fn, sample_rate)
                    fn = len(fidxs)
                    poses = poses[fidxs]
                    poses = torch.from_numpy(poses).float().cuda()
                    poses = poses.reshape([fn, -1, 3])
                    # remove global rotation
                    poses[:, 0] = 0
                    p3d0_tmp = p3d0.repeat([fn, 1, 1])
                    p3d = ang2joint.ang2joint(p3d0_tmp, poses, parent)
                    # self.p3d[(ds, sub, act)] = p3d.cpu().data.numpy()
                    pose = p3d.cpu().data.numpy()[:, self.joint_used, :]
                    self.p3d.append(pose)
                    L, J, C = pose.shape
                    self.p3d[-1] = pose.reshape(L, J * C)

                    if split == 2:
                        valid_frames = np.arange(0, fn - seq_len + 1, skip_rate)
                    else:
                        valid_frames = np.arange(0, fn - seq_len + 1, skip_rate)

                    # tmp_data_idx_1 = [(ds, sub, act)] * len(valid_frames)
                    self.keys.append((ds, sub, act))
                    tmp_data_idx_1 = [n] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                    n += 1

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        pose = self.p3d[key][fs]

        mask = np.zeros((pose.shape[0], pose.shape[1]))
        mask[0:self.in_n, :] = 1
        mask[self.in_n:self.in_n + self.out_n, :] = 0

        data = {
            "pose": pose,
            "mask": mask,
            "timepoints": np.arange(self.in_n + self.out_n)
        }

        return data


class D3DPW(Dataset):

    def __init__(self, data_dir, input_n, output_n, skip_rate, split=0, miss_rate=0.2, all_data=False):
        """
        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_used:
        :param split: 0 train, 1 testing, 2 validation
        :param sample_rate:
        """
        self.path_to_data = os.path.join(data_dir, '3DPW/sequenceFiles')
        self.split = split
        self.in_n = input_n
        self.out_n = output_n
        self.miss_rate = miss_rate
        #self.sample_rate = opt.sample_rate
        self.p3d = []
        self.params = []
        self.keys = []
        self.data_idx = []
        self.joint_used = np.arange(4, 22)
        seq_len = self.in_n + self.out_n

        if split == 0:
            data_path = self.path_to_data + '/train/'
        elif split == 2:
            data_path = self.path_to_data + '/test/'
        elif split == 1:
            data_path = self.path_to_data + '/validation/'
        files = []
        for (dirpath, dirnames, filenames) in walk(data_path):
            files.extend(filenames)

        # from human_body_prior.body_model.body_model import BodyModel
        # from smplx import lbs
        # root_path = os.path.dirname(__file__)
        # bm_path = root_path[:-6] + '/body_models/smplh/neutral/model.npz'
        # bm = BodyModel(bm_path=bm_path, num_betas=16, batch_size=1)
        # beta_mean = np.array([0.41771687, 0.25984767, 0.20500051, 0.13503872, 0.25965645, -2.10198147, -0.11915666,
        #                       -0.5498772, 0.30885323, 1.4813145, -0.60987528, 1.42565269, 2.45862726, 0.23001716,
        #                       -0.64180912, 0.30231911])
        # beta_mean = torch.from_numpy(beta_mean).unsqueeze(0).float()
        # # Add shape contribution
        # v_shaped = bm.v_template + lbs.blend_shapes(beta_mean, bm.shapedirs)
        # # Get the joints
        # # NxJx3 array
        # p3d0 = lbs.vertices2joints(bm.J_regressor, v_shaped)  # [1,52,3]
        # p3d0 = (p3d0 - p3d0[:, 0:1, :]).float().cuda()[:, :22]
        # parents = bm.kintree_table.data.numpy()[0, :]
        skel = np.load('./utils/body_models/smpl_skeleton.npz')
        p3d0 = torch.from_numpy(skel['p3d0']).float().cuda()[:, :22]
        parents = skel['parents']
        parent = {}
        for i in range(len(parents)):
            if i > 21:
                break
            parent[i] = parents[i]
        n = 0

        sample_rate = int(60 // 25)

        for f in files:
            with open(data_path + f, 'rb') as f:
                print('>>> loading {}'.format(f))
                data = pkl.load(f, encoding='latin1')
                joint_pos = data['poses_60Hz']
                for i in range(len(joint_pos)):
                    poses = joint_pos[i]
                    fn = poses.shape[0]
                    fidxs = range(0, fn, sample_rate)
                    fn = len(fidxs)
                    poses = poses[fidxs]
                    poses = torch.from_numpy(poses).float().cuda()
                    poses = poses.reshape([fn, -1, 3])
                    poses = poses[:, :-2]
                    # remove global rotation
                    poses[:, 0] = 0
                    p3d0_tmp = p3d0.repeat([fn, 1, 1])
                    p3d = ang2joint.ang2joint(p3d0_tmp, poses, parent)

                    # self.p3d[(ds, sub, act)] = p3d.cpu().data.numpy()
                    pose = p3d.cpu().data.numpy()[:, self.joint_used, :]
                    self.p3d.append(pose)
                    L, J, C = pose.shape
                    self.p3d[-1] = pose.reshape(L, J * C)
                    # # vis
                    # import utils.vis_util as vis_util
                    # from mpl_toolkits.mplot3d import Axes3D
                    # ax = plt.subplot(111, projection='3d')
                    # vis_util.draw_skeleton_smpl(ax, self.p3d[0][0], parents=parents[:22])

                    if split == 2:
                        # valid_frames = np.arange(0, fn - seq_len + 1, opt.skip_rate_test)
                        # valid_frames = np.arange(0, fn - seq_len + 1, 2)
                        valid_frames = np.arange(0, fn - seq_len + 1)
                    else:
                        valid_frames = np.arange(0, fn - seq_len + 1, skip_rate)

                    # tmp_data_idx_1 = [(ds, sub, act)] * len(valid_frames)
                    tmp_data_idx_1 = [n] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                    n += 1

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.in_n + self.out_n)
        pose = self.p3d[key][fs]

        mask = np.zeros((pose.shape[0], pose.shape[1]))
        mask[0:self.in_n, :] = 1
        mask[self.in_n:self.in_n + self.out_n, :] = 0

        data = {
            "pose": pose,
            "mask": mask,
            "timepoints": np.arange(self.in_n + self.out_n)
        }

        return data