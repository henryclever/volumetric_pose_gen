import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from kinematics_lib import KinematicsLib
import scipy.stats as ss
import torchvision


class CNN(nn.Module):
    def __init__(self, mat_size, out_size, hidden_dim, kernel_size, batch_size):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            mat_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            out_size (int): Number of classes to score
        '''

        super(CNN, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        #print mat_size
        self.loss_vector_type = loss_vector_type
        print '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'

        hidden_dim1= 32
        hidden_dim2 = 48
        hidden_dim3 = 96
        hidden_dim4 = 96

        self.count = 0

        self.CNN_pack1 = nn.Sequential(
            # Vanilla
            # nn.Conv2d(3, hidden_dim1, kernel_size = 5, stride = 2, padding = 1),
            # nn.ReLU(inplace = True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(hidden_dim1, hidden_dim2, kernel_size=5, stride=2, padding= 1),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(hidden_dim2, hidden_dim3, kernel_size=5, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(hidden_dim3, hidden_dim4, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(hidden_dim3, hidden_dim4, kernel_size=3, stride=1, padding= 0),
            # nn.Dropout(p=0.1, inplace=False),

            # 2
            # nn.Conv2d(3, hidden_dim1, kernel_size = 5, stride = 2, padding = 1),
            # nn.ReLU(inplace = True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(hidden_dim1, hidden_dim2, kernel_size=5, stride=2, padding= 1),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(hidden_dim2, hidden_dim3, kernel_size=5, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(hidden_dim3, hidden_dim4, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(hidden_dim3, hidden_dim4, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.1, inplace=False),

            # 3
            # nn.Conv2d(3, hidden_dim1, kernel_size = 5, stride = 2, padding = 1),
            # nn.ReLU(inplace = True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(hidden_dim1, hidden_dim2, kernel_size=5, stride=2, padding= 1),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(hidden_dim2, hidden_dim3, kernel_size=5, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(hidden_dim3, hidden_dim4, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),

            # 4
            # nn.Conv2d(3, hidden_dim1, kernel_size = 7, stride = 2, padding = 1),
            # nn.ReLU(inplace = True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.MaxPool2d(3, stride=2),
            # nn.Conv2d(hidden_dim1, hidden_dim2, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(hidden_dim2, hidden_dim3, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(hidden_dim3, hidden_dim4, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),

            # 5
            # nn.Conv2d(3, 32, kernel_size = 7, stride = 2, padding = 1),
            # nn.ReLU(inplace = True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.MaxPool2d(3, stride=2),
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),

            # 6
            # nn.Conv2d(3, 32, kernel_size = 7, stride = 2, padding = 1),
            # nn.ReLU(inplace = True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.MaxPool2d(3, stride=2),
            # nn.Conv2d(32, 32, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),

            # 7
            nn.Conv2d(3, 96, kernel_size = 7, stride = 2, padding = 3),
            nn.ReLU(inplace = True),
            nn.Dropout(p = 0.1, inplace=False),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding= 0),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.1, inplace=False),
            nn.Conv2d(96, 192, kernel_size=3, stride=1, padding= 0),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.1, inplace=False),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding= 0),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.1, inplace=False),

            # 8
            # nn.Conv2d(3, 32, kernel_size = 7, stride = 2, padding = 1),
            # nn.ReLU(inplace = True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.MaxPool2d(3, stride=2),
            # nn.Conv2d(32, 32, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),/home/henryclever/catkin_ws/src/hrl-assistive/hrl_pose_estimation/src/hrl_pose_estimation/create_dataset.py
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),

            # 9
            # nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 1),
            # nn.ReLU(inplace = True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.MaxPool2d(3, stride=2),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),

            # 10
            # nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 1),
            # nn.ReLU(inplace = True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.MaxPool2d(3, stride=2),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),

            # 11
            # nn.Conv2d(3, 64, kernel_size = 5, stride = 2, padding = 1),
            # nn.ReLU(inplace = True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(64, 64, kernel_size=3, stride=2, padding= 1),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),

            # 12
            # nn.Conv2d(3, 64, kernel_size = 5, stride = 2, padding = 1),
            # nn.ReLU(inplace = True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(64, 64, kernel_size=3, stride=2, padding= 1),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),

            # 13
            # nn.Conv2d(3, 64, kernel_size = 5, stride = 2, padding = 1),
            # nn.ReLU(inplace = True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(64, 64, kernel_size=3, stride=2, padding= 1),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
            # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding= 0),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p = 0.1, inplace=False),
        )

        self.CNN_pack2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
            nn.ReLU(inplace = True),
            #nn.Dropout(p = 0.1, inplace=False),
            #nn.MaxPool2d(3, stride=2),
            #nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            #nn.ReLU(inplace=True),
            #nn.Dropout(p=0.1, inplace=False),
            #nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            #nn.ReLU(inplace=True),
            #nn.Dropout(p=0.1, inplace=False),

        )



        self.CNN_pack3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1, inplace=False),

        )

        self.VGG_fc1 = nn.Sequential(
            #nn.Linear(12288, 2048),
            nn.Linear(12288, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_size),
        )


        print 'x'
        self.CNN_fc1 = nn.Sequential(
            # Vanilla
            # nn.Linear(8832, 2048), #4096 for when we only pad the sides by 5 each instead of 10
            # #nn.ReLU(inplace = True),
            # #nn.Linear(5760, 3000),
            # nn.Linear(2048, 2048),
            # #nn.ReLU(inplace = True),
            # nn.Linear(2048, 256),
            # nn.Linear(256, out_size),

            # nn.Linear(8832, out_size),
            # 3
            # nn.Linear(14400, out_size),
            # 4
            # nn.Linear(13824, out_size),
            # 5
            # nn.Linear(36864, out_size),
            # 6
            # nn.Linear(9216, out_size),
            # 7
            nn.Linear(33600, out_size),
            
            # 8
            # nn.Linear(5120, out_size),
            # 9
            # nn.Linear(10240, out_size),
            # 10
            # nn.Linear(5120, out_size),
            # 11
            # nn.Linear(32256, out_size),
            # 12
            # nn.Linear(30720, out_size),
            # 13
            # nn.Linear(15360, out_size),
        )

        print 'Out size:', out_size

        self.GPU = True
        if self.GPU == True:
            # Use for self.GPU
            dtype = torch.cuda.FloatTensor
            print
            '######################### CUDA is available! #############################'
        else:
            # Use for CPU
            dtype = torch.FloatTensor
            print
            '############################## USING CPU #################################'

        if loss_vector_type == 'anglesR' or loss_vector_type == 'anglesDC' or loss_vector_type == 'anglesEU':


            from smpl.smpl_webuser.serialization import load_model

            model_path_f = '/home/henry/git/SMPL_python_v.1.0.0/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
            human_f = load_model(model_path_f)
            self.v_template_f = torch.Tensor(np.array(human_f.v_template)).type(dtype)
            self.shapedirs_f = torch.Tensor(np.array(human_f.shapedirs)).permute(0, 2, 1).type(dtype)
            self.J_regressor_f = np.zeros((human_f.J_regressor.shape)) + human_f.J_regressor
            self.J_regressor_f = torch.Tensor(np.array(self.J_regressor_f).astype(float)).permute(1, 0).type(dtype)


            model_path_m = '/home/henry/git/SMPL_python_v.1.0.0/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
            human_m = load_model(model_path_m)
            self.v_template_m = torch.Tensor(np.array(human_m.v_template)).type(dtype)
            self.shapedirs_m = torch.Tensor(np.array(human_m.shapedirs)).permute(0, 2, 1).type(dtype)
            self.J_regressor_m = np.zeros((human_m.J_regressor.shape)) + human_m.J_regressor
            self.J_regressor_m = torch.Tensor(np.array(self.J_regressor_m).astype(float)).permute(1, 0).type(dtype)

            self.parents = np.array([4294967295, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]).astype(np.int32)


            #print batch_size
            self.N = batch_size
            self.shapedirs_repeat_f = self.shapedirs_f.unsqueeze(0).repeat(self.N, 1, 1, 1).permute(0, 2, 1, 3).unsqueeze(0)
            self.shapedirs_repeat_m = self.shapedirs_m.unsqueeze(0).repeat(self.N, 1, 1, 1).permute(0, 2, 1, 3).unsqueeze(0)
            self.shapedirs_repeat = torch.cat((self.shapedirs_repeat_f, self.shapedirs_repeat_m), 0) #this is 2 x N x B x R x D
            self.B = self.shapedirs_repeat.size()[2] #this is 10
            self.R = self.shapedirs_repeat.size()[3] #this is 6890, or num of verts
            self.D = self.shapedirs_repeat.size()[4] #this is 3, or num dimensions
            self.shapedirs_repeat = self.shapedirs_repeat.permute(1,0,2,3,4).view(self.N, 2, self.B*self.R*self.D)

            self.v_template_repeat_f = self.v_template_f.unsqueeze(0).repeat(self.N, 1, 1).unsqueeze(0)
            self.v_template_repeat_m = self.v_template_m.unsqueeze(0).repeat(self.N, 1, 1).unsqueeze(0)
            self.v_template_repeat = torch.cat((self.v_template_repeat_f, self.v_template_repeat_m), 0)#this is 2 x N x R x D
            self.v_template_repeat = self.v_template_repeat.permute(1,0,2,3).view(self.N, 2, self.R*self.D)

            self.J_regressor_repeat_f = self.J_regressor_f.unsqueeze(0).repeat(self.N, 1, 1).unsqueeze(0)
            self.J_regressor_repeat_m = self.J_regressor_m.unsqueeze(0).repeat(self.N, 1, 1).unsqueeze(0)
            self.J_regressor_repeat = torch.cat((self.J_regressor_repeat_f, self.J_regressor_repeat_m), 0)#this is 2 x N x R x 24
            self.J_regressor_repeat = self.J_regressor_repeat.permute(1,0,2,3).view(self.N, 2, self.R*24)

            self.zeros_cartesian = torch.zeros([self.N, 24]).type(dtype)
            self.ones_cartesian = torch.ones([self.N, 24]).type(dtype)




    def forward_betas(self, images, synth_real_switch, targets, is_training = True):

        '''
        Take a batch of images and run them through the CNN to
        produce a scores for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, out_size) specifying the scores
            for each example and category.
        '''
        #############################################################################
        # TODO: Implement the forward pass. This should take few lines of code.
        #############################################################################
        #print images.size(), 'CNN input size'


        scores = self.CNN_pack1(images)
        scores_size = scores.size()

        scores = scores.view(images.size(0),scores_size[1]*scores_size[2]*scores_size[3])

        scores = self.CNN_fc1(scores)

        scores = torch.mul(scores.clone(), 0.01)

        num_joints = scores.shape[1]/3

        #get it so the initial joints positions all start at the middle of the bed ish
        for fc_output_ct in range(num_joints):
            scores[:, fc_output_ct*3 + 0] = torch.add(scores[:, fc_output_ct*3 + 0], 0.6)
            scores[:, fc_output_ct*3 + 1] = torch.add(scores[:, fc_output_ct*3 + 1], 1.2)
            scores[:, fc_output_ct*3 + 2] = torch.add(scores[:, fc_output_ct*3 + 2], 0.1)

        targets_est_np = scores.clone().data*1000.

        #print scores.shape
        if is_training == False:
            scores = torch.cat((scores[:, 45:48],
                               scores[:, 9:12],
                               scores[:, 57:60],
                               scores[:, 54:57],
                               scores[:, 63:66],
                               scores[:, 60:63],
                               scores[:, 15:18],
                               scores[:, 12:15],
                               scores[:, 24:27],
                               scores[:, 21:24]), dim =1)

            targets_est_reduced_np = scores.clone().data*1000.

        else:
            #print synth_real_switch
            #print targets_est_np.size()
            for joint_num in range(24):
                if joint_num in [0, 1, 2, 6, 9, 10, 11, 12, 13, 14, 16, 17, 22, 23]:
                    targets_est_np[:, joint_num * 3] = targets_est_np[:, joint_num * 3] * synth_real_switch.data
                    targets_est_np[:, joint_num * 3 + 1] = targets_est_np[:, joint_num * 3 + 1] * synth_real_switch.data
                    targets_est_np[:, joint_num * 3 + 2] = targets_est_np[:, joint_num * 3 + 2] * synth_real_switch.data

            targets_est_reduced_np = 0


        #print scores.size(), 'scores fc2'

        #here we want to compute our score as the Euclidean distance between the estimated x,y,z points and the target.
        scores = targets/1000. - scores
        scores = scores.pow(2)

        num_joints = scores.shape[1] / 3
        for fc_output_ct in range(num_joints):
            scores[:, fc_output_ct] = scores[:, fc_output_ct*3 + 0] +  scores[:, fc_output_ct*3 + 1] +  scores[:, fc_output_ct*3 + 2]

        scores = scores[:, 0:num_joints]
        scores = scores.sqrt()

        #print "targets: ", targets[0, :]
        #print "scores: ", scores[0, :]
        #print "targets est np: ", targets_est_np[0, :]

        if is_training == True: scores = torch.mul(torch.add(1.0, torch.mul(1.4, torch.sub(1, synth_real_switch))).unsqueeze(1), scores)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores, targets_est_np, targets_est_reduced_np

