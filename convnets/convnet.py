import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from kinematics_lib import KinematicsLib
import scipy.stats as ss
import torchvision


class CNN(nn.Module):
    def __init__(self, mat_size, out_size, hidden_dim, kernel_size, loss_vector_type, batch_size, split = False):
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

        self.count = 0
        self.split = split


        self.CNN_pack1 = nn.Sequential(

            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1, inplace=False),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1, inplace=False),

        )

        self.CNN_pack2 = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size = 7, stride = 2, padding = 3),
            nn.ReLU(inplace = True),
            nn.Dropout(p = 0.1, inplace=False),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding= 0),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.1, inplace=False),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding= 0),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.1, inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding= 0),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.1, inplace=False),

        )


        if self.split == False:
            self.CNN_fc1 = nn.Sequential(
                nn.Linear(33600, out_size),
            )
        if self.split == True:
            self.CNN_fc1 = nn.Sequential(
                nn.Linear(33600, out_size-10),
            )

        self.CNN_fc2 = nn.Sequential(
            nn.Linear(11200, 10),
        )

        print 'Out size:', out_size

        self.GPU = True
        if self.GPU == True:
            # Use for self.GPU
            dtype = torch.cuda.FloatTensor
            print('######################### CUDA is available! #############################')
        else:
            # Use for CPU
            dtype = torch.FloatTensor
            print('############################## USING CPU #################################')

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




    def forward_direct(self, images, synth_real_switch, targets, is_training = True):

        #pass through convnet for feature extraction
        scores = self.CNN_pack1(images)
        scores_size = scores.size()
        scores = scores.view(images.size(0),scores_size[1]*scores_size[2]*scores_size[3])

        #collect in linear layer
        scores = self.CNN_fc1(scores)


        #scale things so the model starts close to the home position. Has nothing to do with weighting.
        scores = torch.mul(scores.clone(), 0.01)

        num_joints = scores.shape[1]/3

        #get it so the initial joints positions all start at the middle of the bed ish
        for fc_output_ct in range(num_joints):
            scores[:, fc_output_ct*3 + 0] = torch.add(scores[:, fc_output_ct*3 + 0], 0.6)
            scores[:, fc_output_ct*3 + 1] = torch.add(scores[:, fc_output_ct*3 + 1], 1.2)
            scores[:, fc_output_ct*3 + 2] = torch.add(scores[:, fc_output_ct*3 + 2], 0.1)

        #print scores.size(), 'SCORE SIZE', targets.size()


        #print scores.shape
        if is_training == True:

            targets_est_np = scores.clone().data * 1000.
            #print synth_real_switch
            #print targets_est_np.size()
            for joint_num in range(24):
                if joint_num in [0, 1, 2, 6, 9, 10, 11, 12, 13, 14, 16, 17, 22, 23]:
                    targets_est_np[:, joint_num * 3] = targets_est_np[:, joint_num * 3] * synth_real_switch.data
                    targets_est_np[:, joint_num * 3 + 1] = targets_est_np[:, joint_num * 3 + 1] * synth_real_switch.data
                    targets_est_np[:, joint_num * 3 + 2] = targets_est_np[:, joint_num * 3 + 2] * synth_real_switch.data

            targets_est_reduced_np = 0

            # here we want to compute our score as the Euclidean distance between the estimated x,y,z points and the target.
            scores = targets / 1000. - scores
            scores = scores.pow(2)

            num_joints = scores.shape[1] / 3
            for fc_output_ct in range(num_joints):
                scores[:, fc_output_ct] = scores[:, fc_output_ct * 3 + 0] + scores[:, fc_output_ct * 3 + 1] + scores[:,fc_output_ct * 3 + 2]



            for joint_num in range(24):
                #print scores[:, 10+joint_num].size(), 'score size'
                #print synth_real_switch.size(), 'switch size'
                if joint_num in [0, 1, 2, 6, 9, 10, 11, 12, 13, 14, 16, 17, 22, 23]: #torso is 3 but forget training it
                    scores[:, 10+joint_num] = torch.mul(synth_real_switch,
                                                        (scores[:, joint_num*3 + 0] +
                                                         scores[:, joint_num*3 + 1] +
                                                         scores[:, joint_num*3 + 2]).sqrt())

                else:
                    scores[:, 10+joint_num] = (scores[:, 106+joint_num*3] +
                                                 scores[:, 107+joint_num*3] +
                                                 scores[:, 108+joint_num*3]).sqrt()



            scores = scores[:, 0:24]



        else:

            targets_est_np = scores.clone().data*1000.

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

        print scores.size(), scores[0, :]

        return scores, targets_est_np, targets_est_reduced_np



    def forward_kinematic_angles(self, images, gender_switch, synth_real_switch, targets=None, is_training = True, betas=None, angles_gt = None, root_shift = None):

        scores_cnn = self.CNN_pack1(images)
        scores_size = scores_cnn.size()
        # print scores_size, 'scores conv1'

        # ''' # NOTE: Uncomment
        # This combines the height, width, and filters into a single dimension
        scores_cnn = scores_cnn.view(images.size(0),scores_size[1] *scores_size[2]*scores_size[3] )
        #print 'size for fc layer:', scores_cnn.size()


        scores = self.CNN_fc1(scores_cnn) #this is N x 229: betas, root shift, Rotation matrices


        #scale things so the model starts close to the home position. Has nothing to do with weighting.
        scores[:, 0:10] = torch.mul(scores[:, 0:10].clone(), 0.1)
        scores[:, 10:] = torch.mul(scores[:, 10:].clone(), 0.01)

        scores[:, 10] = torch.add(scores[:, 10].clone(), 0.6)
        scores[:, 11] = torch.add(scores[:, 11].clone(), 1.2)
        scores[:, 12] = torch.add(scores[:, 12].clone(), 0.1)

        #print scores[34, :]

        test_ground_truth = False #can only use True when the dataset is entirely synthetic

        if test_ground_truth == False:
            betas_est = scores[:, 0:10].clone().detach() #make sure to detach so the gradient flow of joints doesn't corrupt the betas
            root_shift_est = scores[:, 10:13].clone()

            if self.loss_vector_type == 'anglesDC':
                Rs_est = self.batch_rodrigues(scores[:, 13:85].view(-1, 24, 3).clone()).view(-1, 24, 3, 3)
            elif self.loss_vector_type == 'anglesEU':
                Rs_est = self.batch_euler_to_R(scores[:, 13:85].view(-1, 24, 3).clone()).view(-1, 24, 3, 3)

        else:
            #print betas[13, :], 'betas'
            betas_est = betas
            scores[:, 0:10] = betas
            root_shift_est = root_shift

            if self.loss_vector_type == 'anglesDC':
                Rs_est = self.batch_rodrigues(angles_gt.view(-1, 24, 3)).view(-1, 24, 3, 3)
            elif self.loss_vector_type == 'anglesEU':
                Rs_est = self.batch_euler_to_R(angles_gt.view(-1, 24, 3)).view(-1, 24, 3, 3)


        gender_switch = gender_switch.unsqueeze(1)
        current_batch_size = gender_switch.size()[0]


        shapedirs = torch.bmm(gender_switch, self.shapedirs_repeat[0:current_batch_size, :, :])\
                         .view(current_batch_size, self.B, self.R*self.D)

        betas_shapedirs_mult = torch.bmm(betas_est.unsqueeze(1), shapedirs)\
                                    .squeeze(1)\
                                    .view(current_batch_size, self.R, self.D)

        v_template = torch.bmm(gender_switch, self.v_template_repeat[0:current_batch_size, :, :])\
                          .view(current_batch_size, self.R, self.D)

        v_shaped = betas_shapedirs_mult + v_template

        J_regressor_repeat = torch.bmm(gender_switch, self.J_regressor_repeat[0:current_batch_size, :, :])\
                                  .view(current_batch_size, self. R, 24)

        Jx = torch.bmm(v_shaped[:, :, 0].unsqueeze(1), J_regressor_repeat).squeeze(1)
        Jy = torch.bmm(v_shaped[:, :, 1].unsqueeze(1), J_regressor_repeat).squeeze(1)
        Jz = torch.bmm(v_shaped[:, :, 2].unsqueeze(1), J_regressor_repeat).squeeze(1)


        J_est = torch.stack([Jx, Jy, Jz], dim=2)  # these are the joint locations with home pose (pose is 0 degree on all angles)
        J_est = J_est - J_est[:, 0:1, :] + root_shift_est.unsqueeze(1)

        targets_est, A_est = self.batch_global_rigid_transformation(Rs_est, J_est, self.parents, rotate_base=False)

        targets_est = targets_est.contiguous().view(-1, 72)

        targets_est_np = targets_est.data*1000. #after it comes out of the forward kinematics


        betas_est_np = betas_est.data*1000.

        scores = scores.unsqueeze(0)
        scores = scores.unsqueeze(0)
        scores = F.pad(scores, (0, 100, 0, 0))
        scores = scores.squeeze(0)
        scores = scores.squeeze(0)

        #print targets_est.size()
        #print scores.size()

        #tweak this to change the lengths vector
        scores[:, 34:106] = torch.mul(targets_est[:, 0:72], 1.)

        #print scores[13, 34:106]



        if is_training == True:

            #print targets_est_np.size()
            for joint_num in range(24):
                if joint_num in [0, 1, 2, 6, 9, 10, 11, 12, 13, 14, 16, 17, 22, 23]:
                    targets_est_np[:, joint_num * 3] = targets_est_np[:, joint_num * 3] * synth_real_switch.data
                    targets_est_np[:, joint_num * 3 + 1] = targets_est_np[:, joint_num * 3 + 1] * synth_real_switch.data
                    targets_est_np[:, joint_num * 3 + 2] = targets_est_np[:, joint_num * 3 + 2] * synth_real_switch.data


            scores[:, 0:10] = torch.mul(synth_real_switch.unsqueeze(1),
                                        torch.sub(scores[:, 0:10], betas))*.2

            scores[:, 34:106] = targets[:, 0:72]/1000 - scores[:, 34:106]
            scores[:, 106:178] = ((scores[:, 34:106].clone())*1.).pow(2)

            #print scores[13, 106:178]

            for joint_num in range(24):
                #print scores[:, 10+joint_num].size(), 'score size'
                #print synth_real_switch.size(), 'switch size'
                if joint_num in [0, 1, 2, 6, 9, 10, 11, 12, 13, 14, 16, 17, 22, 23]: #torso is 3 but forget training it
                    scores[:, 10+joint_num] = torch.mul(synth_real_switch,
                                                        (scores[:, 106+joint_num*3] +
                                                         scores[:, 107+joint_num*3] +
                                                         scores[:, 108+joint_num*3]).sqrt())

                else:
                    scores[:, 10+joint_num] = (scores[:, 106+joint_num*3] +
                                                 scores[:, 107+joint_num*3] +
                                                 scores[:, 108+joint_num*3]).sqrt()

                    #print scores[:, 10+joint_num], 'score size'
                    #print synth_real_switch, 'switch size'

            scores = scores.unsqueeze(0)
            scores = scores.unsqueeze(0)
            scores = F.pad(scores, (0, -151, 0, 0))
            scores = scores.squeeze(0)
            scores = scores.squeeze(0)


            #here multiply by 24/10 when you are regressing to real data so it balances with the synthetic data
            scores = torch.mul(torch.add(1.0, torch.mul(1.4, torch.sub(1, synth_real_switch))).unsqueeze(1), scores)

            #print scores[7, :]


            targets_est_reduced_np = 0

        else:
            if self.GPU == True:
                targets_est_reduced = torch.empty(targets_est.size()[0], 30, dtype=torch.float).cuda()
            else:    
                targets_est_reduced = torch.empty(targets_est.size()[0], 30, dtype=torch.float)
            #scores[:, 80] = torch.add(scores[:, 80], 0.2)
            #scores[:, 81] = torch.add(scores[:, 81], 0.2)
            targets_est_reduced[:, 0:3] = scores[:, 79:82] #head 34 + 3*15 = 79
            targets_est_reduced[:, 3:6] = scores[:, 43:46] #torso 34 + 3*3 = 45
            targets_est_reduced[:, 6:9] = scores[:, 91:94] #right elbow 34 + 19*3 = 91
            targets_est_reduced[:, 9:12] = scores[:, 88:91] #left elbow 34 + 18*3 = 88
            targets_est_reduced[:, 12:15] = scores[:, 97:100] #right wrist 34 + 21*3 = 97
            targets_est_reduced[:, 15:18] = scores[:, 94:97] #left wrist 34 + 20*3 = 94
            targets_est_reduced[:, 18:21] = scores[:, 49:52] #right knee 34 + 3*5
            targets_est_reduced[:, 21:24] = scores[:, 46:49] #left knee 34 + 3*4
            targets_est_reduced[:, 24:27] = scores[:, 58:61] #right ankle 34 + 3*8
            targets_est_reduced[:, 27:30] = scores[:, 55:58] #left ankle 34 + 3*7

            targets_est_reduced_np = targets_est_reduced.data*1000.

            #print(targets.size(), targets[0, :])

            scores[:, 10:40] = (targets/1000. - targets_est_reduced[:, 0:30]).pow(2)

            #print(scores.size(), scores[0, 10:40])


            for joint_num in range(10):
                scores[:, joint_num] = (scores[:, 10+joint_num*3] + scores[:, 11+joint_num*3] + scores[:, 12+joint_num*3]).sqrt()


            scores = scores.unsqueeze(0)
            scores = scores.unsqueeze(0)
            scores = F.pad(scores, (0, -175, 0, 0))
            scores = scores.squeeze(0)
            scores = scores.squeeze(0)


            scores = torch.mul(2.4, scores)

        #print(scores[0, :], scores.size(), np.sum(np.abs(scores.data.numpy())))

        return  scores, targets_est_np, targets_est_reduced_np, betas_est_np





    def forward_kinematic_R(self, images, gender_switch, targets=None, is_training = True, betas=None, angles_gt = None, root_shift = None):

        scores_cnn = self.CNN_pack1(images)
        scores_size = scores_cnn.size()
        # print scores_size, 'scores conv1'

        # ''' # NOTE: Uncomment
        # This combines the height, width, and filters into a single dimension
        scores_cnn = scores_cnn.view(images.size(0),scores_size[1] *scores_size[2]*scores_size[3] )
        #print 'size for fc layer:', scores_cnn.size()


        scores = self.CNN_fc1(scores_cnn) #this is N x 229: betas, root shift, Rotation matrices

        scores[:, 0:10] = torch.mul(scores[:, 0:10].clone(), 0.1)
        scores[:, 10:] = torch.mul(scores[:, 10:].clone(), 0.01)


        scores[:, 10] = torch.add(scores[:, 10].clone(), 0.6)
        scores[:, 11] = torch.add(scores[:, 11].clone(), 1.2)
        scores[:, 12] = torch.add(scores[:, 12].clone(), 0.1)

        #print scores[34, :]

        for rotation_matrix_num in range(24):
            scores[:, 13+rotation_matrix_num*9+0] += 1
            scores[:, 13+rotation_matrix_num*9+4] += 1
            scores[:, 13+rotation_matrix_num*9+8] += 1


        test_ground_truth = False

        if test_ground_truth == False:
            betas_est = scores[:, 0:10].clone()
            root_shift_est = scores[:, 10:13].clone()
            Rs_est = scores[:, 13:229].view(-1, 24, 3, 3).clone()
        else:
            #print betas[13, :], 'betas'
            betas_est = betas
            scores[:, 0:10] = betas
            root_shift_est = root_shift
            Rs_est = self.batch_rodrigues(angles_gt.view(-1, 24, 3)).view(-1, 24, 3, 3)



        gender_switch = gender_switch.unsqueeze(1)
        current_batch_size = gender_switch.size()[0]


        shapedirs = torch.bmm(gender_switch, self.shapedirs_repeat[0:current_batch_size, :, :])\
                         .view(current_batch_size, self.B, self.R*self.D)

        betas_shapedirs_mult = torch.bmm(betas_est.unsqueeze(1), shapedirs) \
                                    .squeeze(1)\
                                    .view(current_batch_size, self.R, self.D) #NxRxD

        v_template = torch.bmm(gender_switch, self.v_template_repeat[0:current_batch_size, :, :])\
                          .view(current_batch_size, self.R, self.D)

        v_shaped = betas_shapedirs_mult + v_template

        J_regressor_repeat = torch.bmm(gender_switch, self.J_regressor_repeat[0:current_batch_size, :, :])\
                                  .view(current_batch_size, self. R, 24)

        Jx = torch.bmm(v_shaped[:, :, 0].unsqueeze(1), J_regressor_repeat).squeeze(1)
        Jy = torch.bmm(v_shaped[:, :, 1].unsqueeze(1), J_regressor_repeat).squeeze(1)
        Jz = torch.bmm(v_shaped[:, :, 2].unsqueeze(1), J_regressor_repeat).squeeze(1)



        #v_shaped = torch.matmul(betas_est, self.shapedirs_f).permute(1, 0, 2) + self.v_template_f

        #Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor_f)
        #Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor_f)
        #Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor_f)


        J_est = torch.stack([Jx, Jy, Jz], dim=2)  # these are the joint locations with home pose (pose is 0 degree on all angles)
        J_est = J_est - J_est[:, 0:1, :] + root_shift_est.unsqueeze(1)

        targets_est, A_est = self.batch_global_rigid_transformation(Rs_est, J_est, self.parents, rotate_base=False)

        targets_est = targets_est.contiguous().view(-1, 72)

        targets_est_np = targets_est.data*1000. #after it comes out of the forward kinematics
        betas_est_np = betas_est.data*1000.

        #tweak this to change the lengths vector
        scores[:, 34:106] = torch.mul(targets_est[:, 0:72], 1.)

        #print scores[13, 34:106]

        if is_training == True:
            scores[:, 34:106] = targets[:, 0:72]/1000 - scores[:, 34:106]
            scores[:, 106:178] = ((scores[:, 34:106].clone())*1.).pow(2)

            #print scores[13, 106:178]

            for joint_num in range(24):
                scores[:, 10+joint_num] = (scores[:, 106+joint_num*3] + scores[:, 107+joint_num*3] + scores[:, 108+joint_num*3]).sqrt()

            scores = scores.unsqueeze(0)
            scores = scores.unsqueeze(0)
            scores = F.pad(scores, (0, -195, 0, 0))
            scores = scores.squeeze(0)
            scores = scores.squeeze(0)

            targets_est_reduced_np = 0
        else:
            targets_est_reduced = torch.empty(targets_est.size()[0], 30, dtype=torch.float)
            targets_est_reduced[:, 0:3] = scores[:, 79:82] #head 34 + 3*15 = 79
            targets_est_reduced[:, 3:6] = scores[:, 43:46] #torso 34 + 3*3 = 45
            targets_est_reduced[:, 6:9] = scores[:, 91:94] #right elbow 34 + 19*3 = 91
            targets_est_reduced[:, 9:12] = scores[:, 88:91] #left elbow 34 + 18*3 = 88
            targets_est_reduced[:, 12:15] = scores[:, 97:100] #right wrist 34 + 21*3 = 97
            targets_est_reduced[:, 15:18] = scores[:, 94:97] #left wrist 34 + 20*3 = 94
            targets_est_reduced[:, 18:21] = scores[:, 49:52] #right knee 34 + 3*5
            targets_est_reduced[:, 21:24] = scores[:, 46:49] #left knee 34 + 3*4
            targets_est_reduced[:, 24:27] = scores[:, 58:61] #right ankle 34 + 3*8
            targets_est_reduced[:, 27:30] = scores[:, 55:58] #left ankle 34 + 3*7

            targets_est_reduced_np = targets_est_reduced.data*1000.

            scores[:, 10:40] = targets_est_reduced[:, 0:30].pow(2)


            for joint_num in range(10):
                    scores[:, joint_num] = (scores[:, 10+joint_num*3] + scores[:, 11+joint_num*3] + scores[:, 12+joint_num*3]).sqrt()


            scores = scores.unsqueeze(0)
            scores = scores.unsqueeze(0)
            scores = F.pad(scores, (0, -219, 0, 0))
            scores = scores.squeeze(0)
            scores = scores.squeeze(0)

        return  scores, targets_est_np, targets_est_reduced_np, betas_est_np


    def batch_rodrigues(self, theta):
        # theta N x 3
        batch_size = theta.shape[0]

        #print theta[0, :], 'THETA'
        l1norm = torch.norm(theta + 1e-8, p=2, dim=2)
        angle = torch.unsqueeze(l1norm, -1)
        #print angle[0, :], 'ANGLE'
        normalized = torch.div(theta, angle)
        #print normalized[0, :], 'NORM'
        angle = angle * 0.5
        v_cos = torch.cos(angle)
        v_sin = torch.sin(angle)
        quat = torch.cat([v_cos, v_sin * normalized], dim=2)
        #print quat[0, :]

        """Convert quaternion coefficients to rotation matrix.
        Args:
            quat: size = [B, 4] 4 <===>(w, x, y, z)
        Returns:
            Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
        """
        norm_quat = quat
        norm_quat = norm_quat / norm_quat.norm(p=2, dim=2, keepdim=True)

        #print norm_quat.shape

        w, x, y, z = norm_quat[:, :, 0], norm_quat[:, :, 1], norm_quat[:, :, 2], norm_quat[:, :, 3]
#
        w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z


        rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                              2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                              2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=2)

        #print "got R"
        return rotMat

    def batch_euler_to_R(self, theta):
        batch_size_current = theta.size()[0]

        cosx = torch.cos(theta[:, :, 0])
        sinx = torch.sin(theta[:, :, 0])
        cosy = torch.cos(theta[:, :, 1])
        siny = torch.sin(theta[:, :, 1])
        cosz = torch.cos(theta[:, :, 2])
        sinz = torch.sin(theta[:, :, 2])

        b_zeros = self.zeros_cartesian[:batch_size_current, :]
        b_ones = self.ones_cartesian[:batch_size_current, :]

        R_x = torch.stack([b_ones, b_zeros, b_zeros,
                           b_zeros, cosx, -sinx,
                           b_zeros, sinx, cosx], dim=2)\
                    .view(batch_size_current, 24, 3, 3)


        R_y = torch.stack([cosy, b_zeros, siny,
                           b_zeros, b_ones, b_zeros,
                           -siny, b_zeros, cosy], dim=2)\
                    .view(batch_size_current, 24, 3, 3)


        R_z = torch.stack([cosz, -sinz, b_zeros,
                           sinz, cosz, b_zeros,
                           b_zeros, b_zeros, b_ones], dim=2)\
                    .view(batch_size_current, 24, 3, 3)

        R_x = R_x.view(batch_size_current*24, 3, 3)
        R_y = R_y.view(batch_size_current*24, 3, 3)
        R_z = R_z.view(batch_size_current*24, 3, 3)

        R = torch.bmm(torch.bmm(R_z, R_y), R_x).view(batch_size_current, 24, 3, 3)
        return R



    def batch_global_rigid_transformation(self, Rs, Js, parent, rotate_base=False):
        N = Rs.shape[0]
        if rotate_base:
            np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float)
            np_rot_x = np.reshape(np.tile(np_rot_x, [N, 1]), [N, 3, 3])
            if self.GPU == True:
                rot_x = Variable(torch.from_numpy(np_rot_x).float()).cuda()
            else:
                rot_x = Variable(torch.from_numpy(np_rot_x).float())
            root_rotation = torch.matmul(Rs[:, 0, :, :], rot_x)
        else:
            root_rotation = Rs[:, 0, :, :]
        Js = torch.unsqueeze(Js, -1)

        def make_A(R, t):
            R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])
            if self.GPU == True:
                t_homo = torch.cat([t, Variable(torch.ones(N, 1, 1)).cuda()], dim=1)
            else:
                t_homo = torch.cat([t, Variable(torch.ones(N, 1, 1))], dim=1)
            return torch.cat([R_homo, t_homo], 2)

        A0 = make_A(root_rotation, Js[:, 0])
        results = [A0]

        for i in range(1, parent.shape[0]):
            j_here = Js[:, i] - Js[:, parent[i]]
            A_here = make_A(Rs[:, i], j_here)
            res_here = torch.matmul(results[parent[i]], A_here)
            results.append(res_here)

        results = torch.stack(results, dim=1)

        new_J = results[:, :, :3, 3]
        if self.GPU == True:
            Js_w0 = torch.cat([Js, Variable(torch.zeros(N, 24, 1, 1)).cuda()], dim=2)
        else:
            Js_w0 = torch.cat([Js, Variable(torch.zeros(N, 24, 1, 1))], dim=2)
        init_bone = torch.matmul(results, Js_w0)
        init_bone = F.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
        A = results - init_bone

        return new_J, A
