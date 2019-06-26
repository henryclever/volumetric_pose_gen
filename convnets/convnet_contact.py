import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from kinematics_lib import KinematicsLib
from mesh_depth_lib import MeshDepthLib
import scipy.stats as ss
import torchvision
import resnet
import time

class CNN(nn.Module):
    def __init__(self, mat_size, out_size, hidden_dim, kernel_size, loss_vector_type, batch_size, split = False, filepath = '/home/henry/'):
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

            nn.Conv2d(6, 256, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1, inplace=False),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1, inplace=False),

        )


        self.CNN_pack2 = nn.Sequential(

            nn.Conv2d(6, 32, kernel_size = 7, stride = 2, padding = 3),
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
                nn.Linear(89600, 1000),
                nn.Linear(1000, out_size),
            )
            #self.resnet = resnet.resnet34(pretrained=True, num_classes=1000)
        if self.split == True:
            self.CNN_fc1 = nn.Sequential(
                nn.Linear(89600, 1000),
                nn.Linear(1000, out_size-10),
            )
            #self.resnet = resnet.resnet34(pretrained=True, output_size=out_size-10, num_classes=out_size-10)

        self.CNN_fc2 = nn.Sequential(
            nn.Linear(11200, 10),
        )

        print 'Out size:', out_size

        if torch.cuda.is_available():
            self.GPU = True
            # Use for self.GPU
            dtype = torch.cuda.FloatTensor
            dtypeInt = torch.cuda.IntTensor
            print('######################### CUDA is available! #############################')
        else:
            self.GPU = False
            # Use for CPU
            dtype = torch.FloatTensor
            dtypeInt = torch.IntTensor
            print('############################## USING CPU #################################')
        self.dtype = dtype
        self.dtypeInt = dtypeInt

        self.meshDepthLib = MeshDepthLib(loss_vector_type, filepath, batch_size, verts_type = "all")

    def forward_kinematic_angles_realtime(self, images):
        scores_cnn = self.CNN_pack1(images)
        scores_size = scores_cnn.size()
        # print scores_size, 'scores conv1'

        # ''' # NOTE: Uncomment
        # This combines the height, width, and filters into a single dimension
        scores_cnn = scores_cnn.view(images.size(0), scores_size[1] * scores_size[2] * scores_size[3])
        # print 'size for fc layer:', scores_cnn.size()

        scores = self.CNN_fc1(scores_cnn)  # this is N x 229: betas, root shift, Rotation matrices

        # weight the outputs, which are already centered around 0. First make them uniformly smaller than the direct output, which is too large.
        scores = torch.mul(scores.clone(), 0.01)

        # normalize the output of the network based on the range of the parameters
        if self.GPU == True:
            output_norm = 10 * [6.0] + [0.91, 1.98, 0.15] + list(
                torch.abs(self.meshDepthLib.bounds.view(72, 2)[:, 1] - self.meshDepthLib.bounds.view(72, 2)[:, 0]).cpu().numpy())
        else:
            output_norm = 10 * [6.0] + [0.91, 1.98, 0.15] + list(
                torch.abs(self.meshDepthLib.bounds.view(72, 2)[:, 1] - self.meshDepthLib.bounds.view(72, 2)[:, 0]).numpy())
        for i in range(85):
            scores[:, i] = torch.mul(scores[:, i].clone(), output_norm[i])

        # add a factor so the model starts close to the home position. Has nothing to do with weighting.
        scores[:, 10] = torch.add(scores[:, 10].clone(), 0.6)
        scores[:, 11] = torch.add(scores[:, 11].clone(), 1.2)
        scores[:, 12] = torch.add(scores[:, 12].clone(), 0.1)


        betas_est = scores[:,0:10].clone().detach().cpu().numpy()  # make sure to detach so the gradient flow of joints doesn't corrupt the betas
        root_shift_est = scores[:, 10:13].clone().detach().cpu().numpy()

        # normalize for tan activation function
        scores[:, 13:85] -= torch.mean(self.meshDepthLib.bounds[0:72, 0:2], dim=1)
        scores[:, 13:85] *= (2. / torch.abs(self.meshDepthLib.bounds[0:72, 0] - self.meshDepthLib.bounds[0:72, 1]))
        scores[:, 13:85] = scores[:, 13:85].tanh()
        scores[:, 13:85] /= (2. / torch.abs(self.meshDepthLib.bounds[0:72, 0] - self.meshDepthLib.bounds[0:72, 1]))
        scores[:, 13:85] += torch.mean(self.meshDepthLib.bounds[0:72, 0:2], dim=1)

        if self.loss_vector_type == 'anglesDC':

            angles_est = scores[:, 13:85].clone().detach().cpu().numpy()

        elif self.loss_vector_type == 'anglesEU':

            angles_est = KinematicsLib().batch_dir_cos_angles_from_euler_angles(scores[:, 13:85].view(-1, 24, 3).clone(), self.meshDepthLib.zeros_cartesian, self.meshDepthLib.ones_cartesian)

        return np.squeeze(betas_est), np.squeeze(root_shift_est), np.squeeze(angles_est)





    def forward_kinematic_angles(self, images, gender_switch, synth_real_switch, targets=None, is_training = True, betas=None, angles_gt = None, root_shift = None, reg_angles = False):
        #self.GPU = False
        #self.dtype = torch.FloatTensor

        #print torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated(), torch.cuda.memory_cached(), "p7b"

        scores_cnn = self.CNN_pack1(images)

        try:
             x = self.meshDepthLib.bounds
             print "GOT HERE!"
        except:
            self.meshDepthLib = MeshDepthLib(loss_vector_type=self.loss_vector_type, filepath='/home/henry/', batch_size=images.size(0), verts_type = "all")

        #print torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated(), torch.cuda.memory_cached(), "p7c"

        scores_size = scores_cnn.size()

        # ''' # NOTE: Uncomment
        # This combines the height, width, and filters into a single dimension
        scores_cnn = scores_cnn.view(images.size(0),scores_size[1] *scores_size[2]*scores_size[3])
        #print 'size for fc layer:', scores_cnn.size()


        scores = self.CNN_fc1(scores_cnn) #this is N x 229: betas, root shift, Rotation matrices

        #print scores.size()
        #print scores[0, :]
        #scores = self.resnet(images)
        #print scores.size()
        #print scores[0, :]


        #print torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated(), torch.cuda.memory_cached(), "p7d"

        #weight the outputs, which are already centered around 0. First make them uniformly smaller than the direct output, which is too large. 
        scores = torch.mul(scores.clone(), 0.01)

        #normalize the output of the network based on the range of the parameters
        if self.GPU == True:
            output_norm = 10*[6.0] + [0.91, 1.98, 0.15] + list(torch.abs(self.meshDepthLib.bounds.view(72,2)[:, 1] - self.meshDepthLib.bounds.view(72,2)[:, 0]).cpu().numpy())
        else:
            output_norm = 10 * [6.0] + [0.91, 1.98, 0.15] + list(torch.abs(self.meshDepthLib.bounds.view(72, 2)[:, 1] - self.meshDepthLib.bounds.view(72, 2)[:, 0]).numpy())
        for i in range(85):
            scores[:, i] = torch.mul(scores[:, i].clone(), output_norm[i])


        #add a factor so the model starts close to the home position. Has nothing to do with weighting.
        scores[:, 10] = torch.add(scores[:, 10].clone(), 0.6)
        scores[:, 11] = torch.add(scores[:, 11].clone(), 1.2)
        scores[:, 12] = torch.add(scores[:, 12].clone(), 0.1)

        #print scores[34, :]

        if reg_angles == True:
            add_idx = 72
        else:
            add_idx = 0


        test_ground_truth = False #can only use True when the dataset is entirely synthetic AND when we use anglesDC


        if test_ground_truth == False or is_training == False:
            betas_est = scores[:, 0:10].clone()#.detach() #make sure to detach so the gradient flow of joints doesn't corrupt the betas
            root_shift_est = scores[:, 10:13].clone()

            # normalize for tan activation function
            scores[:, 13:85] -= torch.mean(self.meshDepthLib.bounds[0:72, 0:2], dim=1)
            scores[:, 13:85] *= (2. / torch.abs(self.meshDepthLib.bounds[0:72, 0] - self.meshDepthLib.bounds[0:72, 1]))
            scores[:, 13:85] = scores[:, 13:85].tanh()
            scores[:, 13:85] /= (2. / torch.abs(self.meshDepthLib.bounds[0:72, 0] - self.meshDepthLib.bounds[0:72, 1]))
            scores[:, 13:85] += torch.mean(self.meshDepthLib.bounds[0:72, 0:2], dim=1)

            if self.loss_vector_type == 'anglesDC':

                Rs_est = KinematicsLib().batch_rodrigues(scores[:, 13:85].view(-1, 24, 3).clone()).view(-1, 24, 3, 3)

            elif self.loss_vector_type == 'anglesEU':

                Rs_est = KinematicsLib().batch_euler_to_R(scores[:, 13:85].view(-1, 24, 3).clone(), self.meshDepthLib.zeros_cartesian, self.meshDepthLib.ones_cartesian).view(-1, 24, 3, 3)

        else:
            #print betas[13, :], 'betas'
            betas_est = betas
            scores[:, 0:10] = betas.clone()
            scores[:, 13:85] = angles_gt.clone()
            root_shift_est = root_shift
            #print root_shift[0, :], "root shift"

            if self.loss_vector_type == 'anglesDC':

                #normalize for tan activation function
                scores[:, 13:85] -= torch.mean(self.meshDepthLib.bounds[0:72,0:2], dim = 1)
                scores[:, 13:85] *= (2. / torch.abs(self.meshDepthLib.bounds[0:72, 0] - self.meshDepthLib.bounds[0:72, 1]))
                #scores[:, 13:85] = scores[:, 13:85].tanh()
                scores[:, 13:85] /= (2. / torch.abs(self.meshDepthLib.bounds[0:72, 0] - self.meshDepthLib.bounds[0:72, 1]))
                scores[:, 13:85] += torch.mean(self.meshDepthLib.bounds[0:72,0:2], dim = 1)


                Rs_est = KinematicsLib().batch_rodrigues(scores[:, 13:85].view(-1, 24, 3).clone()).view(-1, 24, 3, 3)
            elif self.loss_vector_type == 'anglesEU':

                #convert angles DC to EU
                scores[:, 13:85] = KinematicsLib().batch_euler_angles_from_dir_cos_angles(scores[:, 13:85].view(-1, 24, 3).clone()).contiguous().view(-1, 72)

                #normalize for tan activation function
                scores[:, 13:85] -= torch.mean(self.meshDepthLib.bounds[0:72,0:2], dim = 1)
                scores[:, 13:85] *= (2. / torch.abs(self.meshDepthLib.bounds[0:72, 0] - self.meshDepthLib.bounds[0:72, 1]))
                scores[:, 13:85] = scores[:, 13:85].tanh()
                scores[:, 13:85] /= (2. / torch.abs(self.meshDepthLib.bounds[0:72, 0] - self.meshDepthLib.bounds[0:72, 1]))
                scores[:, 13:85] += torch.mean(self.meshDepthLib.bounds[0:72,0:2], dim = 1)

                Rs_est = KinematicsLib().batch_euler_to_R(scores[:, 13:85].view(-1, 24, 3).clone(), self.meshDepthLib.zeros_cartesian, self.meshDepthLib.ones_cartesian).view(-1, 24, 3, 3)

        #print Rs_est[0, :]

        gender_switch = gender_switch.unsqueeze(1)
        current_batch_size = gender_switch.size()[0]

        #print root_shift_est[0, :], 'root shift'

        #break things up into sub batches and pass through the mesh
        num_normal_sub_batches = current_batch_size/self.meshDepthLib.N
        if current_batch_size%self.meshDepthLib.N != 0:
            sub_batch_incr_list = num_normal_sub_batches*[self.meshDepthLib.N] + [current_batch_size%self.meshDepthLib.N]
        else:
            sub_batch_incr_list = num_normal_sub_batches*[self.meshDepthLib.N]
        start_incr, end_incr = 0, 0
        for sub_batch_incr in sub_batch_incr_list:
            end_incr += sub_batch_incr
            verts_sub, J_est_sub, targets_est_sub = self.meshDepthLib.compute_tensor_mesh(gender_switch, betas_est, Rs_est, root_shift_est, start_incr, end_incr)
            if start_incr == 0:
                verts = verts_sub.clone()
                J_est = J_est_sub.clone()
                targets_est = targets_est_sub.clone()
            else:
                verts = torch.cat((verts, verts_sub), dim = 0)
                J_est  = torch.cat((J_est, J_est_sub), dim = 0)
                targets_est = torch.cat((targets_est, targets_est_sub), dim = 0)
            start_incr += sub_batch_incr



        #print torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated(), torch.cuda.memory_cached(), "p9"
        #print verts[0, 0:10, :]

        bed_angle_batch = torch.mean(images[:, 2, 1:3, 0], dim = 1)

        mesh_matrix_batch, contact_matrix_batch = self.meshDepthLib.compute_depth_contact_planes(verts, bed_angle_batch)

        #print torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated(), torch.cuda.memory_cached(), "p10"

        verts_red = torch.stack([verts[:, 1325, :],
                                verts[:, 336, :],  # head
                                verts[:, 1032, :],  # l knee
                                verts[:, 4515, :],  # r knee
                                verts[:, 1374, :],  # l ankle
                                verts[:, 4848, :],  # r ankle
                                verts[:, 1739, :],  # l elbow
                                verts[:, 5209, :],  # r elbow
                                verts[:, 1960, :],  # l wrist
                                verts[:, 5423, :]]).permute(1, 0, 2)  # r wrist

        verts_offset = torch.Tensor(verts_red.clone().detach().cpu().numpy()).type(self.dtype)
        targets_est_detached = torch.Tensor(targets_est.clone().detach().cpu().numpy()).type(self.dtype)
        synth_joint_addressed = [3, 15, 4, 5, 7, 8, 18, 19, 20, 21]

        for real_joint in range(10):
            verts_offset[:, real_joint, :] = verts_offset[:, real_joint, :] - targets_est_detached[:, synth_joint_addressed[real_joint], :]


        #here we need to the ground truth to make it a surface point for the mocap markers
        if is_training == True:
            synth_real_switch_repeated = synth_real_switch.unsqueeze(1).repeat(1, 3)
            for real_joint in range(10):
                targets_est[:, synth_joint_addressed[real_joint], :] = synth_real_switch_repeated * targets_est[:, synth_joint_addressed[real_joint], :].clone() \
                                       + torch.add(-synth_real_switch_repeated, 1) * (targets_est[:, synth_joint_addressed[real_joint], :].clone() + verts_offset[:, real_joint, :])

        else:
            for real_joint in range(10):
                targets_est[:, synth_joint_addressed[real_joint], :] = targets_est[:, synth_joint_addressed[real_joint], :] + verts_offset[:, real_joint, :]





        targets_est = targets_est.contiguous().view(-1, 72)

        targets_est_np = targets_est.data*1000. #after it comes out of the forward kinematics

        betas_est_np = betas_est.data*1000.

        scores = scores.unsqueeze(0)
        scores = scores.unsqueeze(0)
        scores = F.pad(scores, (0, 100 + add_idx, 0, 0))
        scores = scores.squeeze(0)
        scores = scores.squeeze(0)

        #print targets_est.size()
        #print scores.size()

        #tweak this to change the lengths vector
        scores[:, 34+add_idx:106+add_idx] = torch.mul(targets_est[:, 0:72], 1.)

        #print scores[13, 34:106]



        if is_training == True:

            #print targets_est_np.size()
            for joint_num in range(24):
                if joint_num in [0, 1, 2, 6, 9, 10, 11, 12, 13, 14, 16, 17, 22, 23]: #if in this then mult by zero
                    targets_est_np[:, joint_num * 3] = targets_est_np[:, joint_num * 3] * synth_real_switch.data
                    targets_est_np[:, joint_num * 3 + 1] = targets_est_np[:, joint_num * 3 + 1] * synth_real_switch.data
                    targets_est_np[:, joint_num * 3 + 2] = targets_est_np[:, joint_num * 3 + 2] * synth_real_switch.data




            scores[:, 0:10] = torch.mul(synth_real_switch.unsqueeze(1), torch.sub(scores[:, 0:10], betas))#*.2

            #compare the output angles to the target values
            if reg_angles == True:
                if self.loss_vector_type == 'anglesDC':
                    scores[:, 34:106] = angles_gt.clone().view(-1, 72) - scores[:, 13:85]

                    scores[:, 34:106] = torch.mul(synth_real_switch.unsqueeze(1), torch.sub(scores[:, 34:106], angles_gt.clone().view(-1, 72)))


                elif self.loss_vector_type == 'anglesEU':
                    scores[:, 34:106] = KinematicsLib().batch_euler_angles_from_dir_cos_angles(angles_gt.view(-1, 24, 3).clone()).contiguous().view(-1, 72) - scores[:, 13:85]

                scores[:, 34:106] = torch.mul(synth_real_switch.unsqueeze(1), scores[:, 34:106].clone())

            if self.GPU == True:
                penetration_weights = torch.Tensor(KinematicsLib().get_penetration_weights(images, targets[:, 0:72])).cuda()
            else:
                penetration_weights = torch.Tensor(KinematicsLib().get_penetration_weights(images, targets[:, 0:72]))

            #print np.shape(penetration_weights)
            #print penetration_weights[0, :]

            #compare the output joints to the target values
            scores[:, 34+add_idx:106+add_idx] = targets[:, 0:72]/1000 - scores[:, 34+add_idx:106+add_idx]
            scores[:, 106+add_idx:178+add_idx] = ((scores[:, 34+add_idx:106+add_idx].clone())+0.0000001).pow(2)


            #print scores[13, 106:178]

            for joint_num in range(24):
                #print scores[:, 10+joint_num].size(), 'score size'
                #print synth_real_switch.size(), 'switch size'
                if joint_num in [0, 1, 2, 6, 9, 10, 11, 12, 13, 14, 16, 17, 22, 23]: #torso is 3 but forget training it
                    scores[:, 10+joint_num] = torch.mul(synth_real_switch,
                                                        (scores[:, 106+add_idx+joint_num*3] +
                                                         scores[:, 107+add_idx+joint_num*3] +
                                                         scores[:, 108+add_idx+joint_num*3]).sqrt())

                else:
                    scores[:, 10+joint_num] = (scores[:, 106+add_idx+joint_num*3] +
                                               scores[:, 107+add_idx+joint_num*3] +
                                               scores[:, 108+add_idx+joint_num*3]).sqrt()

                    #print scores[:, 10+joint_num], 'score size'
                    #print synth_real_switch, 'switch size'



            scores = scores.unsqueeze(0)
            scores = scores.unsqueeze(0)
            scores = F.pad(scores, (0, -151, 0, 0))
            scores = scores.squeeze(0)
            scores = scores.squeeze(0)

            #print scores[0, :]
            #here multiply by 24/10 when you are regressing to real data so it balances with the synthetic data
            scores = torch.mul(torch.add(1.0, torch.mul(1.4, torch.sub(1, synth_real_switch))).unsqueeze(1), scores)
            #scores = torch.mul(torch.add(1.0, torch.mul(3.0, torch.sub(1, synth_real_switch))).unsqueeze(1), scores)
            #scores = torch.mul(torch.mul(2.4, torch.sub(1, synth_real_switch)).unsqueeze(1), scores)

            # here multiply by 5 when you are regressing to real data because there is only 1/5 the amount of it
            #scores = torch.mul(torch.mul(5.0, torch.sub(1, synth_real_switch)).unsqueeze(1), scores)

            #print scores[0, :]
            #print scores[7, :]


            targets_est_reduced_np = 0

            scores[:, 0:10] = torch.mul(scores[:, 0:10].clone(), (1/1.7312621950698526)) #weight the betas by std
            scores[:, 10:34] = torch.mul(scores[:, 10:34].clone(), (1/0.1282715100608753)) #weight the 24 joints by std

            #scores[:, 10:34] = torch.mul(scores[:, 10:34].clone(), penetration_weights)

            if reg_angles == True: scores[:, 34:106] = torch.mul(scores[:, 34:106].clone(), (1/0.2130542427733348)) #weight the angles by how many there are

            #scores[:, 0:10] = torch.mul(scores[:, 0:10].clone(), (1./10)) #weight the betas by how many betas there are
            #scores[:, 10:34] = torch.mul(scores[:, 10:34].clone(), (1./24)) #weight the joints by how many there are
            #if reg_angles == True: scores[:, 34:106] = torch.mul(scores[:, 34:106].clone(), (1./72)) #weight the angles by how many there are

        else:
            if self.GPU == True:
                targets_est_reduced = torch.empty(targets_est.size()[0], 30, dtype=torch.float).cuda()
            else:    
                targets_est_reduced = torch.empty(targets_est.size()[0], 30, dtype=torch.float)
            #scores[:, 80] = torch.add(scores[:, 80], 0.2)
            #scores[:, 81] = torch.add(scores[:, 81], 0.2)
            targets_est_reduced[:, 0:3] = scores[:, 79+add_idx:82+add_idx] #head 34 + 3*15 = 79
            targets_est_reduced[:, 3:6] = scores[:, 43+add_idx:46+add_idx] #torso 34 + 3*3 = 45
            targets_est_reduced[:, 6:9] = scores[:, 91+add_idx:94+add_idx] #right elbow 34 + 19*3 = 91
            targets_est_reduced[:, 9:12] = scores[:, 88+add_idx:91+add_idx] #left elbow 34 + 18*3 = 88
            targets_est_reduced[:, 12:15] = scores[:, 97+add_idx:100+add_idx] #right wrist 34 + 21*3 = 97
            targets_est_reduced[:, 15:18] = scores[:, 94+add_idx:97+add_idx] #left wrist 34 + 20*3 = 94
            targets_est_reduced[:, 18:21] = scores[:, 49+add_idx:52+add_idx] #right knee 34 + 3*5
            targets_est_reduced[:, 21:24] = scores[:, 46+add_idx:49+add_idx] #left knee 34 + 3*4
            targets_est_reduced[:, 24:27] = scores[:, 58+add_idx:61+add_idx] #right ankle 34 + 3*8
            targets_est_reduced[:, 27:30] = scores[:, 55+add_idx:58+add_idx] #left ankle 34 + 3*7

            targets_est_reduced_np = targets_est_reduced.data*1000.

            #print(targets.size(), targets[0, :])

            if targets is not None:
                scores[:, 10:40] = (targets/1000. - targets_est_reduced[:, 0:30]).pow(2)

            #print(scores.size(), scores[0, 10:40])


            for joint_num in range(10):
                scores[:, joint_num] = (scores[:, 10+joint_num*3] + scores[:, 11+joint_num*3] + scores[:, 12+joint_num*3]).sqrt()


            scores = scores.unsqueeze(0)
            scores = scores.unsqueeze(0)
            scores = F.pad(scores, (0, -175-add_idx, 0, 0))
            scores = scores.squeeze(0)
            scores = scores.squeeze(0)


            #here multiply by 24/10 when you are regressing to real data so it balances with the synthetic data
            scores = torch.mul(2.4, scores)

            scores[:, 0:10] = torch.mul(scores[:, 0:10].clone(), (1/0.1282715100608753)) #weight the 10 joints by std
            scores[:, 0:10] = torch.mul(scores[:, 0:10].clone(), (1./24)) #weight the joints by how many there are USE 24 EVEN ON REAL DATA

        mesh_matrix_batch = mesh_matrix_batch.type(self.dtype)
        contact_matrix_batch = contact_matrix_batch.type(self.dtype)

        #print scores[0, :]
        return  scores, mesh_matrix_batch, contact_matrix_batch, targets_est_np, targets_est_reduced_np, betas_est_np

