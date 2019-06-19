import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from kinematics_lib import KinematicsLib
import scipy.stats as ss
import torchvision
import resnet

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

            nn.Conv2d(5, 256, kernel_size=7, stride=2, padding=3),
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

            nn.Conv2d(5, 32, kernel_size = 7, stride = 2, padding = 3),
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
            print('######################### CUDA is available! #############################')
        else:
            self.GPU = False
            # Use for CPU
            dtype = torch.FloatTensor
            print('############################## USING CPU #################################')
        self.dtype = dtype


        if loss_vector_type == 'anglesR' or loss_vector_type == 'anglesDC' or loss_vector_type == 'anglesEU':


            #print torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated(), torch.cuda.memory_cached(), "p1"

            from smpl.smpl_webuser.serialization import load_model

            model_path_f = filepath+'git/SMPL_python_v.1.0.0/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
            human_f = load_model(model_path_f)
            self.v_template_f = torch.Tensor(np.array(human_f.v_template)).type(dtype)
            self.shapedirs_f = torch.Tensor(np.array(human_f.shapedirs)).permute(0, 2, 1).type(dtype)
            self.J_regressor_f = np.zeros((human_f.J_regressor.shape)) + human_f.J_regressor
            self.J_regressor_f = torch.Tensor(np.array(self.J_regressor_f).astype(float)).permute(1, 0).type(dtype)
            self.posedirs_f = torch.Tensor(np.array(human_f.posedirs)).type(dtype)
            self.weights_f = torch.Tensor(np.array(human_f.weights)).type(dtype)

            model_path_m = filepath+'/git/SMPL_python_v.1.0.0/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
            human_m = load_model(model_path_m)
            self.v_template_m = torch.Tensor(np.array(human_m.v_template)).type(dtype)
            self.shapedirs_m = torch.Tensor(np.array(human_m.shapedirs)).permute(0, 2, 1).type(dtype)
            self.J_regressor_m = np.zeros((human_m.J_regressor.shape)) + human_m.J_regressor
            self.J_regressor_m = torch.Tensor(np.array(self.J_regressor_m).astype(float)).permute(1, 0).type(dtype)
            self.posedirs_m = torch.Tensor(np.array(human_m.posedirs)).type(dtype)
            self.weights_m = torch.Tensor(np.array(human_m.weights)).type(dtype)

            #print torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated(), torch.cuda.memory_cached(),"p2"

            self.parents = np.array([4294967295, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]).astype(np.int32)

            #print batch_size
            batch_sub_divider = 8

            self.N = batch_size/batch_sub_divider
            self.shapedirs_f = self.shapedirs_f.unsqueeze(0).repeat(self.N, 1, 1, 1).permute(0, 2, 1, 3).unsqueeze(0)
            self.shapedirs_m = self.shapedirs_m.unsqueeze(0).repeat(self.N, 1, 1, 1).permute(0, 2, 1, 3).unsqueeze(0)
            self.shapedirs = torch.cat((self.shapedirs_f, self.shapedirs_m), 0) #this is 2 x N x B x R x D
            self.B = self.shapedirs.size()[2] #this is 10
            self.R = self.shapedirs.size()[3] #this is 6890, or num of verts
            self.D = self.shapedirs.size()[4] #this is 3, or num dimensions
            self.R_used = 6890
            self.shapedirs = self.shapedirs.permute(1,0,2,3,4).view(self.N, 2, self.B*self.R*self.D)

            #print torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated(), torch.cuda.memory_cached(), "p3"


            self.v_template_f = self.v_template_f.unsqueeze(0).repeat(self.N, 1, 1).unsqueeze(0)
            self.v_template_m = self.v_template_m.unsqueeze(0).repeat(self.N, 1, 1).unsqueeze(0)
            self.v_template = torch.cat((self.v_template_f, self.v_template_m), 0)#this is 2 x N x R x D
            self.v_template = self.v_template.permute(1,0,2,3).view(self.N, 2, self.R*self.D)

            #print torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated(), torch.cuda.memory_cached(),"p4"

            self.J_regressor = torch.cat((self.J_regressor_f.unsqueeze(0).repeat(self.N, 1, 1).unsqueeze(0),
                                          self.J_regressor_m.unsqueeze(0).repeat(self.N, 1, 1).unsqueeze(0)), 0)#this is 2 x N x R x 24
            self.J_regressor = self.J_regressor.permute(1,0,2,3).view(self.N, 2, self.R*24)

            #print torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated(), torch.cuda.memory_cached(),"p5"

            self.posedirs = torch.cat((self.posedirs_f.unsqueeze(0).repeat(self.N, 1, 1, 1).unsqueeze(0),
                                       self.posedirs_m.unsqueeze(0).repeat(self.N, 1, 1, 1).unsqueeze(0)), 0)
            # self.posedirs = self.posedirs.permute(1, 0, 2, 3, 4).view(self.N, 2, self.R*self.D*207)
            self.posedirs = self.posedirs.permute(1, 0, 2, 3, 4).view(self.N, 2, self.R_used * self.D * 207)

            self.weights_repeat_f = self.weights_f.unsqueeze(0).repeat(self.N, 1, 1).unsqueeze(0)
            self.weights_repeat_m = self.weights_m.unsqueeze(0).repeat(self.N, 1, 1).unsqueeze(0)
            self.weights_repeat = torch.cat((self.weights_repeat_f, self.weights_repeat_m), 0)
            # self.weights_repeat = self.weights_repeat.permute(1, 0, 2, 3).view(self.N, 2, self.R * 24)
            self.weights_repeat = self.weights_repeat.permute(1, 0, 2, 3).view(self.N, 2, self.R_used * 24)

            self.zeros_cartesian = torch.zeros([batch_size, 24]).type(dtype)
            self.ones_cartesian = torch.ones([batch_size, 24]).type(dtype)

            self.filler_taxels = []
            for i in range(27):
                for j in range(64):
                    self.filler_taxels.append([i, j, 20000])
            self.filler_taxels = torch.Tensor(self.filler_taxels).type(torch.IntTensor).unsqueeze(0).repeat(batch_size, 1, 1)

            #print torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated(), torch.cuda.memory_cached(),"p6"

            if self.loss_vector_type == 'anglesDC':
                self.bounds = torch.Tensor(np.array([[-np.pi/3, np.pi/3], [-np.pi/3, np.pi/3], [-np.pi/3, np.pi/3],
                                       [-2.753284558994594, -0.14634814003149707], [-1.0403111466710133, 1.1185343875601006], [-0.421484532214729, 0.810063927501682],
                                       [-2.753284558994594, -0.14634814003149707], [-1.1185343875601006, 1.0403111466710133],  [-0.810063927501682, 0.421484532214729],
                                       [-np.pi / 6, np.pi / 6], [-np.pi / 36, np.pi / 36], [-np.pi / 36, np.pi / 36],
                                       [0.0, 2.7020409229712863], [-0.01, 0.01], [-0.01, 0.01],  # knee
                                       [0.0, 2.7020409229712863], [-0.01, 0.01], [-0.01, 0.01],
                                       [-np.pi / 6, np.pi / 6], [-np.pi / 36, np.pi / 36], [-np.pi / 36, np.pi / 36],
                                       [-np.pi / 6, np.pi / 6], [-np.pi / 6, np.pi / 6], [-np.pi / 6, np.pi / 6],  # ankle, pi/36 or 5 deg
                                       [-np.pi / 6, np.pi / 6], [-np.pi / 6, np.pi / 6], [-np.pi / 6, np.pi / 6],  # ankle, pi/36 or 5 deg
                                       [-np.pi / 6, np.pi / 6], [-np.pi / 36, np.pi / 36], [-np.pi / 36, np.pi / 36],
                                       [-0.01, 0.01], [-0.01, 0.01], [-0.01, 0.01],  # foot
                                       [-0.01, 0.01], [-0.01, 0.01], [-0.01, 0.01],  # foot
                                       [-np.pi / 6, np.pi / 6], [-np.pi / 36, np.pi / 36], [-np.pi / 36, np.pi / 36],  # neck
                                       [-1.8674195346872975* 1 / 3, 1.410545172086535  * 1 / 3], [-1.530112726921327 * 1 / 3, 1.2074724617209949 * 1 / 3], [-1.9550515937478927 * 1 / 3, 1.7587935205169856 * 1 / 3],
                                       [-1.8674195346872975 * 1 / 3, 1.410545172086535  * 1 / 3], [-1.2074724617209949 * 1 / 3, 1.530112726921327 * 1 / 3], [-1.7587935205169856 * 1 / 3, 1.9550515937478927 * 1 / 3],
                                       [-np.pi / 6, np.pi / 6], [-np.pi / 36, np.pi / 36], [-np.pi / 36, np.pi / 36],  # head
                                       [-1.8674195346872975 * 2 / 3, 1.410545172086535  * 2 / 3], [-1.530112726921327 * 2 / 3, 1.2074724617209949 * 2 / 3], [-1.9550515937478927 * 2 / 3, 1.7587935205169856 * 2 / 3],
                                       [-1.8674195346872975 * 2 / 3, 1.410545172086535  * 2 / 3], [-1.2074724617209949 * 2 / 3, 1.530112726921327 * 2 / 3], [-1.7587935205169856 * 2 / 3, 1.9550515937478927 * 2 / 3],
                                       [-0.01, 0.01], [-2.463868908637374, 0.0], [-0.01, 0.01],  # elbow
                                       [-0.01, 0.01], [0.0, 2.463868908637374],  [-0.01, 0.01],  # elbow
                                       [-np.pi / 6, np.pi / 6], [-np.pi / 6, np.pi / 6], [-np.pi / 6, np.pi / 6],  # wrist, pi/36 or 5 deg
                                       [-np.pi / 6, np.pi / 6], [-np.pi / 6, np.pi / 6], [-np.pi / 6, np.pi / 6],  # wrist, pi/36 or 5 deg
                                       [-0.01, 0.01], [-0.01, 0.01], [-0.01, 0.01],  # hand
                                       [-0.01, 0.01], [-0.01, 0.01], [-0.01, 0.01]])).type(dtype)

            elif self.loss_vector_type == 'anglesEU':
                self.bounds = torch.Tensor(np.array([[-np.pi/3, np.pi/3], [-np.pi/3, np.pi/3], [-np.pi/3, np.pi/3],
                                       [-2.753284558994594, -0.2389229307048895], [-1.0047479181618846, 0.8034397361593714], [-0.8034397361593714, 1.0678805158941416],
                                       [-2.753284558994594, -0.2389229307048895], [-0.8034397361593714, 1.0047479181618846], [-1.0678805158941416, 0.8034397361593714],
                                       [-np.pi / 6, np.pi / 6], [-np.pi / 36, np.pi / 36], [-np.pi / 36, np.pi / 36],
                                       [0.0, 2.7020409229712863], [-0.01, 0.01], [-0.01, 0.01],  # knee
                                       [0.0, 2.7020409229712863], [-0.01, 0.01], [-0.01, 0.01],
                                       [-np.pi / 6, np.pi / 6], [-np.pi / 36, np.pi / 36], [-np.pi / 36, np.pi / 36],
                                       [-np.pi / 6, np.pi / 6], [-np.pi / 6, np.pi / 6], [-np.pi / 6, np.pi / 6],  # ankle, pi/36 or 5 deg
                                       [-np.pi / 6, np.pi / 6], [-np.pi / 6, np.pi / 6], [-np.pi / 6, np.pi / 6],  # ankle, pi/36 or 5 deg
                                       [-np.pi / 6, np.pi / 6], [-np.pi / 36, np.pi / 36], [-np.pi / 36, np.pi / 36],
                                       [-0.01, 0.01], [-0.01, 0.01], [-0.01, 0.01],  # foot
                                       [-0.01, 0.01], [-0.01, 0.01], [-0.01, 0.01],  # foot
                                       [-np.pi / 6, np.pi / 6], [-np.pi / 36, np.pi / 36], [-np.pi / 36, np.pi / 36],  # neck
                                       [-1.5704982490935508 * 1 / 3, 1.6731615204412293* 1 / 3], [-1.5359250989762832 * 1 / 3, 0.4892616775215104 * 1 / 3], [-2.032907094968176 * 1 / 3, 1.927742086422412 * 1 / 3],
                                       [-1.5704982490935508 * 1 / 3, 1.6731615204412293 * 1 / 3], [-0.4892616775215104 * 1 / 3, 1.5359250989762832 * 1 / 3], [-1.927742086422412 * 1 / 3, 2.032907094968176 * 1 / 3],
                                       [-np.pi / 6, np.pi / 6], [-np.pi / 36, np.pi / 36], [-np.pi / 36, np.pi / 36],  # head
                                       [-1.5704982490935508 * 2 / 3, 1.6731615204412293 * 2 / 3], [-1.5359250989762832 * 2 / 3, 0.4892616775215104 * 2 / 3], [-2.032907094968176 * 2 / 3, 1.927742086422412 * 2 / 3],
                                       [-1.5704982490935508 * 2 / 3, 1.6731615204412293 * 2 / 3], [-0.4892616775215104 * 2 / 3, 1.5359250989762832 * 2 / 3], [-1.927742086422412 * 2 / 3, 2.032907094968176 * 2 / 3],
                                       [-0.01, 0.01], [-2.463868908637374, 0.0], [-0.01, 0.01],  # elbow
                                       [-0.01, 0.01], [0.0, 2.463868908637374],  [-0.01, 0.01],  # elbow
                                       [-np.pi / 6, np.pi / 6], [-np.pi / 6, np.pi / 6], [-np.pi / 6, np.pi / 6],  # wrist, pi/36 or 5 deg
                                       [-np.pi / 6, np.pi / 6], [-np.pi / 6, np.pi / 6], [-np.pi / 6, np.pi / 6],  # wrist, pi/36 or 5 deg
                                       [-0.01, 0.01], [-0.01, 0.01], [-0.01, 0.01],  # hand
                                       [-0.01, 0.01], [-0.01, 0.01], [-0.01, 0.01]])).type(dtype)



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
                    targets_est_np[:, joint_num * 3 + 0] = targets_est_np[:, joint_num * 3 + 0] * synth_real_switch.data
                    targets_est_np[:, joint_num * 3 + 1] = targets_est_np[:, joint_num * 3 + 1] * synth_real_switch.data
                    targets_est_np[:, joint_num * 3 + 2] = targets_est_np[:, joint_num * 3 + 2] * synth_real_switch.data

            targets_est_reduced_np = 0

            # here we want to compute our score as the Euclidean distance between the estimated x,y,z points and the target.
            scores = targets / 1000. - scores
            scores = scores.pow(2)


            for joint_num in range(24):
                #print scores[:, 10+joint_num].size(), 'score size'
                #print synth_real_switch.size(), 'switch size'
                if joint_num in [0, 1, 2, 6, 9, 10, 11, 12, 13, 14, 16, 17, 22, 23]: #torso is 3 but forget training it
                    scores[:, joint_num] = torch.mul(synth_real_switch,
                                                        (scores[:, joint_num*3 + 0] +
                                                         scores[:, joint_num*3 + 1] +
                                                         scores[:, joint_num*3 + 2]).sqrt())

                else:
                    scores[:, joint_num] = (scores[:, joint_num*3 + 0] +
                                            scores[:, joint_num*3 + 1] +
                                            scores[:, joint_num*3 + 2]).sqrt()


            #print scores.size(), scores[0, :]

            scores = scores[:, 0:24]

            #print scores.size(), scores[0, :]
            scores = torch.mul(torch.add(1.0, torch.mul(1.4, torch.sub(1, synth_real_switch))).unsqueeze(1), scores)
            #print scores.size(), scores[0, :]
            scores[:, 0:24] = torch.mul(scores[:, 0:24].clone(), (1/0.1282715100608753)) #weight the 24 joints by std
            scores[:, 0:24] = torch.mul(scores[:, 0:24].clone(), (1./24)) #weight the joints by how many there are

            #print scores.size(), scores[0, :]

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

            scores = scores[:, 0:10]
            scores = scores.sqrt()


            #here multiply by 24/10 when you are regressing to real data so it balances with the synthetic data
            scores = torch.mul(2.4, scores)
            scores[:, 0:10] = torch.mul(scores[:, 0:10].clone(), (1/0.1282715100608753)) #weight the 10 joints by std
            scores[:, 0:10] = torch.mul(scores[:, 0:10].clone(), (1./24)) #weight the joints by how many there are USE 24 EVEN ON REAL DATA


        #print scores.size(), scores[0, :]

        return scores, targets_est_np, targets_est_reduced_np




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
                torch.abs(self.bounds.view(72, 2)[:, 1] - self.bounds.view(72, 2)[:, 0]).cpu().numpy())
        else:
            output_norm = 10 * [6.0] + [0.91, 1.98, 0.15] + list(
                torch.abs(self.bounds.view(72, 2)[:, 1] - self.bounds.view(72, 2)[:, 0]).numpy())
        for i in range(85):
            scores[:, i] = torch.mul(scores[:, i].clone(), output_norm[i])

        # add a factor so the model starts close to the home position. Has nothing to do with weighting.
        scores[:, 10] = torch.add(scores[:, 10].clone(), 0.6)
        scores[:, 11] = torch.add(scores[:, 11].clone(), 1.2)
        scores[:, 12] = torch.add(scores[:, 12].clone(), 0.1)


        betas_est = scores[:,0:10].clone().detach().cpu().numpy()  # make sure to detach so the gradient flow of joints doesn't corrupt the betas
        root_shift_est = scores[:, 10:13].clone().detach().cpu().numpy()

        # normalize for tan activation function
        scores[:, 13:85] -= torch.mean(self.bounds[0:72, 0:2], dim=1)
        scores[:, 13:85] *= (2. / torch.abs(self.bounds[0:72, 0] - self.bounds[0:72, 1]))
        scores[:, 13:85] = scores[:, 13:85].tanh()
        scores[:, 13:85] /= (2. / torch.abs(self.bounds[0:72, 0] - self.bounds[0:72, 1]))
        scores[:, 13:85] += torch.mean(self.bounds[0:72, 0:2], dim=1)

        if self.loss_vector_type == 'anglesDC':

            angles_est = scores[:, 13:85].clone().detach().cpu().numpy()

        elif self.loss_vector_type == 'anglesEU':

            angles_est = KinematicsLib().batch_dir_cos_angles_from_euler_angles(scores[:, 13:85].view(-1, 24, 3).clone(), self.zeros_cartesian, self.ones_cartesian)

        return np.squeeze(betas_est), np.squeeze(root_shift_est), np.squeeze(angles_est)



    def compute_tensor_mesh(self, gender_switch, betas_est, Rs_est, root_shift_est, start_incr, end_incr):

        sub_batch_size = end_incr - start_incr

        shapedirs = torch.bmm(gender_switch[start_incr:end_incr, :, :],
                                self.shapedirs[0:sub_batch_size, :, :])\
                         .view(sub_batch_size, self.B, self.R*self.D)


        betas_shapedirs_mult = torch.bmm(betas_est[start_incr:end_incr, :].unsqueeze(1), shapedirs)\
                                    .squeeze(1)\
                                    .view(sub_batch_size, self.R, self.D)

        v_template = torch.bmm(gender_switch[start_incr:end_incr, :, :], self.v_template[0:sub_batch_size, :, :])\
                          .view(sub_batch_size, self.R, self.D)

        v_shaped = betas_shapedirs_mult + v_template

        J_regressor = torch.bmm(gender_switch[start_incr:end_incr, :, :], self.J_regressor[0:sub_batch_size, :, :])\
                                  .view(sub_batch_size, self.R, 24)

        Jx = torch.bmm(v_shaped[:, :, 0].unsqueeze(1), J_regressor).squeeze(1)
        Jy = torch.bmm(v_shaped[:, :, 1].unsqueeze(1), J_regressor).squeeze(1)
        Jz = torch.bmm(v_shaped[:, :, 2].unsqueeze(1), J_regressor).squeeze(1)


        J_est = torch.stack([Jx, Jy, Jz], dim=2)  # these are the joint locations with home pose (pose is 0 degree on all angles)
        #J_est = J_est - J_est[:, 0:1, :] + root_shift_est.unsqueeze(1)

        targets_est, A_est = KinematicsLib().batch_global_rigid_transformation(Rs_est[start_incr:end_incr, :], J_est, self.parents, self.GPU, rotate_base=False)

        targets_est = targets_est - J_est[:, 0:1, :] + root_shift_est[start_incr:end_incr, :].unsqueeze(1)

        # assemble a reduced form of the transformed mesh
        #v_shaped_red = torch.stack([v_shaped[:, 1325, :],
        #                            v_shaped[:, 336, :],  # head
        #                            v_shaped[:, 1032, :],  # l knee
        #                            v_shaped[:, 4515, :],  # r knee
        #                            v_shaped[:, 1374, :],  # l ankle
        #                            v_shaped[:, 4848, :],  # r ankle
        #                            v_shaped[:, 1739, :],  # l elbow
        #                            v_shaped[:, 5209, :],  # r elbow
        #                            v_shaped[:, 1960, :],  # l wrist
        #                            v_shaped[:, 5423, :]]).permute(1, 0, 2)  # r wrist

        pose_feature = (Rs_est[start_incr:end_incr, 1:, :, :]).sub(1.0, torch.eye(3).type(self.dtype)).view(-1, 207)
        posedirs = torch.bmm(gender_switch[start_incr:end_incr, :, :], self.posedirs[0:sub_batch_size, :, :]) \
            .view(sub_batch_size, self.R_used * self.D, 207) \
            .permute(0, 2, 1)

        v_posed = torch.bmm(pose_feature.unsqueeze(1), posedirs).view(-1, self.R_used, self.D)

        v_posed = v_posed.clone() + v_shaped

        weights_repeat = torch.bmm(gender_switch[start_incr:end_incr, :, :], self.weights_repeat[0:sub_batch_size, :, :]) \
            .squeeze(1) \
            .view(sub_batch_size, self.R_used, 24)
        T = torch.bmm(weights_repeat, A_est.view(sub_batch_size, 24, 16)).view(sub_batch_size, -1, 4, 4)
        v_posed_homo = torch.cat([v_posed, torch.ones(sub_batch_size, v_posed.shape[1], 1).type(self.dtype)], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))

        verts = v_homo[:, :, :3, 0] - J_est[:, 0:1, :] + root_shift_est[start_incr:end_incr, :].unsqueeze(1)

        return verts, J_est, targets_est



    def forward_kinematic_angles(self, images, gender_switch, synth_real_switch, targets=None, is_training = True, betas=None, angles_gt = None, root_shift = None, reg_angles = False):
        #self.GPU = False
        #self.dtype = torch.FloatTensor

        #print torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated(), torch.cuda.memory_cached(), "p7b"

        scores_cnn = self.CNN_pack1(images)


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
            output_norm = 10*[6.0] + [0.91, 1.98, 0.15] + list(torch.abs(self.bounds.view(72,2)[:, 1] - self.bounds.view(72,2)[:, 0]).cpu().numpy())
        else:
            output_norm = 10 * [6.0] + [0.91, 1.98, 0.15] + list(torch.abs(self.bounds.view(72, 2)[:, 1] - self.bounds.view(72, 2)[:, 0]).numpy())
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
            scores[:, 13:85] -= torch.mean(self.bounds[0:72, 0:2], dim=1)
            scores[:, 13:85] *= (2. / torch.abs(self.bounds[0:72, 0] - self.bounds[0:72, 1]))
            scores[:, 13:85] = scores[:, 13:85].tanh()
            scores[:, 13:85] /= (2. / torch.abs(self.bounds[0:72, 0] - self.bounds[0:72, 1]))
            scores[:, 13:85] += torch.mean(self.bounds[0:72, 0:2], dim=1)

            if self.loss_vector_type == 'anglesDC':

                Rs_est = KinematicsLib().batch_rodrigues(scores[:, 13:85].view(-1, 24, 3).clone()).view(-1, 24, 3, 3)

            elif self.loss_vector_type == 'anglesEU':

                Rs_est = KinematicsLib().batch_euler_to_R(scores[:, 13:85].view(-1, 24, 3).clone(), self.zeros_cartesian, self.ones_cartesian).view(-1, 24, 3, 3)

        else:
            #print betas[13, :], 'betas'
            betas_est = betas
            scores[:, 0:10] = betas.clone()
            scores[:, 13:85] = angles_gt.clone()
            root_shift_est = root_shift

            if self.loss_vector_type == 'anglesDC':

                #normalize for tan activation function
                scores[:, 13:85] -= torch.mean(self.bounds[0:72,0:2], dim = 1)
                scores[:, 13:85] *= (2. / torch.abs(self.bounds[0:72, 0] - self.bounds[0:72, 1]))
                scores[:, 13:85] = scores[:, 13:85].tanh()
                scores[:, 13:85] /= (2. / torch.abs(self.bounds[0:72, 0] - self.bounds[0:72, 1]))
                scores[:, 13:85] += torch.mean(self.bounds[0:72,0:2], dim = 1)


                Rs_est = KinematicsLib().batch_rodrigues(scores[:, 13:85].view(-1, 24, 3).clone()).view(-1, 24, 3, 3)
            elif self.loss_vector_type == 'anglesEU':

                #convert angles DC to EU
                scores[:, 13:85] = KinematicsLib().batch_euler_angles_from_dir_cos_angles(scores[:, 13:85].view(-1, 24, 3).clone()).contiguous().view(-1, 72)

                #normalize for tan activation function
                scores[:, 13:85] -= torch.mean(self.bounds[0:72,0:2], dim = 1)
                scores[:, 13:85] *= (2. / torch.abs(self.bounds[0:72, 0] - self.bounds[0:72, 1]))
                scores[:, 13:85] = scores[:, 13:85].tanh()
                scores[:, 13:85] /= (2. / torch.abs(self.bounds[0:72, 0] - self.bounds[0:72, 1]))
                scores[:, 13:85] += torch.mean(self.bounds[0:72,0:2], dim = 1)

                Rs_est = KinematicsLib().batch_euler_to_R(scores[:, 13:85].view(-1, 24, 3).clone(), self.zeros_cartesian, self.ones_cartesian).view(-1, 24, 3, 3)

        #print Rs_est[0, :]

        gender_switch = gender_switch.unsqueeze(1)
        current_batch_size = gender_switch.size()[0]

        #break things up into sub batches and pass through the mesh
        num_normal_sub_batches = current_batch_size/self.N
        if current_batch_size%self.N != 0:
            sub_batch_incr_list = num_normal_sub_batches*[self.N] + [current_batch_size%self.N]
        else:
            sub_batch_incr_list = num_normal_sub_batches*[self.N]
        start_incr, end_incr = 0, 0
        for sub_batch_incr in sub_batch_incr_list:
            end_incr += sub_batch_incr
            verts_sub, J_est_sub, targets_est_sub = self.compute_tensor_mesh(gender_switch, betas_est, Rs_est, root_shift_est, start_incr, end_incr)
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


        #compute the depth and contact maps from the mesh
        print verts[0, :]
        print verts.size()
        verts_taxel = verts / 0.0286
        verts_taxel[:, :, 2] *= 1000
        verts_taxel[:, :, 0] *= 1.04

        verts_taxel_int = (verts_taxel).type(torch.IntTensor)
        print verts_taxel_int.type()
        print self.filler_taxels.type()

        print verts_taxel_int.size()
        print self.filler_taxels.size()
        verts_taxel_int = torch.cat((self.filler_taxels, verts_taxel_int), dim=1)

        print verts_taxel_int.size()
        print torch.mul(100000, verts_taxel_int[:,:,1]).size()
        vertice_sorting_method = verts_taxel_int[:, :, 0:1] * 10000000 + \
                                 verts_taxel_int[:, :, 1:2] * 100000 + \
                                 verts_taxel_int[:, :, 2:3]


        print verts_taxel_int.size()
        print vertice_sorting_method.size()

        verts_taxel_int = torch.cat((vertice_sorting_method, verts_taxel_int), dim = 2)

        #verts_taxel_int = verts_taxel_int[vertice_sorting_method.argsort()]

        #verts_taxel_int = verts_taxel_int.unsqueeze(3)
        print verts_taxel_int.size()


        #verts_taxel_int.sort(dim=3)

        #print
        x, _ = torch.unique(verts_taxel_int[0, :, :], sorted = True, return_inverse=True, dim = 0)
        for i in range(x.size()[0]):
            print x[i, :]
        print x.size()

        #vertice_sorting_method_2 = verts_taxel_int[:, :, 1:2] * 100 + verts_taxel_int[:, :, 2:3]
        vertice_sorting_method_2 = verts_taxel_int[:, :, 1:3]
        print vertice_sorting_method_2[0, :, :], 'sort meth 2'
        print vertice_sorting_method_2.size()
        x2, keys = torch.unique(vertice_sorting_method_2[0, :, :], sorted = True, return_inverse=True, dim = 0)

        print x2, "x2"
        print x2.size()
        print keys


        #for item in range(unique_ind.size()[0]):
        #    print unique_ind[item]

        #for i in range(verts_taxel_int.size()[1]):
        #    print verts_taxel_int[0, i, :]
        print verts_taxel_int.size()



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

            scores[:, 10:34] = torch.mul(scores[:, 10:34].clone(), penetration_weights)

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

        #print scores[0, :]
        return  scores, targets_est_np, targets_est_reduced_np, betas_est_np

