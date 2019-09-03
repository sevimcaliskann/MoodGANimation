import torch
from collections import OrderedDict
from torch.autograd import Variable
import utils.util as util
import utils.plots as plot_utils
from .models import BaseModel
from networks.networks import NetworksFactory
import os
import numpy as np


class GANimation(BaseModel):
    def __init__(self, opt):
        super(GANimation, self).__init__(opt)
        self._name = 'GANimation'

        # create networks
        self._init_create_networks()

        # init train variables
        if self._is_train:
            self._init_train_vars()

        # load networks and optimizers
        if not self._is_train or self._opt.load_epoch > 0:
            self.load()

        # prefetch variables
        self._init_prefetch_inputs()

        # init
        self._init_losses()

    def _init_create_networks(self):
        # generator network
        self._G = self._create_generator()
        self._G.init_weights()
        if len(self._gpu_ids) > 1:
            self._G = torch.nn.DataParallel(self._G, device_ids=self._gpu_ids[0])
        self._G.cuda()

        # discriminator network
        self._D = self._create_discriminator()
        self._D.init_weights()
        if len(self._gpu_ids) > 1:
            self._D = torch.nn.DataParallel(self._D, device_ids=self._gpu_ids[0])
        self._D.cuda()

    def _create_generator(self):
        return NetworksFactory.get_by_name('generator_wasserstein_gan', c_dim=self._opt.cond_nc+3)

    def _create_discriminator(self):
        return NetworksFactory.get_by_name('discriminator_wasserstein_gan', c_dim=self._opt.cond_nc)

    def _init_train_vars(self):
        self._current_lr_G = self._opt.lr_G
        self._current_lr_D = self._opt.lr_D

        # initialize optimizers
        self._optimizer_G = torch.optim.Adam(self._G.parameters(), lr=self._current_lr_G,
                                             betas=[self._opt.G_adam_b1, self._opt.G_adam_b2])
        self._optimizer_D = torch.optim.Adam(self._D.parameters(), lr=self._current_lr_D,
                                             betas=[self._opt.D_adam_b1, self._opt.D_adam_b2])

    def _init_prefetch_inputs(self):
        self._input_frames = self._Tensor(self._opt.batch_size, self._opt.frames_cnt, 3, self._opt.image_size, self._opt.image_size)
        self._input_annotations = self._Tensor(self._opt.batch_size, self._opt.frames_cnt, self._opt.cond_nc)
        #self._input_desired_cond = self._Tensor(self._opt.batch_size, self._opt.cond_nc)
        #self._input_cond_id = None
        #self._input_target_frame = self._Tensor(self._opt.batch_size, 3, self._opt.image_size, self._opt.image_size)
        self._first_frame = self._Tensor(self._opt.batch_size, 3, self._opt.image_size, self._opt.image_size)
        self._first_ann = self._Tensor(self._opt.batch_size, 1, self._opt.cond_nc)
        #self._input_real_cond_path = None

    def _init_losses(self):
        # define loss functions
        self._criterion_cycle = torch.nn.L1Loss().cuda()
        self._robust_cycle = torch.nn.SmoothL1Loss().cuda()
        self._criterion_D_cond = torch.nn.MSELoss().cuda()

        # init losses G
        self._loss_g_fake = Variable(self._Tensor([0]))
        #self._loss_g_cond = Variable(self._Tensor([0]))
        self._loss_g_cyc = Variable(self._Tensor([0]))
        self._loss_g_mask_1 = Variable(self._Tensor([0]))
        self._loss_g_mask_2 = Variable(self._Tensor([0]))
        #self._loss_g_idt = Variable(self._Tensor([0]))
        self._loss_g_masked_fake = Variable(self._Tensor([0]))
        self._loss_g_masked_cond = Variable(self._Tensor([0]))
        #self._loss_g_cyc_cond = Variable(self._Tensor([0]))
        self._loss_g_mask_1_smooth = Variable(self._Tensor([0]))
        self._loss_g_mask_2_smooth = Variable(self._Tensor([0]))
        #self._loss_inv_mask = Variable(self._Tensor([0]))
        #self._loss_color_maps = Variable(self._Tensor([0]))
        #self._loss_rec_real_img_rgb = Variable(self._Tensor([0]))
        #self._loss_g_fake_imgs_smooth = Variable(self._Tensor([0]))
        #self._loss_g_unmasked_rgb = Variable(self._Tensor([0]))

        # init losses D
        self._loss_d_real = Variable(self._Tensor([0]))
        self._loss_d_cond = Variable(self._Tensor([0]))
        self._loss_d_fake = Variable(self._Tensor([0]))
        self._loss_d_gp = Variable(self._Tensor([0]))

    def set_input(self, input):
        self._input_frames.resize_(input['frames'].size()).copy_(input['frames'])
        self._input_annotations.resize_(input['annotations'].size()).copy_(input['annotations'])
        #self._input_desired_cond.resize_(input['desired_cond'].size()).copy_(input['desired_cond'])
        #self._input_target_frame.resize_(input['target_frame'].size()).copy_(input['target_frame'])
        self._first_frame.resize_(input['first_frame'].size()).copy_(input['first_frame'])
        self._first_ann.resize_(input['first_ann'].size()).copy_(input['first_ann'])
        #self._input_cond_id = input['cond_id']

        if len(self._gpu_ids) > 0:
            self._input_frames = self._input_frames.cuda(self._gpu_ids[0], async=True)
            self._input_annotations = self._input_annotations.cuda(self._gpu_ids[0], async=True)
            #self._input_target_frame = self._input_target_frame.cuda(self._gpu_ids[0], async=True)
            self._first_frame = self._first_frame.cuda(self._gpu_ids[0], async=True)
            self._first_ann = self._first_ann.cuda(self._gpu_ids[0], async=True)
            #self._input_desired_cond = self._input_desired_cond.cuda(self._gpu_ids[0], async=True)

    def set_train(self):
        self._G.train()
        self._D.train()
        self._is_train = True

    def set_eval(self):
        self._G.eval()
        self._is_train = False

    # get image paths
    def get_image_paths(self):
        return OrderedDict([('cond_id', self._input_cond_id)])

    def forward(self, keep_data_for_visuals=False, return_estimates=False):
        if not self._is_train:
            first_frame = Variable(self._first_frame, volatile=True)
            fake_imgs_masked = None
            fake_videos = list()
            fake_mask_videos = list()
            fake_videos_masked = list()
            for idx in range(1, self._opt.frames_cnt):
                real_cond = Variable(self._input_annotations[:, idx, :], volatile=True)

                if fake_imgs_masked is None:
                    fake_imgs, fake_img_mask = self._G.forward(first_frame, real_cond)
                    fake_img_mask = self._do_if_necessary_saturate_mask(fake_img_mask, saturate=self._opt.do_saturate_mask)
                    fake_imgs_masked = fake_img_mask * first_frame + (1 - fake_img_mask) * fake_imgs
                else:
                    fake_imgs, fake_img_mask = self._G.forward(fake_imgs_masked, real_cond)
                    fake_img_mask = self._do_if_necessary_saturate_mask(fake_img_mask, saturate=self._opt.do_saturate_mask)
                    fake_imgs_masked = fake_img_mask * fake_imgs_masked + (1 - fake_img_mask) * fake_imgs
                fake_videos_masked.append(fake_imgs_masked)
                fake_mask_videos.append(fake_img_mask)
                fake_videos.append(fake_imgs)

            fake_videos = torch.transpose(torch.stack(fake_videos), 0, 1)
            fake_mask_videos = torch.transpose(torch.stack(fake_mask_videos), 0, 1)
            fake_videos_masked = torch.transpose(torch.stack(fake_videos_masked), 0, 1)

            imgs = None
            data = None
            if return_estimates:
                # generate images
                vid_real_img = util.tensor2vid(self._input_frames[:, 1:, :, :, :])
                vid_fake_imgs = util.tensor2vid(fake_videos.data)
                #im_fake_img_mask_norm = util.tensor2maskim(fake_img_mask_norm.data)
                vid_mask = util.tensor2maskvid(fake_mask_videos.data)
                vid_fake_masked = util.tensor2vid(fake_videos_masked.data)
                #im_target_img = util.tensor2im(target_frame.data)
                #im_rec_imgs = util.tensor2im(rec_real_img_rgb.data)
                #im_rec_img_mask_norm = util.tensor2maskim(rec_real_img_mask.data)
                #im_rec_imgs_masked = util.tensor2im(rec_real_imgs.data)
                #im_concat_img = np.concatenate([im_real_img, im_fake_imgs_masked, im_fake_img_mask_norm, im_fake_imgs,
                #                                im_rec_imgs, im_rec_img_mask_norm, im_rec_imgs_masked],
                #                               1)
                vid_concat_img = np.concatenate([vid_real_img, vid_fake_masked, vid_fake_imgs, vid_mask], 2)

                vid_real_img_batch = util.tensor2vid(self._input_frames[:, 1:, :, :, :], idx=-1)
                vid_fake_imgs_batch = util.tensor2vid(fake_videos.data, idx=-1)
                #im_fake_img_mask_norm_batch = util.tensor2maskim(fake_img_mask_norm.data, idx=-1, nrows=1)
                vid_fake_masked_batch = util.tensor2vid(fake_videos_masked.data, idx=-1)
                vid_mask_batch = util.tensor2vid(fake_mask_videos.data, idx=-1)
                vid_concat_img_batch = np.concatenate([vid_real_img_batch, vid_fake_masked_batch, vid_fake_imgs_batch, vid_mask_batch], 2)

                imgs = OrderedDict([('real_img', vid_real_img),
                                    ('fake_imgs', vid_fake_imgs),
                                    #('fake_img_mask', im_fake_img_mask_norm),
                                    ('fake_imgs_masked', vid_fake_masked),
                                    ('img_mask', vid_mask),
                                    ('concat', vid_concat_img),
                                    ('real_img_batch', vid_real_img_batch),
                                    ('fake_imgs_batch', vid_fake_imgs_batch),
                                    ('fake_img_mask_batch', vid_mask_batch),
                                    ('fake_imgs_masked_batch', vid_fake_masked_batch),
                                    ('concat_batch', vid_concat_img_batch),
                                    ])

                #data = OrderedDict([('real_path', self._input_cond_id),
                #                    ('desired_cond', desired_cond.data[0, ...].cpu().numpy().astype('str'))
                #                    ])

            # keep data for visualization
            if keep_data_for_visuals:
                self._vis_first_frame = util.tensor2im(self._first_frame)
                self._vis_real_img = util.tensor2vid(self._input_frames[:, 1:, :, :, :])
                self._vis_fake_img_unmasked = util.tensor2vid(fake_videos.data)
                self._vis_fake_img = util.tensor2vid(fake_videos_masked.data)
                self._vis_fake_img_mask = util.tensor2maskvid(vid_mask.data)
                self._vis_annotations = self._input_annotations.cpu()[0, ...].numpy()
                self._vis_batch_real_img = util.tensor2vid(self._input_frames, idx=-1)
                self._vis_batch_fake_img_mask = util.tensor2maskvid(vid_mask.data, idx=-1)
                self._vis_batch_fake_img = util.tensor2vid(fake_videos_masked.data, idx=-1)

            return imgs

    def optimize_parameters(self, train_generator=True, keep_data_for_visuals=False):
        if self._is_train:
            # convert tensor to variables
            self._B = self._input_frames.size(0)
            self._frames = Variable(self._input_frames)
            self._annotations = Variable(self._input_annotations)
            self._first = Variable(self._first_frame)
            self._first_ann = Variable(self._first_ann)

            # train D
            loss_D, fake_vids_masked = self._forward_D()
            self._optimizer_D.zero_grad()
            loss_D.backward()
            self._optimizer_D.step()

            loss_D_gp = 0
            for i in range(fake_vids_masked.size(1)):
                loss_D_gp += self._gradinet_penalty_D(fake_vids_masked[:, i, :, :, :])
            self._optimizer_D.zero_grad()
            loss_D_gp.backward()
            self._optimizer_D.step()

            # train G
            if train_generator:
                loss_G = self._forward_G(keep_data_for_visuals)
                self._optimizer_G.zero_grad()
                loss_G.backward()
                self._optimizer_G.step()

    def _forward_G(self, keep_data_for_visuals):
        # generate fake images
        self._loss_g_masked_fake = 0
        self._loss_g_masked_cond = 0
        self._loss_g_cyc = 0
        self._loss_g_mask_1 = 0
        self._loss_g_mask_1_smooth = 0


        fake_imgs_masked = None
        fake_videos = list()
        fake_mask_videos = list()
        fake_videos_masked = list()
        for idx in range(1, self._opt.frames_cnt):
            real_img = self._frames[:, idx, :, :, :]
            real_cond = self._annotations[:, idx, :]

            if fake_imgs_masked is None:
                fake_imgs, fake_img_mask = self._G.forward(self._first, real_cond)
                fake_img_mask = self._do_if_necessary_saturate_mask(fake_img_mask, saturate=self._opt.do_saturate_mask)
                fake_imgs_masked = fake_img_mask * self._first + (1 - fake_img_mask) * fake_imgs
            else:
                fake_imgs, fake_img_mask = self._G.forward(fake_imgs_masked, real_cond)
                fake_img_mask = self._do_if_necessary_saturate_mask(fake_img_mask, saturate=self._opt.do_saturate_mask)
                fake_imgs_masked = fake_img_mask * fake_imgs_masked + (1 - fake_img_mask) * fake_imgs

            d_fake_desired_img_masked_prob, d_fake_desired_img_masked_cond = self._D.forward(fake_imgs_masked)
            self._loss_g_masked_fake += self._compute_loss_D(d_fake_desired_img_masked_prob, True) * self._opt.lambda_D_prob
            self._loss_g_masked_cond += self._criterion_D_cond(d_fake_desired_img_masked_cond, real_cond) / self._B * self._opt.lambda_D_cond
            self._loss_g_cyc += self._criterion_cycle(fake_imgs_masked, real_img) * self._opt.lambda_cyc
            self._loss_g_mask_1 += torch.mean(fake_img_mask) * self._opt.lambda_mask
            self._loss_g_mask_1_smooth += self._compute_loss_smooth(fake_img_mask) * self._opt.lambda_mask_smooth

            fake_videos_masked.append(fake_imgs_masked)
            fake_mask_videos.append(fake_img_mask)
            fake_videos.append(fake_imgs)

        fake_videos = torch.transpose(torch.stack(fake_videos), 0, 1)
        fake_mask_videos = torch.transpose(torch.stack(fake_mask_videos), 0, 1)
        fake_videos_masked = torch.transpose(torch.stack(fake_videos_masked), 0, 1)

        # keep data for visualization
        if keep_data_for_visuals:
            self._vis_real_img = util.tensor2vid(self._frames.data)
            self._vis_fake_img_unmasked = util.tensor2vid(fake_videos.data)
            self._vis_fake_img = util.tensor2vid(fake_videos_masked.data)
            self._vis_fake_img_mask = util.tensor2maskvid(fake_mask_videos.data)
            self._vis_first_frame = util.tensor2im(self._first.data)
            self._vis_annotations = self._annotations.cpu()[0, ...].numpy()
            self._vis_first_ann = self._first_ann.cpu()[0, ...].numpy()
            #self._vis_batch_real_img = util.tensor2im(self._input_frames, idx=-1)
            self._vis_batch_real_img = util.tensor2vid(self._frames, idx=-1)
            self._vis_batch_fake_img_mask = util.tensor2maskvid(fake_mask_videos.data, idx=-1)
            self._vis_batch_fake_img = util.tensor2vid(fake_videos_masked.data, idx=-1)

        # combine losses
        return self._loss_g_masked_fake + self._loss_g_masked_cond + \
               self._loss_g_cyc + \
               self._loss_g_mask_1 + self._loss_g_mask_1_smooth


    def _forward_D(self):
        self._loss_d_real = 0
        self._loss_d_cond = 0
        self._loss_d_fake = 0
        fake_imgs_masked = None
        fake_videos_masked = list()
        for idx in range(1, self._opt.frames_cnt):
            real_img = self._frames[:, idx, :, :, :]
            real_cond = self._annotations[:, idx, :]

            if fake_imgs_masked is None:
                fake_imgs, fake_img_mask = self._G.forward(self._first, real_cond)
                fake_img_mask = self._do_if_necessary_saturate_mask(fake_img_mask, saturate=self._opt.do_saturate_mask)
                fake_imgs_masked = fake_img_mask * self._first + (1 - fake_img_mask) * fake_imgs
            else:
                fake_imgs, fake_img_mask = self._G.forward(fake_imgs_masked, real_cond)
                fake_img_mask = self._do_if_necessary_saturate_mask(fake_img_mask, saturate=self._opt.do_saturate_mask)
                fake_imgs_masked = fake_img_mask * fake_imgs_masked + (1 - fake_img_mask) * fake_imgs


            d_real_img_prob, d_real_img_cond = self._D.forward(real_img)
            self._loss_d_real += self._compute_loss_D(d_real_img_prob, True) * self._opt.lambda_D_prob
            self._loss_d_cond += self._criterion_D_cond(d_real_img_cond, self._desired_cond) / self._B * self._opt.lambda_D_cond
            # D(fake_I)
            d_fake_desired_img_prob, _ = self._D.forward(fake_imgs_masked.detach())
            self._loss_d_fake += self._compute_loss_D(d_fake_desired_img_prob, False) * self._opt.lambda_D_prob
            fake_videos_masked.append(fake_imgs_masked)
        fake_videos_masked = torch.transpose(torch.stack(fake_videos_masked), 0, 1)


        return self._loss_d_real + self._loss_d_cond + self._loss_d_fake, fake_videos_masked
        #return self._loss_d_real + self._loss_d_fake, fake_imgs_masked

    def _gradinet_penalty_D(self, fake_imgs_masked):
        #adaptive = np.mean(np.linalg.norm(self._real_cond.cpu().detach().numpy() - self._desired_cond.cpu().detach().numpy(), axis=1))+1
        # interpolate sample
        alpha = torch.rand(self._B, 1, 1, 1).cuda().expand_as(self._first_frame)
        interpolated = Variable(alpha * self._target_frame.detach() + (1 - alpha) * fake_imgs_masked.detach(), requires_grad=True)
        interpolated_prob, _ = self._D(interpolated)

        # compute gradients
        grad = torch.autograd.grad(outputs=interpolated_prob,
                                   inputs=interpolated,
                                   grad_outputs=torch.ones(interpolated_prob.size()).cuda(),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        # penalize gradients
        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        self._loss_d_gp = torch.mean((grad_l2norm - 1) ** 2) * self._opt.lambda_D_gp

        return self._loss_d_gp

    def _compute_loss_D(self, estim, is_real):
        return -torch.mean(estim) if is_real else torch.mean(estim)

    def _compute_loss_smooth(self, mat):
        return torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])**2) + \
               torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :])**2)
        #return torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
        #       torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))

    def get_current_errors(self):
        loss_dict = OrderedDict([#('g_fake', self._loss_g_fake.detach()),
                                 #('g_cond', self._loss_g_cond.data[0]),
                                 ('g_mskd_fake', self._loss_g_masked_fake.detach()),
                                 ('g_mskd_cond', self._loss_g_masked_cond.detach()),
                                 ('g_cyc', self._loss_g_cyc.detach()),
                                 #('g_rgb', self._loss_rec_real_img_rgb.detach()),
                                 #('g_rgb_un', self._loss_g_unmasked_rgb.detach()),
                                 #('g_rgb_s', self._loss_g_fake_imgs_smooth.detach()),
                                 ('g_m1', self._loss_g_mask_1.detach()),
                                 #('g_m2', self._loss_g_mask_2.detach()),
                                 ('g_m1_s', self._loss_g_mask_1_smooth.detach()),
                                 #('g_m2_s', self._loss_g_mask_2_smooth.detach()),
                                 #('g_mask_inv', self._loss_inv_mask.detach()),
                                 #('g_color_maps', self._loss_color_maps.detach()),

                                 #('g_idt', self._loss_g_idt.detach()),
                                 ('d_real', self._loss_d_real.detach()),
                                 ('d_cond', self._loss_d_cond.detach()),
                                 ('d_fake', self._loss_d_fake.detach()),
                                 ('d_gp', self._loss_d_gp.detach())])

        return loss_dict

    def get_current_scalars(self):
        return OrderedDict([('lr_G', self._current_lr_G), ('lr_D', self._current_lr_D)])

    def get_current_visuals(self):
        # visuals return dictionary
        visuals = OrderedDict()
        # input visuals
        #title_input_img = os.path.basename(self._input_real_img_path[0])

        visuals['1_input_img'] = np.flip(self._vis_real_img, axis =3)
        visuals['2_fake_img'] = np.flip(self._vis_fake_img, axis=3)
        visuals['4_fake_img_unmasked'] = np.flip(self._vis_fake_img_unmasked, axis=3)
        visuals['5_fake_img_mask'] = np.flip(self._vis_fake_img_mask, axis=3)
        visuals['6_first_frame'] = np.flip(self._vis_first_frame, axis=2)
        visuals['8_annotations'] = self._vis_annotations
        visuals['9_first_ann'] = self._vis_first_ann
        # visuals['8_fake_img_mask_sat'] = self._vis_fake_img_mask_saturated
        # visuals['9_rec_real_img_mask_sat'] = self._vis_rec_real_img_mask_saturated
        #visuals['10_batch_real_img'] = np.flip(self._vis_batch_real_img, axis=2)
        visuals['11_batch_fake_img'] = np.flip(self._vis_batch_fake_img, axis=3)
        visuals['12_batch_fake_img_mask'] = np.flip(self._vis_batch_fake_img_mask, axis=3)
        visuals['13_batch_real_img'] = np.flip(self._vis_batch_real_img, axis=3)
        # visuals['11_idt_img'] = self._vis_idt_img

        return visuals

    def save(self, label):
        # save networks
        self._save_network(self._G, 'G', label)
        self._save_network(self._D, 'D', label)

        # save optimizers
        self._save_optimizer(self._optimizer_G, 'G', label)
        self._save_optimizer(self._optimizer_D, 'D', label)

    def load(self):
        load_epoch = self._opt.load_epoch

        # load G
        self._load_network(self._G, 'G', load_epoch)

        if self._is_train:
            # load D
            self._load_network(self._D, 'D', load_epoch)

            # load optimizers
            self._load_optimizer(self._optimizer_G, 'G', load_epoch)
            self._load_optimizer(self._optimizer_D, 'D', load_epoch)

    def update_learning_rate(self):
        # updated learning rate G
        lr_decay_G = self._opt.lr_G / self._opt.nepochs_decay
        self._current_lr_G -= lr_decay_G
        for param_group in self._optimizer_G.param_groups:
            param_group['lr'] = self._current_lr_G
        print('update G learning rate: %f -> %f' %  (self._current_lr_G + lr_decay_G, self._current_lr_G))

        # update learning rate D
        lr_decay_D = self._opt.lr_D / self._opt.nepochs_decay
        self._current_lr_D -= lr_decay_D
        for param_group in self._optimizer_D.param_groups:
            param_group['lr'] = self._current_lr_D
        print('update D learning rate: %f -> %f' %  (self._current_lr_D + lr_decay_D, self._current_lr_D))

    def _l1_loss_with_target_gradients(self, input, target):
        return torch.sum(torch.abs(input - target)) / input.detach().nelement()

    def _do_if_necessary_saturate_mask(self, m, saturate=False):
        return torch.clamp(0.55*torch.tanh(3*(m-0.5))+0.5, 0, 1) if saturate else m
