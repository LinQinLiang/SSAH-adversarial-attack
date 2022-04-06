import numpy as np
import torch
import random
import argparse
from collections import OrderedDict
import torchvision
import torchvision.utils as vutils
import os
import logging

from utils import *
from checkpoints.resnet import ResNet
from attack.ssah_attack import *
from utils.eval_metric_utils import *
from utils.auxiliary_utils import *
from utils.fid_score import return_fid


def parse_arg():
    parser = argparse.ArgumentParser(description='attack with feature layer and frequency constraint')
    parser.add_argument('--bs', type=int, default=10000, help="batch size")
    parser.add_argument('--dataset-root', type=str, default='dataset', help='dataset path')
    parser.add_argument('--dataset', type=str, default='cifar10', help='data to attack')
    parser.add_argument('--classifier', type=str, default='resnet20', help='model to attack')
    parser.add_argument('--seed', type=int, default=18, help='random seed')
    parser.add_argument('--perturb-mode', type=str, default='SSAH', help='attack method')
    parser.add_argument('--max-epoch', type=int, default=1, help='always 1 in attack')
    parser.add_argument('--workers', type=int, default=8, help='num workers to load img')
    parser.add_argument('--wavelet', type=str, default='haar', choices=['haar', 'Daubechies', 'Cohen'])
    parser.add_argument('--test-fid', action='store_true', help='test fid value')

    # SSAH Attack Parameters
    parser.add_argument('--num-iteration', type=int, default=150, help='MAX NUMBER ITERATION')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='LEARNING RATE')
    parser.add_argument('--m', type=float, default=0.2, help='MARGIN')
    parser.add_argument('--alpha', type=float, default=1.0, help='HYPER PARAMETER FOR ADV COST')
    parser.add_argument('--lambda-lf', type=float, default=0.1, help='HYPER PARAMETER FOR LOW FREQUENCY CONSTRAINT')
    parser.add_argument('--outdir', type=str, default='result', help='dir to save the attack examples')
    parser.add_argument('--exp-name', type=str, default='SSAH', help='Experiment Name')

    args = parser.parse_args()

    return args

# parse and log
opt = parse_arg()
opt.outdir = os.path.join(opt.outdir,opt.exp_name)
set_logger(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if opt.classifier == 'resnet20' and opt.dataset == 'cifar10':
    path = 'checkpoints/cifar10-r20.pth.tar'
    checkpoint = torch.load(path)
    state = checkpoint['state_dict']
    classifier = ResNet(20, 10)
    classifier.load_state_dict(state)
elif opt.classifier == 'resnet20' and opt.dataset == 'cifar100':
    path = 'checkpoints/cifar100-r20.pth.tar'
    checkpoint = torch.load(path)
    state = checkpoint['state_dict']
    classifier = ResNet(20, 100)
    new_state_dict = OrderedDict()
    for k, v in state.items():
        if 'module.' in k:
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    classifier.load_state_dict(new_state_dict)
elif opt.classifier == 'resnet50' and opt.dataset == 'imagenet_val':
    classifier = torchvision.models.resnet50(pretrained=True)
classifier.eval()
classifier = classifier.to(device)

if opt.seed != -1:
    print('set seed : ', opt.seed)
    setup_seed(opt.seed)

if opt.dataset == 'cifar10':
    data, num_images = load_cifar10(opt)
elif opt.dataset == 'cifar100':
    data, num_images = load_cifar100(opt)
else:
    data, num_images = load_imagenet_val(opt)
print("Attack Benign Image of {} dataset ({} images) with perturb mode: {} :".format(
    opt.dataset, num_images, opt.perturb_mode
))

total_img = 0
att_suc_img = 0
PerD = PerceptualDistance(opt)
img_list_for_fid = None
adv_img_list_for_fid = None
fid = 0

att = SSAH(model=classifier,
           num_iteration=opt.num_iteration,
           learning_rate=opt.learning_rate,
           device=device,
           Targeted=False,
           dataset=opt.dataset,
           m=opt.m,
           alpha=opt.alpha,
           lambda_lf=opt.lambda_lf,
           wave=opt.wavelet)

for batch, (inputs, targets) in enumerate(data):
    inputs = inputs.to(device)
    targets = targets.to(device)
    common_id = common(targets, predict(classifier, inputs, opt))
    total_img += len(common_id)
    inputs = inputs[common_id].cuda()
    targets = targets[common_id].cuda()

    # attack and calculate ASR
    adv = att(inputs)

    att_suc_id = attack_success(targets, predict(classifier, adv, opt))
    att_suc_img += len(att_suc_id)

    adv = adv[att_suc_id]
    inputs = inputs[att_suc_id]

    l2, l_inf, low_fre, ssim, CIEDE2000 = PerD.cal_perceptual_distances(inputs, adv)
    PerD.update(l2, l_inf, low_fre, ssim, CIEDE2000, adv.size(0))

    # Test the fid Value：we save the ori and adv img into .png profile and test them use fid
    # save the 5k imgs to test the fid
    if opt.test_fid:
        benign_img = os.path.join(opt.outdir, opt.dataset + '/' + 'benign-3521/')
        adv_img = os.path.join(opt.outdir, opt.dataset + '/' + 'adv-SSAH-3521/')
        if img_list_for_fid is None and adv_img_list_for_fid is None:
            img_list_for_fid = []
            adv_img_list_for_fid = []
            if not os.path.exists(benign_img):
                os.makedirs(benign_img)
            if not os.path.exists(adv_img):
                os.makedirs(adv_img)
        img_list_for_fid.append(inputs.detach())
        adv_img_list_for_fid.append(adv.detach())

        if att_suc_img >= 5000:
            opt.test_fid = False
            img_list_for_fid = torch.cat(img_list_for_fid,dim=0)
            adv_img_list_for_fid = torch.cat(adv_img_list_for_fid,dim=0)
            for id in range(5000):
                vutils.save_image(img_list_for_fid[id].detach(),
                                  '%s/%05d.png' % (benign_img, id),
                                  normalize=True,
                                  )
                vutils.save_image(adv_img_list_for_fid[id].detach(),
                                  '%s/%05d.png' % (adv_img, id),
                                  normalize=True,
                                  )
            fid = return_fid(benign_img, adv_img)
            del img_list_for_fid
            del adv_img_list_for_fid

    infostr = {"Evaluating Adversarial images of {} dataset ({} images) with perturb mode: {} :".format(
        opt.dataset, total_img, opt.perturb_mode)}
    logging.info(infostr)

    infostr = {"Batch={:<5} "
          "Fooling Rate: {:.2f}% "
          "L2: {:.2f} "
          "L_inf: {:.2f} "
          "SSIM: {:.2f} "
          "CIEDE2000: {:.2f} "
          "Low Frequency: {:.2f} "
          "FID Value: {:.2f}".format(batch,
                                     100.0 * att_suc_img / total_img,
                                     PerD.l2_avg,
                                     PerD.l_inf_avg,
                                     PerD.ssim_avg,
                                     PerD.CIEDE2000_avg,
                                     PerD.LF_avg,
                                     fid)}
    logging.info(infostr)

