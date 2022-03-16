import numpy as np
import torch
import random
import argparse
from collections import OrderedDict
import torchvision
import torchvision.utils as vutils
import os

from utils import *
from checkpoints.resnet import ResNet
from attack.ssah_attack import *
from utils.eval_metric_utils import *
from utils.auxiliary_utils import *
from utils.fid_score import return_fid


def parse_arg():
    parser = argparse.ArgumentParser(description='attack with feature layer and frequency constraint')
    parser.add_argument('--bs', type=int, default=10000, help="batch size")
    parser.add_argument('--dataset', type=str, default='cifar10', help='data to attack')
    parser.add_argument('--classifier', type=str, default='resnet20', help='model to attack')
    parser.add_argument('--seed', type=int, default=18, help='random seed')
    parser.add_argument('--perturb_mode', type=str, default='SSAH', help='attack method')
    parser.add_argument('--max_epoch', type=int, default=1, help='always 1 in attack')
    parser.add_argument('--workers', type=int, default=8, help='num workers to load img')
    parser.add_argument('--wavelet', type=str, default='haar', choices=['haar', 'Daubechies', 'Cohen'])
    parser.add_argument('--test_fid', type=bool, default=False, help='test fid value')

    # SSAH Attack Parameters
    parser.add_argument('--num_iteration', type=int, default=150, help='MAX NUMBER ITERATION')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='LEARNING RATE')
    parser.add_argument('--m', type=float, default=0.2, help='MARGIN')
    parser.add_argument('--alpha', type=float, default=1.0, help='HYPER PARAMETER FOR ADV COST')
    parser.add_argument('--beta', type=float, default=0.1, help='HYPER PARAMETER FOR LOW FREQUENCY CONSTRAINT')
    parser.add_argument('--outdir', type=str, default='../../result', help='dir to save the attack examples')

    args = parser.parse_args()

    return args


opt = parse_arg()
device = torch.device("cuda")

if opt.classifier == 'resnet20' and opt.dataset == 'cifar10':
    path = '../../checkpoints/cifar10-r20.pth.tar'
    checkpoint = torch.load(path)
    state = checkpoint['state_dict']
    classifier = ResNet(20, 10)
    classifier.load_state_dict(state)
elif opt.classifier == 'resnet20' and opt.dataset == 'cifar100':
    path = '../../checkpoints/cifar100-r20.pth.tar'
    checkpoint = torch.load(path)
    state = checkpoint['state_dict']
    classifier = ResNet(20, 100)
    new_state_dict = OrderedDict()
    for k, v in state.items():
        name = k[7:]  # remove `module.`
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
print("Attack Benign Image of {} dataset({} images) with perturb mode: {} :".format(
    opt.dataset, num_images, opt.perturb_mode
))

l2 = 0
inf = 0
lowFre = 0
total_img = 0
att_suc_img = 0

att = SSAH(model=classifier,
           num_iteration=opt.num_iteration,
           learning_rate=opt.learning_rate,
           device=torch.device('cuda'),
           Targeted=False,
           dataset=opt.dataset,
           m=opt.m,
           alpha=opt.alpha,
           beta=opt.beta,
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

    lp = LpDistance(inputs, adv, opt)
    l2 += lp.Lp2()
    inf += lp.Lpinf()
    lowFre += lp.LowFreNorm()

    # Test the fid Value：we save the ori and adv img into .png profile and test them use fid
    # save the 5k imgs to test the fid
    if opt.test_fid:
        if batch == 0:
            benign_img = os.path.join(opt.outdir, opt.dataset + '/' + 'benign-SSA-H3521/')
            adv_img = os.path.join(opt.outdir, opt.dataset + '/' + 'adv-SSAH-3521/')
            if not os.path.exists(benign_img):
                os.makedirs(benign_img)
            if not os.path.exists(adv_img):
                os.makedirs(adv_img)
            for id in range(adv.shape[0]):
                vutils.save_image(inputs[id].detach(),
                                  '%s/%5d.png' % (benign_img, id),
                                  normalize=True,
                                  )
                vutils.save_image(adv[id].detach(),
                                  '%s/%5d.png' % (adv_img, id),
                                  normalize=True,
                                  )
            fid = return_fid(benign_img, adv_img)
    else:
        fid = 0

print("Evaluating Adversarial images of {} dataset({} images) with perturb mode:{} :".format(
    opt.dataset, total_img, opt.perturb_mode))
print("Batch={:<5} "
      "Fooling Rate: {:.2f}% "
      "L2 Norm: {:.2f} "
      "Lp Norm: {:.2f} "
      "Low Frequency Norm: {:.2f} "
      "FID Value: {:.2f}".format(batch,
                                 100.0 * att_suc_img / total_img,
                                 l2 / att_suc_img,
                                 inf / att_suc_img,
                                 lowFre / att_suc_img,
                                 fid))

