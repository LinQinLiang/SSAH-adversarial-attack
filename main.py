import numpy as np
import torch
import random
import argparse
from collections import OrderedDict
import torchvision

from utils import *
from checkpoints.resnet import ResNet
from attack.ssah_attack import *
from utils.eval_metric_utils import *
from utils.auxiliary_utils import *


def parse_arg():
    parser = argparse.ArgumentParser(description='attack with feature layer and frequency constraint')
    parser.add_argument('--bs', type=int, default=10000, help="batch size")
    parser.add_argument('--dataset', type=str, default='cifar10', help='data to attack')
    parser.add_argument('--classifier', type=str, default='resnet20', help='model to attack')
    parser.add_argument('--seed', type=int, default=18, help='random seed')
    parser.add_argument('--perturb_mode', type=str, default='SSAH', help='attack method')
    parser.add_argument('--max_epoch', type=int, default=1, help='always 1 in attack')
    parser.add_argument('--workers', type=int, default=8, help='num workers to load img')

    args = parser.parse_args()

    return args


def main():
    opt = parse_arg()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    if opt.classifier == 'resnet20' and opt.dataset == 'cifar10':
        path = './checkpoints/cifar10-r20.pth.tar'
        checkpoint = torch.load(path)
        state = checkpoint['state_dict']
        classifier = ResNet(20, 10)
        classifier.load_state_dict(state)
    elif opt.classifier == 'resnet20' and opt.dataset == 'cifar100':
        path = './checkpoints/cifar100-r20.pth.tar'
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

    print('=> attack classifier:', opt.classifier)

    start_epoch = 0

    for _ in range(start_epoch, opt.max_epoch):
        attack(data, classifier, opt)

    print('Attack finished')


def attack(data, classifier, opt):
    best_l2 = 0
    best_inf = 0
    best_lowFre = 0
    total_img = 0
    att_suc_img = 0

    att = SSAH(model=classifier,
               num_iteration=150,
               learning_rate=0.001,
               device=torch.device('cuda'),
               Targeted=False,
               dataset=opt.dataset,
               m=0.2,
               alpha=1,
               beta=0.1)

    for batch, (inputs, targets) in enumerate(data):
        # img has true prediction label
        common_id = common(targets, predict(classifier, inputs.cuda(), opt))
        total_img += len(common_id)
        inputs = inputs[common_id].cuda()
        targets = targets[common_id].cuda()

        # attack and calculate ASR
        adv = att(inputs)

        att_suc_id = attack_success(targets, predict(classifier, adv, opt))
        att_suc_img += len(att_suc_id)

        adv = adv[att_suc_id]
        inputs = inputs[att_suc_id]

        lp = LpDistance(inputs, adv)
        best_l2 += lp.Lp2()
        best_inf += lp.Lpinf()
        best_lowFre += lp.LowFreNorm()

        print("Evaluating Adversarial images of {} dataset({} images) with perturb mode:{} :".format(
            opt.dataset, total_img, opt.perturb_mode))
        print("Batch={:<5} "
              "Fooling Rate: {:.2f}% "
              "L2 Norm: {:^3} "
              "Lp Norm: {:^3} "
              "Low Frequency Norm: {:^3}".format(batch,
                                                 100.0 * att_suc_img / total_img,
                                                 best_l2 / att_suc_img,
                                                 best_inf / att_suc_img,
                                                 best_lowFre / att_suc_img))


if __name__ == '__main__':
    main()
