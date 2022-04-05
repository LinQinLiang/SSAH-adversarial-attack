import torch
import torch.nn.functional as F
from math import exp
import numpy as np
from attack.DWT import *
from differential_color_functions import rgb2lab_diff, ciede2000_diff


class PerceptualDistance(object):
    def __init__(self, opt):

        self.DWT = DWT_2D_tiny(wavename=opt.wavelet)
        self.IDWT = IDWT_2D_tiny(wavename=opt.wavelet)
        self.SSIM = SSIM()

        self.l2_avg = 0
        self.l_inf_avg = 0
        self.LF_avg = 0
        self.ssim_avg = 0

        self.l2_sum = 0
        self.l_inf_sum = 0
        self.LF_sum = 0
        self.ssim_sum = 0

        self.count = 0
        
        self.device = torch.device('cuda')

    def cal_perceptual_distances(self, references, perturbed):
        # l_p norm
        N = references.size(0)
        noise = (perturbed - references).flatten(start_dim=1)
        l2 = torch.sum(torch.pow(torch.norm(noise, p=2, dim=-1), 2))
        l_inf = torch.sum(torch.norm(noise, p=float('inf'), dim=-1))

        #LF
        img_ll = self.DWT(references)
        img_ll = self.IDWT(img_ll)

        adv_ll = self.DWT(perturbed)
        adv_ll = self.IDWT(adv_ll)

        noise = (adv_ll - img_ll).flatten(start_dim=1)
        norm = torch.norm(noise, p=2, dim=-1)
        low_fre = torch.sum(torch.pow(norm, 2))

        ssim = self.cal_ssim(references,perturbed)
        
        c = self.cal_color_distance(references,perturbed)

        return l2/N, l_inf/N, low_fre/N, ssim/N, c
    
    def cal_color_distance(self, references,perturbed):
        reference_lab = rgb2lab_diff(references, self.device)
        perturbed_lab = rgb2lab_diff(perturbed, self.device)
        color_distance_map = ciede2000_diff(reference_lab, perturbed_lab, self.device)
        color_distance_map = color_distance_map.flatten(start_dim=1)
        norm = torch.norm(color_distance_map, dim=-1)
        return torch.mean(norm)
        

    def cal_ssim(self, references, perturbed):
        ret = self.SSIM
        ssim = torch.zeros(references.shape[0])
        for i in range(references.shape[0]):
            ssim[i] = ret.forward(references[i].unsqueeze(0) * 255, perturbed[i].unsqueeze(0) * 255)
        return torch.sum(ssim)

    def update(self, l2, l_inf, low_fre, ssim, n=1):

        self.l2_sum += l2 * n
        self.l_inf_sum += l_inf * n
        self.LF_sum += low_fre * n
        self.ssim_sum += ssim * n

        self.count += n

        self.l2_avg = self.l2_sum / self.count
        self.l_inf_avg = self.l_inf_sum / self.count
        self.LF_avg = self.LF_sum / self.count
        self.ssim_avg = self.ssim_sum / self.count


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def c_ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return c_ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)
