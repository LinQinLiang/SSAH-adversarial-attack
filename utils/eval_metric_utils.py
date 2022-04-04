import torch
from attack.DWT import *


class PerceptualDistance(object):
    def __init__(self, opt):

        self.DWT = DWT_2D_tiny(wavename=opt.wavelet)
        self.IDWT = IDWT_2D_tiny(wavename=opt.wavelet)

        self.l2_avg = 0
        self.l_inf_avg = 0
        self.LF_avg = 0

        self.l2_sum = 0
        self.l_inf_sum = 0
        self.LF_sum = 0

        self.count = 0

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

        return l2/N, l_inf/N, low_fre/N

    def update(self, l2, l_inf, low_fre, n=1):

        self.l2_sum += l2 * n
        self.l_inf_sum += l_inf * n
        self.LF_sum += low_fre * n

        self.count += n

        self.l2_avg = self.l2_sum / self.count
        self.l_inf_avg = self.l_inf_sum / self.count
        self.LF_avg = self.LF_sum / self.count
