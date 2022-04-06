import torch
from attack.DWT import *
from .differential_color_function import rgb2lab_diff, ciede2000_diff
from .ssim import SSIM


class PerceptualDistance(object):
    def __init__(self, opt):

        self.DWT = DWT_2D_tiny(wavename=opt.wavelet)
        self.IDWT = IDWT_2D_tiny(wavename=opt.wavelet)
        self.SSIM = SSIM()

        self.l2_avg = 0
        self.l_inf_avg = 0
        self.LF_avg = 0
        self.ssim_avg = 0
        self.CIEDE2000_avg = 0

        self.l2_sum = 0
        self.l_inf_sum = 0
        self.LF_sum = 0
        self.ssim_sum = 0
        self.CIEDE2000_sum = 0

        self.count = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        #SSIM
        ssim = self.cal_ssim(references,perturbed)

        #perceptual color distance
        CIEDE2000 = self.cal_color_distance(references,perturbed)

        return l2/N, l_inf/N, low_fre/N, ssim/N, CIEDE2000/N

    def cal_color_distance(self, references,perturbed):
        reference_lab = rgb2lab_diff(references, self.device)
        perturbed_lab = rgb2lab_diff(perturbed, self.device)
        color_distance_map = ciede2000_diff(reference_lab, perturbed_lab, self.device)
        color_distance_map = color_distance_map.flatten(start_dim=1)
        norm = torch.norm(color_distance_map, dim=-1)
        return torch.sum(norm)

    def cal_ssim(self, references, perturbed):
        ret = self.SSIM
        ssim = torch.zeros(references.shape[0])
        for i in range(references.shape[0]):
            ssim[i] = ret.forward(references[i].unsqueeze(0) * 255, perturbed[i].unsqueeze(0) * 255)
        return torch.sum(ssim)

    def update(self, l2, l_inf, low_fre, ssim, CIEDE2000,  n=1):

        self.l2_sum += l2 * n
        self.l_inf_sum += l_inf * n
        self.LF_sum += low_fre * n
        self.ssim_sum += ssim * n
        self.CIEDE2000_sum += CIEDE2000 * n

        self.count += n

        self.l2_avg = self.l2_sum / self.count
        self.l_inf_avg = self.l_inf_sum / self.count
        self.LF_avg = self.LF_sum / self.count
        self.ssim_avg = self.ssim_sum / self.count
        self.CIEDE2000_avg = self.CIEDE2000_sum / self.count
