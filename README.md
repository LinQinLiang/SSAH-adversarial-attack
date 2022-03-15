Introduction
=
This is an official release of the CVPR2022 paper  **"Frequency-driven Imperceptible Adversarial Attack on Semantic Similarity"** (arxiv link). 
![Overview](https://github.com/LinQinLiang/SSAH-adversarial-attack/blob/main/imgs/overview.png)

__Abstract:__ Current adversarial attack research reveals the vulnerability of learning-based classifiers against carefully crafted perturbations. However, most existing attack methods have inherent limitations in cross-dataset generalization as they rely on a classification layer with a closed set of categories. Furthermore, the perturbations generated by these methods may appear in regions easily perceptible to the human visual system (HVS). To circumvent the former problem, we propose a novel algorithm that attacks semantic similarity on feature representations. In this way, we are able to fool classifiers without limiting attacks to a specific dataset. For imperceptibility, we introduce the low-frequency constraint to limit perturbations within high-frequency components, ensuring perceptual similarity between adversarial examples and originals. Extensive experiments on three datasets(CIFAR-10, CIFAR-100, and ImageNet-1K) and three public online platforms indicate that our attack can yield misleading and transferable adversarial examples across architectures and datasets. Additionally, visualization results and quantitative performance (in terms of four different metrics) show that the proposed algorithm generates more imperceptible perturbations than the state-of-the-art methods. Our code will be publicly available.

Requirements
=
* python ==3.6
* torch == 1.7.0
* torchvision >= 0.7
* numpy == 1.19.2
* Pillow == 8.0.1
* pywt

Required Dataset
=
1. The data structure of Cifar10, Cifar100, ImageNet or any other datasets look like below. Please modify the dataloader at `SSAH-Adversarial-master/main.py/` accordingly for your dataset structure.

```
/dataset/
├── Cifar10
│   │   ├── cifar-10-python.tar.gz
├── Cifar-100-python
│   │   ├── cifar-100-python.tar.gz
├── imagenet
│   ├── val
│   │   ├── n02328150

```

Experiments
=
We trained a resnet20 model with 92.6% accuracy with CIFAR1010 and a resnet20 model with 69.63% accuracy with CIFAR100. If you want to have a test, you can download our pre-trained models with the [Google Drivers](https://drive.google.com/drive/folders/1SrNrh7o7Ocok7w9ENuXROy9p_bC2IJVj?usp=sharing). If you want to use our algorithm to attack your own trained model, you can always replace our models in the file ```checkpoints```.

(1)Attack the Models Trained on Cifar10
-
```
CUDA_VISIBLE_DEVICES=0,1 bash scripts/cifar/cifar10-r20.sh
```
(2)Attack the Models Trained on Cifar100
-
```
CUDA_VISIBLE_DEVICES=0,1 bash scripts/cifar/cifar100-r20.sh
```
(2)Attack the Models Trained on Imagenet_val
-
```
CUDA_VISIBLE_DEVICES=0,1 bash scripts/cifar/Imagenet_val-r50.sh
```
***Examples***
-
![example](https://github.com/LinQinLiang/SSAH-adversarial-attack/blob/main/imgs/img.png)

***Results on CIFAR10***
-
|  Name   | Knowledge  |  ASR(%)  |  L2 |  Linf | FID | LF | Paper |
|  ----  | ----  |  ----  | ----  |   ----  | ----  |   ----  | ----  | 
| BIM  | White Box |  100.0 |   0.85   |   0.03     |   14.85     |  0.25       |    [ICLR2017](https://arxiv.org/pdf/1607.02533.pdf)    |
| PGD  | White Box|   100.0 |  1.28|  0.03     |   27.86    |   0.34    |    [arxiv link](https://arxiv.org/pdf/1706.06083.pdf)     |
| MIM  | White Box|   100.0 | 1.90  |   0.03     |     26.00  |   0.48    |     [CVPR2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Dong_Boosting_Adversarial_Attacks_CVPR_2018_paper.pdf)   |
| AutoAttack | White Box |100.0 |  1.91     |    0.03    |    34.93   |  0.61     |     [ICML2020](https://arxiv.org/pdf/2003.01690.pdf)    |
| AdvDrop | White Box | 99.92| 0.90      |   0.07    |   16.34    |   0.34    |     [ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Duan_AdvDrop_Adversarial_Attack_to_DNNs_by_Dropping_Information_ICCV_2021_paper.pdf)    |
| C&W  | White Box| 100.0 |   0.39   |     0.06  |   8.23    |   0.11    |     [IEEE SSP2017](https://arxiv.org/pdf/1608.04644.pdf)   |
| PerC-AL  | White Box | 98.29  | 0.86    |   0.18    |    9.58   |  0.15     |    [CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhao_Towards_Large_Yet_Imperceptible_Adversarial_Image_Perturbations_With_Perceptual_Color_CVPR_2020_paper.pdf)     |
| **SSA** | White Box |99.96  |  0.29    |     0.02  |    5.73   |   0.07    |     [CVPR2022]()    |
| **SSAH** | White Box | 99.94 |   0.26    |  0.02     |   5.03    |    0.03   |    [CVPR2022]()     |


Citation
=
if the code or method help you in the research, please cite the following paper:
```js
@InProceedings(luo2022ssah,
    author = {Luo, cheng and Lin, Qinliang and Xie, weicheng and Wu, Bizhu and Xie, Jinheng and Shen, LinLin},
    title = {Frequency-driven Imperceptible Adversarial Attack on Semantic Similarity},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {june},
    year = {2022}
}
```


