#ImageNet-1K
#SSAH
python  main.py \
        --classifier='resnet50' \
        --dataset='imagenet_val'  \
        --bs=256 \
        --max-epoch=1 \
        --wavelet='haar' \
        --num-iteration=200 \
        --learning-rate=0.0001 \
        --m=0.2 \
        --alpha=1.0 \
        --lambda-lf=0.1\
        --seed=8

#SSA
#python  main.py \
#        --classifier='resnet50' \
#        --dataset='imagenet-val'  \
#        --bs=256 \
#        --max-epoch=1 \
#        --wavelet='haar' \
#        --num-iteration=200 \
#        --learning-rate=0.0001 \
#        --m=0.2 \
#        --alpha=1.0 \
#        --lambda-lf=0.0\
#        --seed=8
