#ImageNet1M
python  ../../main.py \
        --classifier='resnet50' \
        --dataset='imagenet_val'  \
        --bs=256 \
        --max_epoch=1 \
        --wavelet='haar' \
        --num_iteration=200 \
        --learning_rate=0.0001 \
        --m=0.2 \
        --alpha=1.0 \
        --beta=0.1\
        --seed=18
