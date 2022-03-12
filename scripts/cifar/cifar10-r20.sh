#CIFAR10 TEST 10K
python  ../../main.py \
        --classifier='resnet20' \
        --dataset='cifar10'  \
        --bs=10000 \
        --max_epoch=1 \
        --wavelet='haar' \
        --num_iteration=150 \
        --learning_rate=0.001 \
        --m=0.2 \
        --alpha=1 \
        --beta=0.1\
        --seed=18
