#CIFAR10 TEST 10K
#SSAH
python  main.py \
        --classifier='resnet20' \
        --dataset='cifar10'  \
        --bs=5000 \
        --max-epoch=1 \
        --wavelet='haar' \
        --num-iteration=150 \
        --learning-rate=0.001 \
        --m=0.2 \
        --alpha=1 \
        --beta=0.1\
        --seed=8\
        --workers=32\
        --test-fid

#SSA
#python  main.py \
#        --classifier='resnet20' \
#        --dataset='cifar10'  \
#        --bs=5000 \
#        --max-epoch=1 \
#        --wavelet='haar' \
#        --num-iteration=150 \
#        --learning-rate=0.001 \
#        --m=0.2 \
#        --alpha=1 \
#        --beta=0.0\
#        --seed=8\
#        --workers=32\
#        --test-fid