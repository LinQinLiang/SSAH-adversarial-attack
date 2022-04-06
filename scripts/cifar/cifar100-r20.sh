#CIFAR100 TEST 10K
#SSAH
python  main.py \
        --classifier='resnet20' \
        --dataset='cifar100'  \
        --bs=5000 \
        --max-epoch=1 \
        --wavelet='haar' \
        --num-iteration=150 \
        --learning-rate=0.0015 \
        --m=0.2 \
        --alpha=1 \
        --lambda-lf=0.01\
        --seed=8\
        --test-fid


#SSA
#python  main.py \
#        --classifier='resnet20' \
#        --dataset='cifar100'  \
#        --bs=5000 \
#        --max-epoch=1 \
#        --wavelet='haar' \
#        --num-iteration=150 \
#        --learning-rate=0.0015 \
#        --m=0.2 \
#        --alpha=1 \
#        --lambda-lf=0.0\
#        --seed=8\
#        --test-fid
