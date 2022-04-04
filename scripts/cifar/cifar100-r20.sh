#CIFAR100 TEST 10K
#python  ../../main.py \
#        --classifier='resnet20' \
#        --dataset='cifar100'  \
#        --bs=5000 \
#        --max_epoch=1 \
#        --wavelet='haar' \
#        --num_iteration=150 \
#        --learning_rate=0.0015 \
#        --m=0.2 \
#        --alpha=1 \
#        --beta=0.01\
#        --seed=8\
#        --test_fid=True


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
        --beta=0.0\
        --seed=8\
        --test-fid
