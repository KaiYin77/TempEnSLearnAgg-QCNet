CUDA_VISIBLE_DEVICES=1 \
    python train.py --dataset argoverse_v2 --model TempEnsLearnAgg \
    --root /media/ee904/ssd/argoverse_v2_qcnet/ --ckpt_path ./pretrain/QCNet_AV2.ckpt \
    --train_batch_size 4 --val_batch_size 4 --test_batch_size 4 \

