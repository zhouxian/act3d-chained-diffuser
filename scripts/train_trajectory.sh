dataset=/projects/katefgroup/datasets/rlbench/diffusion_trajectories_train/
valset=/projects/katefgroup/datasets/rlbench/diffusion_trajectories_val/

main_dir=diffusion_9_20_multitask_2gpus

lr=1e-4
dense_interpolation=1
interpolation_length=50
B=22
ngpus=1

# CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node $ngpus --master_port $RANDOM \
CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    main_trajectory.py --tasks close_door \
    --dataset $dataset\
    --valset $valset \
    --instructions instructions.pkl \
    --gripper_loc_bounds 10_tough_diffusion_location_bounds.json \
    --num_workers 4 \
    --train_iters 500000 \
    --embedding_dim 120 \
    --action_dim 7 \
    --num_query_cross_attn_layers 6 \
    --use_instruction 1 \
    --use_goal 1 \
    --use_goal_at_test 1 \
    --feat_scales_to_use 1 \
    --attn_rounds 1 \
    --weight_tying 1 \
    --rotation_parametrization 6D \
    --diffusion_timesteps 100 \
    --val_freq 1000 \
    --dense_interpolation $dense_interpolation \
    --interpolation_length $interpolation_length \
    --exp_log_dir $main_dir \
    --batch_size $B \
    --batch_size_val 8 \
    --cache_size 0 \
    --cache_size_val 0 \
    --lr $lr\
    --run_log_dir diffusion_multitask-B$B-lr$lr-DI$dense_interpolation-$interpolation_length
