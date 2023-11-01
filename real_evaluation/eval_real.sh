task=real_reach_target
task=real_press_stapler
task=real_press_hand_san
task=real_put_fruits_in_bowl
task=real_stack_bowls
task=real_unscrew_bottle_cap
task=real_spread_sand
task=real_wipe_coffee
task=real_put_duck_in_oven
task=real_transfer_beans

gripper_bounds_buffer=0.10
weight_tying=1

gp_emb_tying=1
simplify=1
num_sampling_level=3
num_ghost_points_val=20000
embedding_dim=60
n_layer=2
use_instruction=1

simplify_ins=0
ins_pos_emb=1
vis_ins_att=1
vis_ins_att_complex=0
regress_position_offset=0
ins_pos_emb=0
instruction_file=instructions_real.pkl
ckpt=/home/zhouxian/git/hiveformer/train_logs/05_15_real/real_transfer_beans-offset0-N3-T1000-V10000-symrot0-gptie1-simp1-B16-demo100-dim60-L2-lr1e-4-seed0-simpins0-ins_pos_emb0-vis_ins_att1-vis_ins_att_complex0-insinstructions_real.pkl_version170518/model.step=110000-value=0.00000.pth
python eval_real.py\
     --instructions instructions_old/$instruction_file \
     --task $task\
     --variation 0\
     --checkpoint $ckpt \
     --gripper_loc_bounds_file /home/zhouxian/git/hiveformer/tasks/real_tasks_location_bounds.json\
     --weight_tying $weight_tying\
     --gp_emb_tying $gp_emb_tying\
     --simplify $simplify\
     --simplify_ins $simplify_ins\
     --ins_pos_emb $ins_pos_emb\
     --vis_ins_att $vis_ins_att\
     --vis_ins_att_complex $vis_ins_att_complex\
     --image_size 256,256\
     --offline 0\
     --num_episodes 100\
     --use_instruction $use_instruction\
     --num_ghost_points_val $num_ghost_points_val\
     --gripper_bounds_buffer $gripper_bounds_buffer\
     --regress_position_offset $regress_position_offset\
     --num_sampling_level $num_sampling_level\
     --embedding_dim $embedding_dim\
     --num_ghost_point_cross_attn_layers $n_layer\
     --num_query_cross_attn_layers $n_layer\
     # --max_episodes 20
