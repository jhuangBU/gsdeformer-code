
# CUDA_VISIBLE_DEVICES=0 python render.py --source_path datasets/dnerf/mutant --model_path outputs_mutant --deform_type node --node_num 512 --hyper_dim 8 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800

# CUDA_VISIBLE_DEVICES=0 python render.py --source_path datasets/dnerf/mutant --model_path outputs_mutant --deform_type node --node_num 512 --hyper_dim 8 --is_blender --eval --gt_alpha_mask_as_scene_mask --local_frame --resolution 2 --W 800 --H 800

python train_gui.py \
    --source_path datasets/dnerf/mutant \
    --model_path outputs_mutant --deform_type node --node_num 512 --hyper_dim 8 --gui \
    --src_mesh outputs_mutant_node/synth_mutant_v5_proxy.ply \
    --dst_mesh outputs_mutant_node/synth_mutant_v5_proxy_deformed2.ply