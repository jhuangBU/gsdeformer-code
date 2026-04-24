
python train_cli.py \
    --source_path datasets/dnerf/mutant \
    --model_path outputs_mutant --deform_type node --node_num 512 --hyper_dim 8 \
    --src_mesh outputs_mutant_node/synth_mutant_v5_proxy.ply \
    --dst_mesh outputs_mutant_node/synth_mutant_v5_proxy_deformed2.ply \
    --output_ply ./deformed_gaussians_0.000_cli_v2.ply