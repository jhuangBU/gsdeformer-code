#!/bin/bash

# quant_mic quant_ficus quant_hotdog quant_lego_90 quant_lego_135 quant_lego_180
for scene_name in quant_ficus quant_mic quant_hotdog quant_lego_90 quant_lego_135 quant_lego_180; do
    bash infer_gsdeformer_exp_quant.sh $scene_name
done