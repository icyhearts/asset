
python3 examples/offline/offline_inference_recipe.py  --recipe  examples/recipes/qwen/qwen25_05b_instruct_dummy___mxint8.json > temp/qwen25_05b_instruct_dummy___mxint8.json.log.`nowstr.sh` 2>&1 &

        \

python examples/offline/offline_inference_recipe.py --recipe examples/recipes/kimi/kimi_linear_dummy.json --quantization simo --llm-arg 'hf_overrides.quantization_config_file="/share/users/like/package/simo_conda_vllm_sipu/simo/extensions/vllm_simo/example/simo_quantization_config/advanced_fp4_quantization/quant_config_sipu_w4a4_mxfp.json"' --env VLLM_PLUGINS=sipu,general,vllm_simo_extensions 
