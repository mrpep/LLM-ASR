# ginpipe configs/base/eval.gin \
#         configs/datasets/mls.gin \
#         --module_list configs/imports \
#         --project_name asr-eval \
#         --experiment_name qwen1.5B-wavlm-catmlp-originalLR2 \
#         --mods "CKPT_PATH='/home/lpepino/LLM-ASR/experiments/asr/qwen1.5B-wavlm-catmlp-originalLR/epoch-7-step-1728.ckpt'"

# ginpipe configs/base/eval.gin \
#         configs/datasets/mls.gin \
#         --module_list configs/imports \
#         --project_name asr-eval \
#         --experiment_name qwen1.5B-mhubert-catmlp-downsample4-wa \
#         --mods "CKPT_PATH='models_to_evaluate/qwen1.5B-mhubert-catmlp-originalLR-addTedLium-weightedaverage-fixnograd/epoch-5-step-1848.ckpt'"

# ginpipe configs/base/eval.gin \
#         configs/datasets/mls.gin \
#         --module_list configs/imports \
#         --project_name asr-eval \
#         --experiment_name downsamplex4-mls \
#         --mods "CKPT_PATH='/home/lpepino/LLM-ASR/models_to_evaluate/qwen1.5B-mhubert-catmlp-originalLR-addTedLium-weightedaverage-fixnograd/epoch-5-step-1848.ckpt'"


# ginpipe configs/base/eval.gin \
#         configs/datasets/tedlium.gin \
#         --module_list configs/imports \
#         --project_name asr-eval \
#         --experiment_name downsamplex4-tedlium \
#         --mods "CKPT_PATH='/home/lpepino/LLM-ASR/models_to_evaluate/qwen1.5B-mhubert-catmlp-originalLR-addTedLium-weightedaverage-fixnograd/epoch-5-step-1848.ckpt'" \
#                "DEVICE=[1]"

# ginpipe configs/base/eval.gin \
#         configs/datasets/tedlium.gin \
#         --module_list configs/imports \
#         --project_name asr-eval \
#         --experiment_name downsamplex8-tedlium \
#         --mods "CKPT_PATH='/home/lpepino/LLM-ASR/models_to_evaluate/qwen1.5B-mhubert-catmlp-originalLR-addTedLium-weightedaverage-fixnograd-downsamplex8/epoch-5-step-1848.ckpt'" \
#                "DEVICE=[1]"

# ginpipe configs/base/eval.gin \
#         configs/datasets/mls.gin \
#         --module_list configs/imports \
#         --project_name asr-eval \
#         --experiment_name downsamplex8-mls \
#         --mods "CKPT_PATH='/home/lpepino/LLM-ASR/models_to_evaluate/qwen1.5B-mhubert-catmlp-originalLR-addTedLium-weightedaverage-fixnograd-downsamplex8/epoch-5-step-1848.ckpt'"

# ginpipe configs/base/eval.gin \
#         configs/datasets/tedlium.gin \
#         --module_list configs/imports \
#         --project_name asr-eval \
#         --experiment_name downsamplex16-tedlium \
#         --mods "CKPT_PATH='/home/lpepino/LLM-ASR/models_to_evaluate/qwen1.5B-mhubert-catmlp-originalLR-addTedLium-weightedaverage-fixnograd-downsamplex16/epoch-5-step-1848.ckpt'" \
#                "DEVICE=[1]"

# ginpipe configs/base/eval.gin \
#         configs/datasets/mls.gin \
#         --module_list configs/imports \
#         --project_name asr-eval \
#         --experiment_name downsamplex16-mls \
#         --mods "CKPT_PATH='/home/lpepino/LLM-ASR/models_to_evaluate/qwen1.5B-mhubert-catmlp-originalLR-addTedLium-weightedaverage-fixnograd-downsamplex16/epoch-5-step-1848.ckpt'" \
#                "DEVICE=[1]"

# ginpipe configs/base/eval.gin \
#         configs/datasets/mls.gin \
#         --module_list configs/imports \
#         --project_name asr-eval \
#         --experiment_name downsamplex4-frozen-mls \
#         --mods "CKPT_PATH='/home/lpepino/LLM-ASR/models_to_evaluate/qwen1.5B-mhubert-catmlp-originalLR-addTedLium-weightedaverage/epoch-7-step-2464.ckpt'" \
#                "DEVICE=[0]"

# ginpipe configs/base/eval.gin \
#         configs/datasets/tedlium.gin \
#         --module_list configs/imports \
#         --project_name asr-eval \
#         --experiment_name downsamplex4-frozen-tedlium \
#         --mods "CKPT_PATH='/home/lpepino/LLM-ASR/models_to_evaluate/qwen1.5B-mhubert-catmlp-originalLR-addTedLium-weightedaverage/epoch-7-step-2464.ckpt'" \
#                "DEVICE=[1]"

# ginpipe configs/base/eval.gin \
#         configs/datasets/tedlium.gin \
#         --module_list configs/imports \
#         --project_name asr-eval \
#         --experiment_name downsamplex4-onlymls-tedlium \
#         --mods "CKPT_PATH='/home/lpepino/LLM-ASR/models_to_evaluate/qwen1.5B-mhubert-catmlp-originalLR-addTedLium-weightedaverage-fixnograd-onlymls/epoch-5-step-1266.ckpt'" \
#                "DEVICE=[1]"

ginpipe configs/base/eval.gin \
        configs/datasets/mls.gin \
        --module_list configs/imports \
        --project_name asr-eval \
        --experiment_name downsamplex4-onlymls-mls \
        --mods "CKPT_PATH='/home/lpepino/LLM-ASR/models_to_evaluate/qwen1.5B-mhubert-catmlp-originalLR-addTedLium-weightedaverage-fixnograd-onlymls/epoch-5-step-1266.ckpt'" \
               "DEVICE=[0]"