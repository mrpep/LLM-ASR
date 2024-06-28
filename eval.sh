# ginpipe configs/base/eval.gin \
#         configs/datasets/mls.gin \
#         --module_list configs/imports \
#         --project_name asr-eval \
#         --experiment_name qwen1.5B-wavlm-catmlp-originalLR2 \
#         --mods "CKPT_PATH='/home/lpepino/LLM-ASR/experiments/asr/qwen1.5B-wavlm-catmlp-originalLR/epoch-7-step-1728.ckpt'"

ginpipe configs/base/eval.gin \
        configs/datasets/mls.gin \
        --module_list configs/imports \
        --project_name asr-eval \
        --experiment_name qwen1.5B-mhubert-catmlp-originalLR-addTedLium \
        --mods "CKPT_PATH='/home/lpepino/LLM-ASR/experiments/asr/qwen1.5B-mhubert-catmlp-originalLR-addTedLium/epoch-7-step-2464.ckpt'"