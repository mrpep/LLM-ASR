ginpipe configs/base/eval.gin \
	configs/datasets/mls.gin \
	configs/batch/wav_instruction.gin \
	configs/models/llm-asr.gin \
        --module_list configs/imports \
        --project_name asr-eval \
        --experiment_name qwen1.5B-wavlm-catmlp-originalLR \
        --mods "CKPT_PATH='qwen1.5B-wavlm-catmlp-originalLR/epoch-7-step-1728.ckpt'"
