# ginpipe configs/base/train.gin \
#         configs/datasets/mls.gin \
#         configs/batch/wav_instruction.gin \
#         configs/models/llm-asr.gin \
#         configs/models/audio_encoders/wavlm.gin \
#         configs/models/layer_selectors/select1.gin \
#         configs/models/adapters/cat_downsample.gin \
#         --module_list configs/imports \
#         --project_name asr \
#         --experiment_name qwen1.5B-wavlm-catmlp-originalLR

ginpipe configs/base/train.gin \
        configs/datasets/mls.gin \
        configs/datasets/tedlium.gin \
        configs/batch/wav_instruction.gin \
        configs/models/llm-asr.gin \
        configs/models/audio_encoders/wavlm.gin \
        configs/models/layer_selectors/select1.gin \
        configs/models/adapters/cat_downsample.gin \
        --module_list configs/imports \
        --project_name asr \
        --experiment_name qwen1.5B-wavlm-catmlp-originalLR-addTedLium