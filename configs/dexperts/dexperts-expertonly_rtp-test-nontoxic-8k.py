_base_ = [
    '../_base_/datasets/rtp-test-nontoxic-8k.py',
    '../_base_/common.py',
]

model_type = 'gpt2'

model_path = 'pretrained_models/dexperts/experts/toxicity/large/finetuned_gpt2_nontoxic'

batch_size = 64