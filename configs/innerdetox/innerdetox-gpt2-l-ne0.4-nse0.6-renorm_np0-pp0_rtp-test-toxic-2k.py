_base_ = [
    '../_base_/models/innerdetox/innerdetox-gpt2-l.py',
    '../_base_/datasets/rtp-test-toxic-2k.py',
    '../_base_/common.py',
]

neg_prompt_idx = 0 #np
pos_prompt_idx = 0 # pp


innerdetox_hook = dict(
    type='BaseInnerDetoxHook',
    norm_exp=0.4, #ne
    neg_sim_exp=0.6, #nse
    renorm=True,
)
