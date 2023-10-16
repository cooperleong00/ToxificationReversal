_base_ = [
    '../_base_/datasets/rtp-test-toxic-2k.py',
    '../_base_/common.py',
    '../_base_/models/gedi/gedi-gpt2-l.py',
]

disc_weight = 5