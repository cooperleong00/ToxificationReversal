seed = 2023
num_k_samples = 25

generation_config = dict(
    max_new_tokens=20,
    min_new_tokens=5,
    top_p=0.9,
    do_sample=True,
)

neg_prompts = [
    "The following text is abusive, harmful, negative, obscene, racist, rude and toxic: ",
]

pos_prompts = [
    "The following text is kind, polite, positive, respectful and supportive: ",
    ""
]