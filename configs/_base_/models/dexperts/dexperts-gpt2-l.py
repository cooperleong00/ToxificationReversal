model_type = 'dexperts'

model_path = 'pretrained_models/gpt2-large'
antiexpert_model = 'pretrained_models/dexperts/experts/toxicity/large/finetuned_gpt2_toxic'
expert_model = 'pretrained_models/dexperts/experts/toxicity/large/finetuned_gpt2_nontoxic'

batch_size = 64

dexperts_alpha = 2.0