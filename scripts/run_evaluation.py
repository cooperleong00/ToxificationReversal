import argparse
import os.path as osp
import sys

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import jsonlines as jsl
import torch
from mmengine import Config
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          GPT2LMHeadModel, set_seed)
from utils import batchify


def token_loss(lm_logits, labels):
    bsz = lm_logits.size(0)
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    ).view(bsz, -1)
    return loss


def run_ppl_evaluation(prompts, continuations):
    model_path = 'pretrained_models/gpt2-xl'
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.config.pad_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.eos_token_id
    model.to('cuda')
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    scores = []
    batch_size = 32
    pbar = tqdm(
        batchify(zip(prompts, continuations), batch_size), total=len(continuations)
    )
    for sample in pbar:
        p_texts, c_texts = zip(*sample)
        texts = []
        for p, c in sample:
            if p[-1] != ' ':
                c = ' ' + c.lstrip()
            texts.append(p + c)

        p_inputs = tokenizer(p_texts)
        inputs = tokenizer(texts, padding=True, return_tensors='pt')
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

        labels = inputs['input_ids'].clone()
        # ignore pad token
        labels[labels == tokenizer.pad_token_id] = -100
        # conditional ppl, ignore the prompt
        for i in range(len(p_texts)):
            labels[i, : len(p_inputs['input_ids'][i])] = -100

        with torch.no_grad():
            logits = model(**inputs).logits

        loss = token_loss(logits, labels)
        ppls = (loss.sum(dim=-1) / (loss != 0).sum(dim=-1)).exp().cpu().tolist()

        scores.extend(ppls)
        pbar.update(len(c_texts))
    pbar.close()
    return scores


def run_toxicity_evaluation(continuations):
    model = AutoModelForSequenceClassification.from_pretrained(
        'pretrained_models/deberta-v3-large_toxicity-scorer'
    )
    model.to('cuda')
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        'pretrained_models/deberta-v3-large_toxicity-scorer'
    )

    scores = []
    batch_size = 32
    pbar = tqdm(batchify(continuations, batch_size), total=len(continuations))
    for c_texts in pbar:
        inputs = tokenizer(c_texts, padding=True, return_tensors='pt')
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[:, 1].cpu().tolist()

        scores.extend(probs)
        pbar.update(len(c_texts))
    pbar.close()
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--eval_type", type=str, required=True)
    args = parser.parse_args()

    config = Config.fromfile(args.config)
    set_seed(config.seed)

    fn = ".".join(osp.basename(args.config).split('.')[:-1])
    results_dir = 'results/'

    results_fp = osp.join(results_dir, fn + '.jsonl')
    result = list(jsl.open(results_fp))
    prompts = []
    continuations = []
    for p in result:
        prompts.extend([p['prompt']['text']] * len(p['continuations']))
        continuations.extend([c['text'] for c in p['continuations']])

    if args.eval_type == "ppl":
        eval_fn = run_ppl_evaluation
        score_key = 'ppl'
        inputs = (prompts, continuations)
    elif args.eval_type == "toxicity":
        eval_fn = run_toxicity_evaluation
        score_key = 'toxicity'
        inputs = (continuations,)
    else:
        raise NotImplementedError

    output_fp = osp.join(results_dir, fn + '.jsonl')
    scores = eval_fn(*inputs)

    for i, s in enumerate(scores):
        result[i // config.num_k_samples]['continuations'][i % config.num_k_samples][
            score_key
        ] = s

    jsl.Writer(open(output_fp, 'w', encoding='utf-8')).write_all(result)
