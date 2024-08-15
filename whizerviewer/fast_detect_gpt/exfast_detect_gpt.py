# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import numpy as np
import tqdm
import argparse
import json
import time
from data_builder import load_data, save_data
from metrics import get_roc_metrics, get_precision_recall_metrics
import os
import sys

class OpenAIGPT:
    def __init__(self, args):
        from openai import AzureOpenAI
        self.args = args
        self.client = AzureOpenAI(
            azure_endpoint=args.azure_endpoint,
            api_key=args.api_key,
            api_version=args.api_version)  #

    def eval(self, text):
        while True:
            try:
                # get top alternative tokens
                nprefix = 1
                kwargs = {"model": self.args.azure_model, "max_tokens": 0, "echo": True, "logprobs": self.args.top_k}
                response = self.client.completions.create(prompt=f"<|endoftext|>{text}", **kwargs)
                result = response.choices[0]
                tokens = result.logprobs.tokens[nprefix:]
                logprobs = result.logprobs.token_logprobs[nprefix:]
                toplogprobs = result.logprobs.top_logprobs[nprefix:]
                toplogprobs = [dict(item) for item in toplogprobs]
                assert len(tokens) == len(logprobs), f"Expected {len(tokens)} logprobs, got {len(logprobs)}"
                assert len(tokens) == len(toplogprobs), f"Expected {len(tokens)} toplogprobs, got {len(toplogprobs)}"
                return tokens, logprobs, toplogprobs
            except Exception as ex:
                print(ex)
                print('Sleep 10 seconds before retry ...')
                time.sleep(10)


def estimate_distrib_token(toplogprobs, top_k):
    M = 10000  # assuming vocabulary size
    K = top_k  # assuming top-K tokens
    toplogprobs = sorted(toplogprobs.values(), reverse=True)
    assert len(toplogprobs) >= K
    toplogprobs = toplogprobs[:K]
    probs = []  # distribution over ranks
    probs.extend(np.exp(toplogprobs))  # probabilities of the top-K tokens
    p_K = probs[-1]  # the k-th top token
    p_rest = 1 - np.sum(probs)  # the rest probability mass
    alpha = p_rest / (p_K + p_rest)  # approximate the decay factor
    assert alpha ** (M - K + 1) < 1e-6
    # If the assertion is not satisfied, use the following code to calculate the decay factor iteratively
    # alpha0 = 0
    # while abs(alpha - alpha0) > 1e-6:
    #     alpha0 = alpha
    #     beta = alpha ** (M - K + 1)  # the minor part
    #     alpha = 1 - (alpha - beta) * p_K / p_rest
    # estimate the probabilities of the rest tokens
    while len(probs) < M:
        probs.append(probs[-1] * alpha)
    assert abs(np.sum(probs) - 1.0) < 1e-6, f'Invalid total probability: {np.sum(probs)}'
    return probs

from scipy.special import zeta

def estimate_distrib_token_p_series(toplogprobs, top_k):
    ### use p series to estimate the distribution  ## zmj
    M = 10000  # assuming vocabulary size
    K = top_k  # assuming top-K tokens
    toplogprobs = sorted(toplogprobs.values(), reverse=True)
    assert len(toplogprobs) >= K
    toplogprobs = toplogprobs[:K]
    
    probs = []  # distribution over ranks
    probs.extend(np.exp(toplogprobs))  # probabilities of the top-K tokens
    p_rest = 1 - np.sum(probs)  # the rest probability mass
    
    # Calculate the Riemann zeta function for p=2, starting from K+1
    zeta_K = zeta(2) - sum(1/i**2 for i in range(1, K+1))
    
    # Calculate the scaling factor
    scale = p_rest / zeta_K
    
    # Estimate the probabilities of the rest tokens
    tail_probs = [scale / (i**2) for i in range(K+1, M+1)]
    
    # Normalize only the tail probabilities to ensure they sum to p_rest
    tail_probs = np.array(tail_probs) * (p_rest / np.sum(tail_probs))
    
    # Combine the top-K probabilities with the tail probabilities
    probs.extend(tail_probs)
    
    assert abs(np.sum(probs) - 1.0) < 1e-6, f'Invalid total probability: {np.sum(probs)}'
    
    return probs

def estimate_distrib_sequence(toplogprobs, top_k):
    # probs = [estimate_distrib_token(v, top_k) for v in toplogprobs]
    probs = [estimate_distrib_token_p_series(v, top_k) for v in toplogprobs]
    return np.array(probs)

def get_sampling_discrepancy_analytic(args, logprobs, toplogprobs):
    ## 抽样偏差计算 ？
    log_likelihood = np.array(logprobs)
    probs = estimate_distrib_sequence(toplogprobs, args.top_k) # 函数估计完整的概率分布
    lprobs = np.nan_to_num(np.log(probs)) # 计算估计概率的对数,并将可能的NaN值转换为0。
    mean_ref = (probs * lprobs).sum(axis=-1) # 计算参考分布的平均对数似然。这是交叉熵的负值。
    lprobs2 = np.nan_to_num(np.square(lprobs))
    var_ref = (probs * lprobs2).sum(axis=-1) - np.square(mean_ref)
    discrepancy = (log_likelihood.sum(axis=-1) - mean_ref.sum(axis=-1)) / np.sqrt(var_ref.sum(axis=-1))
    discrepancy = discrepancy.mean()
    return discrepancy.item()


def experiment(args):
    gpt = OpenAIGPT(args)
    # load data
    data = load_data(args.dataset_file)
    n_samples = len(data["sampled"])
    # n_samples = 3 # debug
    # evaluate criterion
    name = "exfast_detect_gpt"
    criterion_fn = get_sampling_discrepancy_analytic

    random.seed(args.seed)
    np.random.seed(args.seed)
    results = []
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]
        try:
            # original text
            tokens, logprobs, toplogprobs = gpt.eval(original_text) # token 概率和top k（一般是5）的概率列表
            original_crit = criterion_fn(args, logprobs, toplogprobs)
            # sampled text
            tokens, logprobs, toplogprobs = gpt.eval(sampled_text)
            sampled_crit = criterion_fn(args, logprobs, toplogprobs)
            # result
            results.append({"original": original_text,
                            "original_crit": original_crit,
                            "sampled": sampled_text,
                            "sampled_crit": sampled_crit})
        except Exception as ex:
            print(ex)

    # compute prediction scores for real/sampled passages
    predictions = {'real': [x["original_crit"] for x in results],
                   'samples': [x["sampled_crit"] for x in results]}
    print(f"Total {len(predictions['real'])}, Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
    # results
    results_file = f'{args.output_file}.{name}.{args.top_k}.json'
    results = { 'name': f'{name}_threshold',
                'info': {'n_samples': n_samples},
                'predictions': predictions,
                'raw_results': results,
                'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
                'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
                'loss': 1 - pr_auc}
    with open(results_file, 'w') as fout:
        json.dump(results, fout)
        print(f'Results written into {results_file}')


if __name__ == '__main__':
    # script_dir = os.path.dirname(os.path.realpath(__file__))
    
    sys.path.append('/zhuminjun/LLM/exfast-detect-gpt-main')
    os.chdir('/zhuminjun/LLM/exfast-detect-gpt-main')
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="/zhuminjun/LLM/exfast-detect-gpt-main/exp_gpt3to4/results/xsum_gpt-3.5-turbo.babbage-002")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_file', type=str, default="/zhuminjun/LLM/exfast-detect-gpt-main/exp_gpt3to4/data/xsum_gpt-3.5-turbo")
    parser.add_argument('--azure_model', type=str, default='babbage-002')  # babbage-002, davinci-002
    parser.add_argument('--azure_endpoint', type=str, default='https://westlake5.openai.azure.com/')
    parser.add_argument('--api_key', type=str, default='420a3a9868874fedbdb38b767a85fc52')
    parser.add_argument('--api_version', type=str, default='2023-09-15-preview')
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    experiment(args)
