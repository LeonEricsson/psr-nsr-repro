reproduction and further explorations based on *The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning*. Code is implemented based on [their repo](https://github.com/TianHongZXY/RLVR-Decomposed). Below is the original README.


## Quick Start
### Installation
Our code is implemented based on [verl](https://github.com/volcengine/verl). We recommend to use docker image provided by verl, please refer to their [documents](https://verl.readthedocs.io/en/v0.2.x/start/install.html).

Start from a custom environment:
```
conda create -y -n verl python=3.10.14 && conda activate verl
pip install -e .
pip install vllm==0.8.2
pip install latex2sympy2
pip install fire
pip install tensordict==0.7.2
python -m pip install flash-attn --no-build-isolation
```

## Training
PSR, NSR, W-REINFORCE: specify `advantage` in `run_qwen2.5-math-7b_psr_nsr.sh` to train the model with PSR, NSR, or W-REINFORCE. For W-REINFORCE, set `positive_advantage_weight`, which corresponds to the λ in the paper, recommended value is 0.1.
```
bash run_qwen2.5-math-7b_psr_nsr.sh
```
PPO
```
bash run_qwen2.5-math-7b_ppo.sh
```
GRPO
```
bash run_qwen2.5-math-7b_grpo.sh
```

## Evaluation
Specify `MODEL_PATH` and `OUTPUT_DIR` in `eval.sh`, then `bash eval.sh`.

Calculate Pass@k: `python calculate_metrics --file_path <file_to_evaluate>`

## Troubleshoot
- Out-of-Memory error: Decrease `actor_rollout_ref.actor.ppo_max_token_len_per_gpu`, `actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu`, `actor_rollout_ref.ref.log_prob_max_token_len_per_gpu`

- Frozen after `Started a local Ray instance.`: Add `num_cpus=N` to `ray.init()` in `verl/trainer/main_ppo.py`,  for example, `ray.init(num_cpus=4, runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})`
 
 ## Citation

If you find our paper or code useful, please consider cite our work:

```bibtex
@article{zhu2025rlvr-decomposed,
  title={The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning},
  author={Zhu, Xinyu and Xia, Mengzhou and Wei, Zhepei and Chen, Wei-Lin and Chen, Danqi and Meng, Yu},
  journal={arXiv preprint arXiv:2506.01347},
  year={2025}
}
```
