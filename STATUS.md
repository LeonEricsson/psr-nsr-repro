### Extended **current status**

**Scope & questions the paper tackles**
The work asks how *positive* versus *negative* reward signals drive learning in the **reinforcement-learning-with-verifiable-rewards (RLVR)** setting.  It algebraically splits the standard binary-reward objective into two on-policy pieces—**Positive-Sample Reinforcement (PSR)** and **Negative-Sample Reinforcement (NSR)**—making it possible to study them in isolation .

**Methods explored**

| Track                            | What they did                                                                                                                                                                                                                        | Why it matters                                                                   |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| **Pure PSR / Pure NSR**          | Trained Qwen2.5-Math-7B and Qwen3-4B on 7 500 MATH problems using only the +1 or −1 trajectories, respectively; compared to PPO & GRPO baselines, all with identical rollout machinery (1 024 prompts × 8 samples, verl framework).  | Clean ablation revealing each signal’s behavioural effect.                       |
| **Gradient analysis**            | Derived and visualised token-level gradients, showing PSR sharpens the sampled path whereas NSR redistributes mass toward prior-favoured alternatives .                                                                              | Provides a mechanistic account of diversity collapse vs. preservation.           |
| **Weighted-REINFORCE (λ = 0.1)** | Scales the PSR term and keeps NSR intact, i.e. $L = λ·L_{PSR}+L_{NSR}$ .                                                                                                                                                             | Seeks a middle ground between Pass\@1 accuracy (PSR) and high-k diversity (NSR). |
| **Benchmarks & metrics**         | Evaluated on MATH, AIME-2025, AMC23; full Pass\@k spectrum with unbiased estimator (k up to 256) .                                                                                                                                   | Captures both single-shot correctness and exploration capacity.                  |

**Empirical conclusions so far**

* **PSR** boosts Pass\@1 but entropy collapses, so performance degrades for k ≥ 8.&#x20;
* **NSR** alone “surprisingly” matches or beats PPO/GRPO across almost the entire Pass\@k curve—including k = 256—while maintaining base-model entropy.&#x20;
* **W-REINFORCE (λ = 0.1)** ties PPO at Pass\@1 and wins for most k≤64 on all three datasets, illustrating that *down-weighting* positive updates suffices.&#x20;

---

### High-level exploration plan

| Goal                       | Concrete actions                                                                      | Expected outcome                                                     |
| -------------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| **Reproduction**           | Reproduce GRPO, NSR, PSR results on different model family. Perform pass@k evaluation.| Matching paper                                                       |
| **Regularisation parity**  | Train a DAPO inspired baseline; No KL penalty, Clip-higher.                           | Apples-to-apples curves isolating the effect of + / – updates.       |
| **OOD generalisation**     | Re-use optimiser grid on **Reasoning Gym** and **LiveCodeBench**.                     | Evidence whether conclusions hold beyond math verification.          |

