# zkLLM: Zero Knowledge Proofs for Large Language Models

This repository benchmarks the overhead and accuracy drop in zkLLM. (Note: The README and repository are still under development.)

## Proving Time

Proving time, as reported in pioneering works, includes the time needed for sumcheck and commitment opening. We follow this standard, excluding irrelevant parts like compilation and loading.

We provide the following files to measure the overhead of corresponding tensor operations:
- `tlookup-bm.cu`
- `matmul-bm.cu`
- `hadamard-bm.cu`
- `com-bm.cu`

The first three files are for sumchecks only, while `com-bm.cu` handles commitments. Currently, proving time measurement for the entire inference process is not yet fully automated. The measurement involves listing all operations, storing and loading the input/output tensors of each operation using File I/Os, and then measuring the overheads of the sumcheck and commitment opening one by one.

### Quick Estimation

For a rough estimate of proving time, use:
- `tlookup-bm.py`
- `matmul-bm.py`
- `com-bm.py`
- `hadamard-bm.py`

These scripts measure proving time with tensors of designated sizes but filled with random values. The bottleneck is the attention mechanism or `zkAttn` in LLaMa-2, where the input size is `num_layer * num_head * seq_len * seq_len`, which is `32 * 32 * 2048 * 2048` (7B) or `40 * 40 * 2048 * 2048` (13B).

`zkAttn` involves 5 (or 2) `tlookup`s and two Hadamard products. The most time-consuming operations are listed below. The remaining operations have asymptotically lower complexity, ensuring the total proof time does not exceed 30s correspondingly.

- **Sumcheck of `tlookup`**: There are `40 * 40 = 1600` instances with input dimensions `2048 * 2048 = 4194304`. The input range is `[0, 65536)`. Multiply this by 2 or 5, the total number of `tlookup`s.
```bash
python tlookup-bm.py 1600 4194304 0 65536 # around 80 s
```
- **Sumcheck of Hadamard**: There are `40 * 40 = 1600` instances with tensor dimensions `2048 * 2048 = 4194304`.
```bash
python hadamard-bm.py 1600 4194304 # around 90 s
```

- **Openings**: There are `40 * 40 * 2 = 3200` instances (input and its multiplicative inverse for each `tlookup`) with tensor dimensions `2048 * 2048 = 4194304`. Multiply this by 2 or 5, the total number of `tlookup`s.
```bash
python com-bm.py 3200 4194304 65536 # around 15 s, 65536 is the generator dim
```

The above numbers are from an `A6000` GPU, which is slower than the `A100` used for the experiments. Moreover, even with the same model of GPU, the actual computing time can fluctuate by as much as 2x, hypothetically due to the "unnatural" data type from the lowest levels. The bottleneck costs `5 * 80 + 90 + 5 * 15 = 565` or `2 * 80 + 90 + 2 * 15 = 280` s for 13B models. This confirms the order of magnitude of the proving time.

## Accuracy (Perplexity)

Refer to `./acc`. Feedback indicates an overemphasis on preserving accuracy. Therefore, we are reducing the number of bits used, resulting in fewer `tlookup`s for `zkAttn`.
