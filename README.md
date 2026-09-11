[![Tests](https://github.com/bushesarebetter/tiny-gpt/actions/workflows/tests.yml/badge.svg)](https://github.com/bushesarebetter/tiny-gpt/actions/workflows/tests.yml)

# tiny-gpt

A character-level, decoder-only transformer in PyTorch: causal multi-head self-attention, pre-LayerNorm blocks, learned positional embeddings, and optional weight tying between the token embedding and the output head. It trains on CPU in a few minutes.

The training text is `shakespeare.txt` (65 KB, about 1,700 lines). I typed it out from public-domain Shakespeare: a few dozen sonnets plus well-known speeches and songs from the plays. It's a small corpus and the model overfits quickly. The samples look like Shakespeare from across the room and fall apart up close.

## Setup

```
pip install -r requirements.txt
```

## Run

```
python tinygpt.py train --quick      # 0.6M params, 600 steps, ~4-5 min
python tinygpt.py train              # 0.8M params, 1500 steps, ~25 min on my laptop
python tinygpt.py sample --prompt "JULIET:" --length 500 --temperature 0.8 --top-k 20
```

Training writes `ckpt.pt` (gitignored) and prints a short sample at the end. `--no-tie` unties the embedding and output weights. On my laptop more than 4 threads ran slower, which is why `--threads` defaults to 4.

## Results

| run | steps | final train loss | final val loss | time |
|---|---|---|---|---|
| `--quick` | 600 | 1.841 | 2.072 | 4.7 min |
| default | 1500 | 1.404 | 1.980 | 27 min |

Losses are mean cross-entropy per character in nats, over a 90/10 train/val split of the text. In the default run, val loss bottoms out at 1.964 around step 1250 and then starts to rise while train loss keeps dropping, so it is overfitting by then. Timings were taken with other jobs running on the machine, so expect faster runs on an idle CPU.

Sample from the default checkpoint (temperature 0.7, top-k 10):

```
JULIET:
So more the sweetings of their stands,
To this stand speaks of these of your shing?
```

## Test

```
python -m pytest -q
```

The tests check three things. Changing future tokens leaves earlier logits unchanged (the causal mask). Loss drops over 30 steps on a tiny fixed batch, with tied and untied weights. Sampling returns the requested number of tokens, even past the context length.
