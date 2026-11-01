Can reinforcement learning teach a language model to work around the way its tokenizer chops up DNA?

I trained Qwen3-0.6B with GRPO on reverse-complementing DNA, showing the same sequences three ways: ordinary BPE chunks, one token per base, and Carbon-X-style 6-mers.

With an exact-match reward, it got 0 of 4,568 training attempts right, so RL had literally nothing to learn from. With partial credit, it found a shortcut: up to 98% of its answers became a verbatim copy of the input, which always has the right length and scores about 25% by luck.

The animation shows the grey wrong-length rows filling in over training while the correct bases stay at chance. RL amplifies skills a model already has, and it will game a reward before it learns a missing one.

Code, figures and every number: github.com/silvererudite/frameshift-rl
