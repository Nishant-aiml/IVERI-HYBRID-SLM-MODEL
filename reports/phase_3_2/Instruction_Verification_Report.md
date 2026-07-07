# Instruction Verification Report — SFT Mathematical & Functional Validation

This report documents the functional and mathematical verification of the Supervised Fine-Tuning (SFT) components.

---

## 1. Autoregressive Shift & Alignment

In raw language modeling, inputs `x` and targets `y` are aligned directly. For SFT instruction tuning, the dataset must shift the tokens by 1 step for next-byte prediction:
- Input `x = tokens[:-1]`
- Target `y = tokens[1:]`

In `SFTByteDataset`, this shift is implemented and verified. The loss mask is aligned with the targets `y` by taking `loss_mask = full_mask[1:]`.

---

## 2. Masked Cross-Entropy Loss

The SFT loss is calculated selectively on target response bytes only:

$$\mathcal{L} = - \frac{1}{\sum_{i} m_i} \sum_{i} m_i \log P(y_i | x_{<i})$$

Where $m_i \in \{0, 1\}$ is the boolean mask for the target token $y_i$:
- $m_i = 1$ if position $i$ corresponds to an assistant response byte and is not a padding byte.
- $m_i = 0$ otherwise.

This is implemented in `sft_runner.py` by flat-masking logits and targets before computing the Cross-Entropy loss.

---

## 3. Formatter Span Correctness

We verified that `ConversationFormatter` maps prompt and response boundaries exactly to the byte level. UTF-8 multi-byte characters are parsed correctly (lengths are counted in raw bytes, not characters), maintaining exact target alignment under truncation and padding.
