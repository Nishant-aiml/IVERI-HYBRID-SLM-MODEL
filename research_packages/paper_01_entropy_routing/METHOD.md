# Methodology: Entropy-Guided Dynamic Sparse Expert Routing

## 1. Local Byte Entropy Estimation
Given input byte stream $B = (b_1, b_2, \dots, b_N)$, local entropy is calculated over a sliding window $w=8$:

$$\mathcal{H}(b_t) = -\sum_{i=1}^{V} P(b_i) \log_2 P(b_i)$$

## 2. Dynamic Routing Gate
The routing gate conditions expert selection on entropy threshold $	au$:

$$k_t = egin{cases} 1 & 	ext{if } \mathcal{H}(b_t) < 	au_{	ext{low}} \ 2 & 	ext{if } 	au_{	ext{low}} \le \mathcal{H}(b_t) < 	au_{	ext{high}} \ 4 & 	ext{if } \mathcal{H}(b_t) \ge 	au_{	ext{high}} \end{cases}$$

## 3. Loss Function & Load Balancing
Total objective combines cross-entropy loss with expert load-balancing penalty:

$$\mathcal{L}_{	ext{total}} = \mathcal{L}_{	ext{CE}} + lpha \mathcal{L}_{	ext{balance}}$$
