# cpc
Character prefix conditioning implementation
Author: [Devin Shah](https://dshah.dev)

We want to sample a sequence of tokens $s = {t_1,t_2,...,t_n}$ from a distribution specified by an autoregressive model $p(s)$ given by:

$$p(s) = p(t_1,t_2,\dots,t_n)=\prod_{k=1}^n p(t_k|t_1,\dots,t_{k-1})$$

subject to the constraint that s starts with a character prefix P, i.e., P is a prefix of repr(t₁) + repr(t₂) + ... + repr(tₙ),
where + means string concatenation and repr maps a token to the characters it represents.

We define $q(s) = p(s | s$ starts with $P)$. It's sufficient to find a way to sample autoregressively from $q(s)$, that is, to sample from $q(t_k|t_1,...,t_{k-1})$ for each $k$.

Full problem described [here](https://cursor.com/blog/cpc).
