# cpc
Character prefix conditioning implementation

We want to sample a sequence of tokens s = {t₁, t₂, ..., tₙ} from a distribution specified by an autoregressive model p(s) given by:

$p(s) = p(t₁, t₂, ..., tₙ) = Πₖ₌₁ⁿ p(tₖ | t₁, ..., tₖ₋₁)$

subject to the constraint that s starts with a character prefix P, i.e., P is a prefix of repr(t₁) + repr(t₂) + ... + repr(tₙ),
where + means string concatenation and repr maps a token to the characters it represents.

We define $q(s) = p(s | s$ starts with $P)$. It's sufficient to find a way to sample autoregressively from q(s), that is, to sample from $q(tₖ | t₁, ..., tₖ₋₁)$ for each k.

Full problem described [here](https://cursor.com/blog/cpc).
