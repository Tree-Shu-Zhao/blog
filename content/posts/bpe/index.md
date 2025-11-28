---
title: "Byte Pair Encoding"
date: 2025-11-28
draft: false
tags: ["LLM Fundamentals"]
categories: ["LLM"]
showToc: true
TocOpen: false
weight: 1
math: true
---

# 1. Introduction

Neural networks cannot process text directly, so we must convert raw strings into embeddings (vectors). Tokens serve as the bridge: raw text is first split into tokens, each mapped to an integer ID that indexes into an embedding look-up table. There are several approaches to tokenization:

- **Word-level tokenization** splits text by whitespace and punctuation, treating each word as a token. While intuitive, it cannot handle unseen words, which is a significant limitation in multilingual scenarios where vocabulary can explode.
- **Character-level tokenization** uses individual characters as tokens. This guarantees coverage of any word or sentence, but dramatically increases sequence length, which is a particular concern for transformer-based models where computational cost scales quadratically with length.
- **Subword tokenization** strikes a balance based on a simple insight: common words should remain intact, while rare words should be broken into meaningful subunits. For example, "tokenizers" might become ["token", "izers"], preserving semantic information while handling unseen combinations. This keeps vocabulary sizes manageable (typically 30K–50K tokens) while still allowing the representation of arbitrary text.

# 2. Byte Pair Encoding

Byte Pair Encoding (BPE) [1] is a subword tokenization algorithm. It builds a vocabulary through iterative merging. Starting from individual characters, it repeatedly finds the most frequent adjacent pair and merges it into a new token. After thousands of iterations, the vocabulary naturally captures common words as single tokens, frequent subwords as intermediate units, and individual characters as fallbacks for rare patterns.

## Training

1. Initialize the vocabulary with all unique characters in the corpus
2. Count the frequency of every adjacent token pair
3. Merge the most frequent pair into a new token
4. Repeat steps 2–3 until reaching the desired vocabulary size

## Encoding

1. Split input text into characters
2. Apply the learned merge rules in the same order they were learned
3. Map the resulting tokens to their integer IDs

## Example

Consider training BPE on the corpus: `"aab aab aac"`

### Initialization:

* Vocabulary: `{a, b, c, ' '}`
* Text as tokens: `[a, a, b, ' ', a, a, b, ' ', a, a, c]`

### Iteration 1:

* Pair frequencies: `(a, a): 3`, `(a, b): 2`, `(b, ' '): 2`, `(' ', a): 2`, `(a, c): 1`
* Merge `(a, a)` → new token `X`
* Text becomes: `[X, b, ' ', X, b, ' ', X, c]`

### Iteration 2:

* Pair frequencies: `(X, b): 2`, `(b, ' '): 2`, `(' ', X): 2`, `(X, c): 1`
* Merge `(X, b)` → new token `Y` (ties broken arbitrarily)
* Text becomes: `[Y, ' ', Y, ' ', X, c]`

After two merges, the common pattern `aab` is captured as a single token `Y`, while the rarer `aac` is denoted as `[X, c]`.

# 3. Pre-tokenization

Before applying BPE merges, GPT-2 [2] and subsequent OpenAI models first split text using a regex pattern. This pre-tokenization step serves a critical purpose: it prevents BPE from merging tokens across word boundaries or mixing semantically unrelated character types.

Without pre-tokenization, BPE might merge the space before a word with the word itself, or combine letters with digits, or fuse punctuation with adjacent text. These cross-boundary merges would create tokens like `" the"` or `"ing."` — units that conflate unrelated concepts and waste vocabulary capacity.

GPT-2 uses the regex pattern `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`. This pattern splits text into chunks, and **BPE operates within each chunk independently**. Let's examine each case:

## English contractions

**`'s|'t|'re|'ve|'m|'ll|'d`**

These patterns capture common English suffixes that attach to words via apostrophes. Isolating them ensures consistent tokenization of contractions.

| Input      | Tokens           |
| ---------- | ---------------- |
| `"I'm"`    | `["I", "'m"]`    |
| `"don't"`  | `["don", "'t"]`  |
| `"you've"` | `["you", "'ve"]` |
| `"she'll"` | `["she", "'ll"]` |

## Optional space followed by letters

`  ?\p{L}+`

`\p{L}` matches any Unicode letter. The optional space ` ?` keeps the leading space attached to the word, which is how GPT-2 represents word boundaries. This ensures BPE learns tokens like `" cat"` rather than merging spaces unpredictably.

| Input           | Tokens                |
| --------------- | --------------------- |
| `"hello world"` | `["hello", " world"]` |
| `"New York"`    | `["New", " York"]`    |

## Optional space followed by numbers

`  ?\p{N}+`

`\p{N}` matches any Unicode digit. This keeps numbers as separate chunks, preventing merges like `"price100"` becoming a single token.

| Input        | Tokens                 |
| ------------ | ---------------------- |
| `" 2024"`    | `[" 2024"]`            |
| `"price100"` | `["price", "100"]`     |
| `"room 42a"` | `["room", " 42", "a"]` |

## Optional space followed by punctuation/symbols

`  ?[^\s\p{L}\p{N}]+`

This matches anything that is not whitespace, letters, or numbers — effectively capturing punctuation and special characters as separate chunks.

| Input             | Tokens                          |
| ----------------- | ------------------------------- |
| `"wait..."`       | `["wait", "..."]`               |
| `"Hello, world!"` | `["Hello", ",", " world", "!"]` |
| `"$100"`          | `["$", "100"]`                  |

## Trailing whitespace (not followed by non-whitespace)

`\s+(?!\S)`

The negative lookahead `(?!\S)` ensures this matches whitespace only at the end of text or before more whitespace. This handles trailing spaces without merging them into subsequent words.

| Input          | Tokens               |
| -------------- | -------------------- |
| `"end   "`     | `["end", "   "]`     |
| `"text    \n"` | `["text", "    \n"]` |

## Remaining whitespace

`\s+`

This catches any whitespace sequences not matched by the previous pattern — typically single spaces between words, which are usually attached to the following word by the ` ?\p{L}+` pattern. It serves as a fallback for edge cases.

| Input      | Tokens                       |
| ---------- | ---------------------------- |
| `"a   b"`  | `["a", " ", " ", " ", " b"]` |
| `"x\t\ty"` | `["x", "\t\t", "y"]`         |

# 4. Implementation from Scratch

## Core Functions

```python
# Get pair frequencies
def get_pair_freq(tokens):
    pair_freq = defaultdict(int)
    for p0, p1 in zip(tokens, tokens[1:]):
        pair_freq[(p0, p1)] += 1
    return pair_freq

# Merge the most freqence pair using an assigned ID
def merge(tokens, most_freq_pair, assigned_id):
    res = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == most_freq_pair[0] and tokens[i+1] == most_freq_pair[1]:
            res.append(assigned_id)
            i += 2
        else:
            res.append(tokens[i])
            i += 1
    return res
```

## Training

```python
vocab_size = 276
num_merges = vocab_size - 256 # 20 merges

tokens = list(map(int, text.encode("utf-8")))

merges = {}
for i in range(num_merges):
    pair_freq = get_pair_freq(tokens)
    most_freq_pair = max(pair_freq, key=pair_freq.get) # Each dict key pass to stats.get() -> Get dict value -> Select the max dict value, return the dict key 
    assigned_id = 256 + i
    print(f"merging {most_freq_pair} into {assigned_id}")
    tokens = merge(tokens, most_freq_pair, assigned_id)
    merges[most_freq_pair] = assigned_id
```

## Encoding

```python
def encode(text):
    tokens = list(map(int, text.encode("utf-8")))
    while len(tokens) > 1: # Handle empty string edge case
        pair_freq = get_pair_freq(tokens)
        pair_with_min_id = min(pair_freq, key=lambda p: merges.get(p, float('inf'))) # Select the minimum ID. If a pair is not in merges, set the returned value to float('inf') to skip it.
        if pair not in merges: # All done
            break
        tokens = merge(tokens, pair_with_min_id, merges[pair_with_min_id])
    return tokens
```

## Decoding

```python
vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1] # Dict is ordered by default in Python > 3.7
    
def decode(ids):
    tokens = b"".join([vocab[idx] for idx in ids])
    text = tokens.decode("utf-8", errors="replace") # Avoid errors when token is invalid
    return text
```

---

# References

[1] https://en.wikipedia.org/wiki/Byte-pair_encoding

[2] Radford, Alec, et al. "Language models are unsupervised multitask learners." *OpenAI blog* 1.8 (2019): 9.