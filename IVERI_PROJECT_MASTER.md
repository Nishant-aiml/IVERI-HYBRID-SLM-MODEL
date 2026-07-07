# IVERI — Complete Project Reference
## For AI-Assisted Development | All Phases | All Technical Details

---

## WHAT THIS FILE IS

This is the single source of truth for building the IVERI project.
It contains everything needed — architecture, tech stack, file structure,
phase-by-phase build plan, component specs, and integration details.

Two projects. One vision.

```
Project 1: iveri-core     → the SLM (Small Language Model)
Project 2: nexus-rag      → the RAG library (pip installable, separate repo)
```

They are built separately and integrated later via:
```python
pip install nexus-rag
from nexus_rag import NexusRAG
model = IVERIModel(rag=NexusRAG())
```

---

## THE ONE-LINE DEFINITION

> IVERI CORE is the world's first byte-entropy-native hybrid SLM combining
> BLT + Titans + Mamba2 + MoR + MoE — solving token waste, context limits,
> memory bloat, and inference cost in a single unified architecture.

> NEXUS-RAG is the world's first CEDR-powered retrieval library — using
> byte-level generation entropy to trigger retrieval only at uncertainty
> spikes, injecting results into Titans memory (not context window).

---

## THE FOUR PROBLEMS BEING SOLVED

| Problem | Root Cause | Solution |
|---|---|---|
| Token waste | Fixed BPE vocab, uniform compute | BLT — byte-level entropy patches |
| Context window limit | Quadratic attention, hard cutoff | Titans — neural memory, no limit |
| KV cache bloat | Every token stored every layer | Mamba2 + MoR — linear + selective |
| Hallucinations | Miscalibrated confidence signal | CEDR — byte entropy drives retrieval |

---

---

# PROJECT 1: IVERI-CORE

---

## ARCHITECTURE OVERVIEW

```
INPUT
  Raw bytes
  → BLT Entropy Model (scores byte complexity)
  → Dynamic Patcher (groups bytes by entropy)
  → BLT Local Encoder (bytes → patch vectors)

MEMORY
  → Titans Neural Memory Module
    (deep MLP, updates weights online, no context limit)

BACKBONE (18 blocks)
  Each block:
  ┌────────────────────────────────┐
  │ MoR Router                     │  assigns recursion depth per token
  │ Mamba2 × 6   [linear O(n)]     │  92% of compute, no KV cache
  │ Flash Attention × 1            │  in-context recall, 8% of compute
  │ MoE FFN (4 experts, 2 active)  │  50% parameter efficiency
  │ RMSNorm + SwiGLU + RoPE        │
  └────────────────────────────────┘
  Repeat × 18

OUTPUT
  → BLT Local Decoder (patch vectors → raw bytes)
  → BLT-D: parallel byte generation
```

---

## PARAMETER BUDGET

| Component | Parameters | Purpose |
|---|---|---|
| BLT Entropy Model | 20M | Byte complexity scoring |
| BLT Local Encoder | 20M | Bytes to patch vectors |
| Titans Memory | 15M | Unlimited context memory |
| MoR Routers | 5M | Per-token depth assignment |
| Mamba2 Backbone | 140M | Primary processing (linear) |
| Flash Attention layers | 40M | In-context retrieval |
| MoE FFN layers | 40M | Specialized feed-forward |
| BLT Local Decoder | 20M | Patch to byte reconstruction |
| **TOTAL** | **300M** | Full hybrid architecture |

Scale versions use same architecture, more layers/width:
```
v0.1 Nano   10M    → verify architecture
v0.2 Core   50M    → verify full stack
v1.0 Mini  300M    → full proof of concept
v2.0 Base    1B    → first competitive product
v3.0 Pro     7B    → genuinely competitive
v4.0 Max    70B    → frontier domain champion
```

---

## COMPONENT SPECS

### BLT — Byte Latent Transformer
```
Origin:    Meta AI Research 2024
Repo:      github.com/facebookresearch/blt
Paper:     "Byte Latent Transformer: Patches Scale Better Than Tokens"

What it does:
  - Reads raw bytes instead of BPE tokens
  - Entropy model scores each byte's predictability
  - Low entropy bytes → grouped into large patches (easy content)
  - High entropy bytes → small patches (complex content)
  - BLT-D upgrade: generates multiple bytes per forward pass

Key numbers:
  - 50% fewer sequence positions than BPE
  - 50% inference compute reduction
  - No fixed vocabulary — any language, code, audio natively

Components to build:
  blt_entropy_model    → small MLP, scores byte entropy
  dynamic_patcher      → groups bytes into variable patches
  blt_local_encoder    → cross-attention, bytes → vectors
  blt_local_decoder    → patch vectors → bytes
```

### MAMBA2 — State Space Model
```
Origin:    Albert Gu, Tri Dao — CMU/Princeton 2024
Repo:      github.com/state-spaces/mamba
Paper:     "Transformers are SSMs: Generalized Models..."

What it does:
  - Replaces quadratic attention with linear time SSM
  - Maintains fixed-size hidden state (not growing KV cache)
  - Used for 92% of backbone layers (7:1 ratio with attention)

Key numbers:
  - O(n) complexity vs O(n²) for transformers
  - 3-5x throughput improvement
  - No KV cache growth with sequence length

Components to build:
  mamba2_block         → SSM layer with selective state spaces
  ssm_kernel           → core state transition
  Use mamba-ssm library as base, wrap in custom block
```

### MoR — Mixture of Recursions
```
Origin:    Google DeepMind, KAIST, Mila — NeurIPS 2025
Repo:      github.com/raymin0223/mixture_of_recursions
Paper:     "Mixture-of-Recursions: Learning Dynamic Recursive Depths..."

What it does:
  - Assigns each token a recursion depth via lightweight router
  - Simple tokens (the, a, and) → depth 1, exit immediately
  - Complex tokens → depth 4-8, loop through layers
  - KV cache only stores tokens still in active recursion
  - Parameter sharing across recursion levels

Key numbers:
  - 60-70% KV cache reduction on typical text
  - 20-30% additional throughput boost
  - Surpasses vanilla transformer with fewer parameters

Components to build:
  mor_router           → tiny MLP, outputs depth per token
  recursion_engine     → loop tokens through shared weights
  selective_kv_cache   → only cache active-recursion tokens
```

### TITANS — Neural Memory
```
Origin:    Google DeepMind — NeurIPS 2025
Repo:      github.com/google-deepmind/titans
Paper:     "Titans: Learning to Memorize at Test Time"

What it does:
  - Memory = deep MLP (not fixed lookup table)
  - MLP weights update ONLINE as each token arrives
  - Task model generates learning rate dynamically per token
  - No fixed context window — information encoded in weights
  - Also used by NEXUS-RAG to inject retrieved facts

Key numbers:
  - No hard context limit
  - Memory persists across conversations when saved
  - ~15M parameters for memory module

Components to build:
  titans_memory        → deep MLP memory module
  online_updater       → weight update mechanism per token
  lr_generator         → task model generates learning rate
  memory_reader        → retrieve relevant memory for generation
```

### MoE — Mixture of Experts
```
Origin:    Shazeer et al. Google Brain 2017, popularized 2024-2025
Used in:   DeepSeek-V3, Mixtral

What it does:
  - 4 expert FFN networks instead of 1
  - Router selects top-2 experts per token
  - Only 50% of FFN parameters activate per forward pass
  - Experts can specialize: code, language, math, general

Key numbers:
  - DeepSeek-V3: 671B total, 37B active (MoE)
  - IVERI: 4 experts, 2 active = 50% FFN efficiency

Components to build:
  moe_router           → learns which expert per token
  expert_ffn           → 4 SwiGLU FFN blocks
  expert_selector      → top-2 selection with load balancing
```

---

## TECH STACK — IVERI-CORE

### Required Libraries
```
torch>=2.3.0          core framework
mamba-ssm             official Mamba2 SSM blocks
flash-attn>=2.5       memory-efficient attention
einops                tensor manipulation
rotary-emb            RoPE positional embeddings
bitsandbytes          quantization (4-bit/8-bit)
triton                GPU kernel optimization
accelerate            multi-GPU training
transformers          utilities and reference implementations
datasets              HuggingFace dataset loading
tokenizers            fast tokenizer utilities
wandb                 training monitoring
lm-eval-harness       benchmarking and evaluation
numpy
tqdm
```

### Installation
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install mamba-ssm --no-build-isolation
pip install flash-attn --no-build-isolation
pip install einops rotary-emb bitsandbytes accelerate
pip install transformers datasets tokenizers
pip install wandb lm-eval
pip install numpy tqdm
```

### Reference Repositories
```
facebookresearch/blt              BLT architecture
state-spaces/mamba                Mamba2 SSM
raymin0223/mixture_of_recursions  MoR
google-deepmind/titans            Titans memory
karpathy/nanoGPT                  training scaffold reference
EleutherAI/lm-evaluation-harness  evaluation
```

### Training Datasets
```
Phase 1 (pre-training):
  TinyStories        2.1M stories  simple text, learn basics
  OpenWebText        38GB          diverse web text

Phase 2 (instruction tuning):
  Stanford Alpaca    52K pairs     instruction following
  Code Alpaca        20K pairs     code instructions
  FLAN               large         multi-task tuning

Phase 3 (domain):
  Custom 5G corpus   build this    domain specialization
```

---

## FOLDER STRUCTURE — IVERI-CORE

```
iveri-core/
├── configs/
│   ├── base_config.py          all hyperparams in one place
│   ├── nano_10m.py             10M model config
│   ├── small_50m.py            50M model config
│   └── mini_300m.py            300M model config
│
├── data/
│   ├── dataloader.py           TinyStories + OpenWebText loader
│   ├── preprocessing.py        byte-level preprocessing for BLT
│   └── dataset_utils.py        batching, shuffling utilities
│
├── model/
│   ├── __init__.py
│   ├── blt/
│   │   ├── entropy_model.py    byte entropy scorer
│   │   ├── patcher.py          dynamic byte patcher
│   │   ├── encoder.py          BLT local encoder
│   │   └── decoder.py          BLT local decoder
│   ├── mamba2/
│   │   ├── ssm_block.py        Mamba2 SSM block
│   │   └── ssm_kernel.py       state transition kernel
│   ├── mor/
│   │   ├── router.py           token depth router
│   │   ├── recursion.py        recursive depth engine
│   │   └── kv_cache.py         selective KV cache
│   ├── titans/
│   │   ├── memory.py           neural memory MLP
│   │   ├── updater.py          online weight updater
│   │   └── lr_gen.py           dynamic learning rate generator
│   ├── moe/
│   │   ├── router.py           expert selector
│   │   └── experts.py          4 SwiGLU expert FFNs
│   ├── attention.py            flash attention wrapper
│   ├── norms.py                RMSNorm implementation
│   ├── backbone.py             assembles one full block
│   └── iveri_core.py           assembles full model
│
├── training/
│   ├── trainer.py              main training loop
│   ├── optimizer.py            AdamW + cosine LR schedule
│   ├── checkpointing.py        save/load checkpoints
│   └── mixed_precision.py      FP16/BF16 handling
│
├── evaluation/
│   ├── evaluator.py            runs all benchmarks
│   ├── perplexity.py           perplexity measurement
│   ├── throughput.py           tokens/sec measurement
│   └── memory_tracker.py      VRAM usage tracking
│
├── baselines/
│   ├── tiny_transformer.py     baseline A for comparison
│   └── tiny_mamba.py           baseline B for comparison
│
├── research_log/
│   └── RESEARCH_LOG.md         every experiment recorded here
│
├── scripts/
│   ├── sanity_check.py         forward + backward pass test
│   ├── train_nano.py           train 10M model
│   ├── train_small.py          train 50M model
│   └── benchmark.py            run full benchmark suite
│
├── requirements.txt
├── README.md
└── train.py                    main entry point
```

---

## BASE CONFIG

```python
# configs/base_config.py

config = {
    # Model
    "vocab_size": None,          # None = BLT (no fixed vocab)
    "hidden_dim": 512,           # start small, scale up
    "num_layers": 18,            # backbone blocks
    "num_heads": 8,              # attention heads (flash attn layers)
    "mamba_ratio": 6,            # mamba blocks per attention block
    "num_experts": 4,            # MoE experts
    "num_active_experts": 2,     # active per token
    "max_recursion_depth": 8,    # MoR max depth
    "titans_memory_dim": 256,    # Titans memory MLP width

    # BLT
    "patch_size_min": 1,         # min bytes per patch
    "patch_size_max": 8,         # max bytes per patch
    "entropy_threshold": 0.5,    # patch boundary decision

    # Training
    "batch_size": 32,
    "gradient_accumulation": 4,  # effective batch = 128
    "learning_rate": 3e-4,
    "min_lr": 3e-5,
    "warmup_steps": 1000,
    "max_steps": 50000,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "seq_len": 512,              # start short, increase

    # Hardware
    "mixed_precision": "fp16",
    "gradient_checkpointing": True,
    "device": "cuda",

    # Logging
    "log_every": 10,
    "eval_every": 500,
    "save_every": 1000,
    "wandb_project": "iveri-core",
}

# Scale configs override base
nano_config = {**config, "hidden_dim": 256, "num_layers": 6}   # ~10M
small_config = {**config, "hidden_dim": 512, "num_layers": 12}  # ~50M
mini_config  = {**config, "hidden_dim": 768, "num_layers": 18}  # ~300M
```

---

## PHASE-BY-PHASE BUILD PLAN — IVERI-CORE

### PHASE 0 — Project Setup

**Goal:** Clean repo, verified dependencies, folder structure ready.

**Steps:**
1. Create folder structure exactly as above
2. Create `requirements.txt`
3. Install all dependencies
4. Verify with:
```python
import torch
import mamba_ssm
import flash_attn
print(torch.cuda.is_available())    # must be True
print(torch.cuda.get_device_name()) # RTX 3050
```

**Exit criteria:**
- [ ] All imports work
- [ ] CUDA available
- [ ] Folder structure created
- [ ] `base_config.py` written

---

### PHASE 1 — Build All Components

**Goal:** Every module written and individually testable. No training yet.

**Build order (simpler first):**

#### Step 1.1 — RMSNorm + RoPE + SwiGLU (foundations)
```
File: model/norms.py
Test: norm(torch.randn(2, 10, 512)).shape == (2, 10, 512)
```

#### Step 1.2 — MoE FFN
```
File: model/moe/experts.py + router.py
Test: only 2 of 4 experts fire per token
      output shape correct
      load balancing loss computable
```

#### Step 1.3 — Mamba2 Block
```
File: model/mamba2/ssm_block.py
Wrap: mamba-ssm library blocks
Test: random input (batch, seq, dim) → output same shape
      no crash, no NaN
```

#### Step 1.4 — Flash Attention Block
```
File: model/attention.py
Test: attention(q, k, v) → correct shape
      causal mask works
```

#### Step 1.5 — MoR Router
```
File: model/mor/router.py + recursion.py + kv_cache.py
Test: different tokens get different recursion depths
      verify depth distribution is not collapsed
      verify KV cache only stores active tokens
```

#### Step 1.6 — BLT Components
```
Files: model/blt/entropy_model.py
       model/blt/patcher.py
       model/blt/encoder.py
       model/blt/decoder.py

Test entropy model:
  input: raw bytes [72, 65, 6C, 6C, 6F]
  output: entropy score per byte

Test patcher:
  low entropy bytes → grouped into large patch
  high entropy bytes → small patch

Test encoder:
  patches → latent vectors, correct shape

Test decoder:
  latent vectors → bytes
  "Hello world" → encode → decode → "Hello world"
```

#### Step 1.7 — Titans Memory
```
Files: model/titans/memory.py
       model/titans/updater.py
       model/titans/lr_gen.py

Test:
  memory = TitansMemory(dim=256)
  seq = torch.randn(1, 100, 512)
  out = memory(seq)             # memory weights updated
  assert out.shape == seq.shape
  # run longer sequence, verify no OOM, no crash
```

#### Step 1.8 — Assemble Backbone Block
```
File: model/backbone.py
Assembles: MoR → Mamba2×6 → Attention×1 → MoE FFN → RMSNorm
Test: one_block(x).shape == x.shape
```

#### Step 1.9 — Assemble Full Model
```
File: model/iveri_core.py
Assembles: BLT Encoder → Titans → Backbone×18 → BLT Decoder
```

#### PHASE 1 EXIT GATE — Must all pass before Phase 2

```python
# scripts/sanity_check.py

model = IVERIModel(config=nano_config)

# Test 1: forward pass
dummy = torch.randint(0, 256, (2, 512))   # batch=2, seq=512 bytes
out = model(dummy)
assert out.shape == (2, 512, 256)          # output logits
print("✅ Forward pass")

# Test 2: backward pass
loss = out.mean()
loss.backward()
print("✅ Backward pass")

# Test 3: no NaN
assert not torch.isnan(out).any()
print("✅ No NaN in outputs")

# Test 4: checkpoint
torch.save(model.state_dict(), "test_checkpoint.pt")
model.load_state_dict(torch.load("test_checkpoint.pt"))
print("✅ Checkpoint save/load")

# Test 5: inference
model.eval()
with torch.no_grad():
    tokens = model.generate(dummy[:1], max_new_tokens=20)
print("✅ Inference generates tokens")

print("\n🎉 All Phase 1 gates passed. Ready for Phase 2.")
```

**If any test fails:** debug that component before moving on.
Many architectures fail at this gate. Passing it is a real milestone.

---

### PHASE 2 — Tiny Prototype (5M–20M)

**Goal:** Prove it learns. Loss must go down. Catch instability early.

#### Step 2.1 — Data Pipeline
```
File: data/dataloader.py
Dataset: TinyStories
  from datasets import load_dataset
  ds = load_dataset("roneneldan/TinyStories")

Output: byte-level batches (not tokenized)
        batch shape: (batch_size, seq_len)  dtype=uint8

Test:
  for batch in dataloader:
      assert batch.shape == (32, 512)
      assert batch.dtype == torch.uint8
      break
  print("✅ Dataloader works")
```

#### Step 2.2 — Training Loop
```
File: training/trainer.py

Core loop:
  for step, batch in enumerate(dataloader):
      batch = batch.to(device)
      logits = model(batch)
      loss = cross_entropy(logits, batch)    # next-byte prediction
      loss.backward()
      clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      scheduler.step()
      optimizer.zero_grad()

      if step % 10 == 0:
          log(step, loss.item())
```

#### Step 2.3 — Train 5M Model, 100 Steps

Watch for these failure modes:
```
Loss stuck at same value   → learning rate issue or gradient not flowing
Loss = NaN immediately     → gradient explosion, reduce LR or check Titans
Loss = NaN after 50 steps  → Titans memory instability, check updater
MoR all same depth         → router collapsed, check router init
Loss decreasing then stuck → good! just needs more steps
```

**Pass criteria for 5M:**
- [ ] Loss decreases over 100 steps
- [ ] No NaN
- [ ] Titans stable (no exploding memory weights)
- [ ] MoR routing diverse (not all tokens same depth)
- [ ] RTX 3050 handles it without OOM

#### Step 2.4 — Train 20M Model, 1000 Steps
```
Config: small_config with fewer layers
Target: perplexity < 50 on TinyStories validation
Time estimate: ~2 hours on RTX 3050
```

**Research log entry required after each run.**

---

### PHASE 3 — First Benchmark

**Goal:** Evidence that IVERI architecture provides value over baselines.

#### Build Baselines
```
File: baselines/tiny_transformer.py
  Standard GPT-style transformer
  Same parameter count as IVERI being compared
  Same training data, same steps

File: baselines/tiny_mamba.py
  Pure Mamba-only model
  Same parameter count
  Same training data, same steps
```

#### Benchmark Suite
```
File: evaluation/evaluator.py

Metrics to measure (all three models):

1. Perplexity
   measure on TinyStories validation set
   lower = better

2. Loss curve
   plot loss vs steps
   does IVERI converge faster?

3. Throughput
   tokens_per_second = seq_len * batch_size / time_per_step
   higher = better

4. Peak VRAM
   torch.cuda.max_memory_allocated()
   lower = better

5. KV cache size
   measure actual KV cache growth vs sequence length
   IVERI should grow slower (MoR + Mamba2)

File: scripts/benchmark.py
  runs all three models on all five metrics
  outputs comparison table
  saves to research_log/
```

**Pass criteria:**
- [ ] IVERI perplexity ≤ transformer baseline
- [ ] IVERI throughput > transformer baseline
- [ ] IVERI VRAM < transformer baseline
- [ ] KV cache grows slower in IVERI
- [ ] Results documented with actual numbers

---

### PHASE 4 — Scale Incrementally

**Goal:** Validate architecture scales cleanly. Never scale blindly.

**Rule:** Each step must pass Phase 3 benchmarks before next scale.

```
20M
  Train on OpenWebText subset (10GB)
  Benchmark against transformer 20M
  Document in research log
  Pass? → proceed

50M
  Train on OpenWebText subset (20GB)
  Move to Kaggle (free T4 x2 = 30GB VRAM)
  Benchmark against transformer 50M
  Document in research log
  Pass? → proceed

123M
  Train on OpenWebText full (38GB)
  Kaggle + Colab Pro
  Benchmark against transformer 123M
  Document in research log
  THIS IS THE MILESTONE
```

**Hardware progression:**
```
10M-20M   → RTX 3050 (local)
50M       → RTX 3050 with FP16 + gradient checkpointing
            OR Kaggle free tier
123M      → Kaggle T4 x2 (free, 30GB VRAM)
            OR Colab Pro A100 (~$10/month)
300M+     → Vast.ai RTX 3090 (~$0.30/hr)
```

**VRAM optimization tricks for RTX 3050:**
```python
# Gradient checkpointing
model = model.gradient_checkpointing_enable()

# FP16 training
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    loss = model(batch)

# Gradient accumulation (simulate batch_size=128 with 4GB)
accumulation_steps = 8
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i+1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

### PHASE 5 — Instruction Tuning + Chat

**Goal:** Turn the language model into a usable assistant.

```
Dataset: Stanford Alpaca (52K pairs)
Format:
  ### Instruction:
  {instruction}
  ### Response:
  {response}

Fine-tune 123M base model for 3 epochs
Measure:
  Response quality (manual eval)
  Instruction following rate

Add:
  Conversation history handling
  System prompt support
  Inference script for interactive chat
```

---

### PHASE 6 — Research Stage

**Goal:** Turn experiments into publications and patents.

By this phase your research log answers:

| Research Question | Evidence Source |
|---|---|
| Does BLT reduce compute? | Phase 3/4 throughput measurements |
| Does Titans extend context? | Long-sequence VRAM measurements |
| Does MoR reduce KV cache? | Phase 3 KV cache measurements |
| Does architecture scale? | Phase 4 scaling results |
| Does it beat baselines? | Phase 3/4 benchmark tables |

**Outputs:**
- Paper 1: Architecture paper → NeurIPS 2026 / ICLR 2027
- Paper 2: Efficiency paper → MLSys 2027
- Paper 3: Domain paper → IEEE / INFOCOM
- Patents 1-4: file after first working prototype

---

---

# PROJECT 2: NEXUS-RAG

---

## WHAT IT IS

Standalone pip-installable RAG library.
Works with ANY LLM (LLaMA, Mistral, GPT API, IVERI).
CEDR full mode activates only when used with IVERI CORE (requires BLT entropy).

```
pip install nexus-rag

# Standalone use (any LLM)
from nexus_rag import NexusRAG
rag = NexusRAG()
answer = rag.query("your question", llm=any_llm)

# Full mode (with IVERI)
from nexus_rag import NexusRAG
from model.iveri_core import IVERIModel
rag = NexusRAG(entropy_source="blt")   # CEDR full mode
model = IVERIModel(rag=rag)
```

---

## ARCHITECTURE — NEXUS-RAG

```
STAGE 0: Query Triage (5M classifier)
  Route A → skip retrieval (40% of queries)
  Route B → light retrieval (40%)
  Route C → full NEXUS pipeline (20%)

STAGE 1: 4-Index Parallel Corpus
  Index A: Dense Vector (FAISS) — semantic meaning
  Index B: BM25 Sparse — exact keywords, function names
  Index C: Knowledge Graph (networkx/neo4j) — relationships, AST
  Index D: Episodic Memory — per-user history, past errors

STAGE 2: Hyper Query Engine
  HyDE: generate hypothetical answer, embed that
  Decompose: complex query → sub-queries
  Route: code→AST, factual→BM25+vector, reasoning→graph

STAGE 3: NEXUS Reranker
  Cross-encoder: top-50 → top-5
  Contradiction detector: check chunks against each other
  Freshness scorer: recent docs weighted higher
  Confidence calibrator: low confidence → web fallback

STAGE 4: CEDR — THE WORLD FIRST [PATENT]
  BLT entropy monitor watches generation byte by byte
  Low entropy → model confident → zero retrieval (free)
  Entropy spike → model uncertain → micro-retrieval triggered
  Retrieved fact → Titans memory (NOT context window)
  Generation continues, grounded at exact uncertainty point

STAGE 5: Code Intelligence Layer
  AST-native indexing via tree-sitter
  Function call graphs
  Import dependency graphs
  Error pattern matching
  Test-aware generation

STAGE 6: Hallucination Shield
  Every factual claim cross-referenced
  Confidence score per claim
  Citations attached by default
  Low confidence → flagged or re-retrieved
```

---

## TECH STACK — NEXUS-RAG

```
torch                 neural components (reranker, fusion net)
faiss-cpu             dense vector index
rank-bm25             BM25 sparse retrieval
networkx              knowledge graph structure
tree-sitter           AST parsing (40+ languages)
sentence-transformers  embedding model
chromadb              vector database backend
httpx                 async web search fallback
pydantic              data validation
pytest                testing
```

---

## FOLDER STRUCTURE — NEXUS-RAG

```
nexus-rag/
├── nexus_rag/
│   ├── __init__.py
│   ├── triage/
│   │   └── classifier.py       5M query classifier
│   ├── indexes/
│   │   ├── dense.py            FAISS vector index
│   │   ├── sparse.py           BM25 index
│   │   ├── graph.py            knowledge graph
│   │   └── episodic.py         per-user memory store
│   ├── query/
│   │   ├── hyde.py             hypothetical document embedding
│   │   ├── decomposer.py       query decomposition
│   │   └── router.py           intent-aware routing
│   ├── retrieval/
│   │   ├── retriever.py        parallel multi-index retrieval
│   │   └── fusion.py           learned score combiner
│   ├── reranking/
│   │   ├── reranker.py         cross-encoder reranker
│   │   ├── contradiction.py    contradiction detector
│   │   ├── freshness.py        freshness scorer
│   │   └── confidence.py       confidence calibrator
│   ├── cedr/
│   │   ├── entropy_monitor.py  BLT entropy watcher
│   │   ├── micro_retrieval.py  targeted retrieval at spike
│   │   └── memory_injector.py  inject to Titans (not context)
│   ├── code/
│   │   ├── ast_indexer.py      AST-native code indexing
│   │   ├── call_graph.py       function call graph builder
│   │   └── error_matcher.py    stack trace pattern matching
│   ├── shield/
│   │   ├── fact_checker.py     claim cross-reference
│   │   ├── citation.py         automatic citation attachment
│   │   └── confidence.py       output confidence scoring
│   └── nexus_core.py           assembles full pipeline
├── tests/
├── research_log/
│   └── RESEARCH_LOG.md
├── requirements.txt
├── setup.py
└── README.md
```

---

## PHASE-BY-PHASE BUILD PLAN — NEXUS-RAG

### PHASE N1 — Standalone Core (Month 3-5, parallel to IVERI)

```
N1.1 — Dense vector index (FAISS)
        Test: embed text, retrieve top-5 similar

N1.2 — BM25 sparse index
        Test: keyword search works, exact matches found

N1.3 — Basic retriever (dense + sparse fusion)
        Test: better than either alone on test queries

N1.4 — Query triage classifier
        Test: 40% of simple queries skip retrieval

N1.5 — HyDE query expansion
        Test: hypothetical embedding improves retrieval

N1.6 — Query decomposer
        Test: complex query breaks into sub-queries

N1.7 — Cross-encoder reranker
        Test: top-50 → top-5, quality improves
```

**Exit gate N1:** Works as basic RAG with any LLM. Better than naive RAG.

---

### PHASE N2 — Advanced Features (Month 5-6)

```
N2.1 — Knowledge graph index
        tree-sitter AST parsing
        function call graph builder
        Test: code query retrieves function structure not text

N2.2 — Contradiction detector
        Test: conflicting chunks flagged before generation

N2.3 — Freshness scorer
        Test: recent docs ranked higher

N2.4 — Episodic memory store
        Test: past conversation context retrievable

N2.5 — Hallucination shield
        Test: factual claims cross-referenced, citations attached
```

---

### PHASE N3 — CEDR Integration (Month 7, after IVERI v1 stable)

```
N3.1 — Entropy monitor
        Watches BLT entropy signal during IVERI generation
        Detects spikes above threshold

N3.2 — Micro-retrieval
        At entropy spike → targeted query → 1-2 chunks

N3.3 — Titans memory injection
        Retrieved fact → Titans memory weights (not context)
        Verify: context window size unchanged after retrieval

N3.4 — Token fallback mode
        For use with non-IVERI LLMs
        Uses token probability as weaker entropy signal
        Still better than naive RAG
```

**Exit gate N3:** CEDR full mode working with IVERI.
Measure hallucination rate before vs after.

---

### PHASE N4 — Publish + Integrate

```
pip install nexus-rag published to PyPI
Integration test with IVERI CORE
Benchmark: all RAG systems vs NEXUS-RAG
Write Paper 1 (CEDR paper)
File Patent 1 (CEDR mechanism)
```

---

---

# RESEARCH LOG TEMPLATE

Use this for EVERY experiment. No exceptions.

```markdown
## Experiment [NUMBER]

**Date:** YYYY-MM-DD
**Phase:** Phase X, Step Y
**Project:** iveri-core / nexus-rag

### Setup
- Model size: XM parameters
- Architecture changes from last run:
- Dataset:
- Sequence length:
- Batch size (effective):
- Learning rate:
- Steps trained:
- Hardware:

### Results
- Starting loss:
- Final loss:
- Perplexity:
- Throughput (tokens/sec):
- Peak VRAM:
- KV cache size:

### Observations
- What worked:
- What failed:
- Unexpected behaviour:

### Comparison
- vs previous experiment:
- vs baseline transformer:

### Next Experiment
- What to try next:
- Hypothesis:
```

---

---

# BENCHMARKING REFERENCE

## Key Metrics

| Metric | How to Measure | Target |
|---|---|---|
| Perplexity | exp(avg cross-entropy loss on val set) | Lower than transformer baseline |
| Throughput | tokens/sec during training | Higher than transformer |
| VRAM | torch.cuda.max_memory_allocated() | Lower than transformer |
| KV cache | measure growth vs seq length | Slower growth (MoR + Mamba2) |
| Loss curve | plot loss vs steps | Converges as fast or faster |
| Hallucination | count wrong facts in 100 answers | <5% with NEXUS, <20% without |

## Baseline Models to Compare Against
```
Baseline A: tiny standard GPT transformer (same params)
Baseline B: pure Mamba model (same params)
Baseline C: LLaMA-style model at same scale (from HuggingFace)
```

## Evaluation Datasets
```
TinyStories validation  language model quality
OpenWebText validation  general text quality
HumanEval              coding quality (after instruction tuning)
MMLU (subset)          knowledge and reasoning
Custom 5G dataset      domain quality (build this)
```

---

---

# PATENTS REFERENCE

File provisional patent after first working prototype.
Cost in India: ~₹15,000 for provisional. Gives 12 months protection.

| Patent | What to Claim | Strength |
|---|---|---|
| Patent 1 | Joint training of BLT patches with Titans online memory updates in hybrid SSM-Attention backbone | VERY STRONG |
| Patent 2 | Adaptive KV cache based on joint MoR recursion depth + BLT entropy scores | STRONG |
| Patent 3 | MoE expert routing conditioned on BLT byte-patch entropy | STRONG |
| Patent 4 | Three-phase curriculum training for BLT + Memory + SSM-Attention models | MEDIUM |
| Patent 5 (NEXUS) | CEDR: continuous retrieval triggered by byte-level patch entropy during generation | VERY STRONG |
| Patent 6 (NEXUS) | RAG result injection into Titans neural memory instead of context window | STRONG |

---

---

# PAPERS REFERENCE

| Paper | Title | Target Venue | Timeline |
|---|---|---|---|
| Paper 1 | IVERI: Unified Hybrid Architecture with BLT, Titans, Mamba2, MoR | NeurIPS 2026 / ICLR 2027 | After Phase 4 |
| Paper 2 | NEXUS-RAG: Continuous Entropy-Driven Retrieval via Byte-Level Uncertainty | NeurIPS 2026 / ICLR 2027 | After CEDR works |
| Paper 3 | IVERI-Telecom: Byte-Level Neural Memory for 5G Edge Security | IEEE / INFOCOM | After domain tuning |
| Paper 4 | Inference Cost Analysis: BLT-Titans-MoR vs Pure Transformer | MLSys 2027 | After Phase 4 |

---

---

# HARDWARE REFERENCE

| Task | Hardware | Cost | When |
|---|---|---|---|
| Dev + debugging | RTX 3050 4GB | $0 | Always |
| 10M training | RTX 3050 | $0 | Phase 2 |
| 50M training | Kaggle T4 x2 (30GB) | Free | Phase 4 |
| 123M training | Colab Pro A100 (40GB) | ~$10-20/month | Phase 4 |
| 300M training | Vast.ai RTX 3090 | ~$0.30/hr | Phase 5+ |
| 1B+ training | Vast.ai 4x A100 | ~$2-4/hr | Year 2 |

**RTX 3050 VRAM tricks:**
```python
# Always use these on RTX 3050
torch.backends.cuda.matmul.allow_tf32 = True
model.gradient_checkpointing_enable()
# Use fp16 autocast
# Use gradient accumulation (steps=8 minimum)
# Keep seq_len ≤ 512 for initial runs
```

---

---

# QUICK REFERENCE — WHAT GETS BUILT WHEN

```
MONTH 1   Phase 0 setup + Phase 1 architecture (IVERI-CORE)
MONTH 2   Phase 2 tiny prototype 5M-20M
MONTH 3   Phase 3 first benchmark + start NEXUS-RAG N1
MONTH 4   Phase 4 scale to 50M + NEXUS-RAG N1 complete
MONTH 5   Phase 4 scale to 123M + NEXUS-RAG N2
MONTH 6   Phase 5 instruction tuning + chat interface
MONTH 7   NEXUS-RAG N3 CEDR integration
MONTH 8   Phase 6 papers + patents + publish
```

---

*IVERI Project Master Reference — July 2026*
*Status: Phase 6.3 engineering complete; Phase 6.3.2 scientific integrity restoration complete (OBJ1–8).*
*Production model class: `IVERIModel` (`model/iveri_core.py`). Architecture revision `0.2.0-byte-vocab`.*
*Two repos: iveri-core + nexus-rag*
*No prior art confirmed for full combination as of June 22, 2026*
