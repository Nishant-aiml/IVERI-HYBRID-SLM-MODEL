# Dataset Registry & Plugin Specification Report

This report summarizes the dataset specs registered via plugin YAML files.

## 1. Specification Registry (YAML Plugins)

The pipeline does not hardcode datasets. Instead, it discovers and parses YAML specification files under `data/dataset_specs/`:

1. **foundation.yaml**: Ingests Stage 1 pretraining data:
   - `tinystories` (roneneldan/TinyStories)
   - `fineweb_edu` (HuggingFaceFW/fineweb-edu, sample-10BT)
   - `dclm_baseline` (mlfoundations/dclm-baseline-1.0)
   - `wikipedia` (wikimedia/wikipedia)
   - `finemath` (HuggingFaceFW/finemath)
   - `the_stack_v2_python` (bigcode/the-stack-v2)

2. **instruction.yaml**: Ingests Stage 2 SFT data:
   - `magpie_pro` (magpie-align/Magpie-Pro-1M-v0.1)
   - `tulu3_sft` (allenai/tulu-3-sft-mixture)
   - `openhermes` (teknium/OpenHermes-2.5)
   - `wildchat` (allenai/WildChat)
   - `code_feedback` (m-a-p/Code-Feedback)
   - `numinamath` (AI-MO/NuminaMath-CoT)

3. **coding.yaml**: Ingests Stage 3A Coding Specialization data:
   - `the_stack_v2_deep` (bigcode/the-stack-v2)
   - `nemotron_competitive` (nvidia/Nemotron-SFT-Competitive-Programming-v2)
   - `leetcode` (yunhui/LeetCodeDataset)
   - `opencode_instruct` (nvidia/OpenCodeInstruct)
   - `codeforces` (open-r1/codeforces)

4. **engineering.yaml**: Ingests Stage 3B Proprietary Indian Engineering data (local source):
   - `university_papers` (Anna University/VTU/AKTU question papers)
   - `gate_questions` (GATE CS 1991-2025 questions + answers)
   - `placement_qa` (CS placement interview QA)
   - `subject_explanations` (Semester syllabus concept QA)

5. **preference.yaml**: Ingests Stage 4 alignment data:
   - `ultrafeedback`
   - `tulu3_pref`
   - `magpie_pro_dpo`

## 2. Ingestion Contracts

Each dataset has fields defining:
- `hf_id` and `hf_config`
- `priority` (S/A/B)
- `license`
- `format` (pretrain/sft/preference)
- `mixing_weight`
- `format_type` (alpaca/messages/conversations/dpo)
- `source` (huggingface/local)
