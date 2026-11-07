# 21 TABLE REPRODUCTION
**STATUS: FINAL — Data incorporated from EXP-019 (Valid Baseline)**

| Model | Params | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Seed 46 | Mean Loss | Std | Latency (ms) | VRAM (MB) | Mean Grad Norm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| IVERI-Nano | 10.03M | 3.1087 | 3.0325 | 2.9944 | 3.1308 | 3.0374 | 3.0608 | 0.0569 | 3098.1 | 336 | 6.98-9.26 |
| Transformer| 9.72M | 4.8396 | 4.2607 | NaN | 7.2323 | 6.7163 | 5.7622 | 1.4349 | 36.4 | 206 | 34.5-55.6 |
*(Note: Transformer Seed 44 diverged. Mean/Std calculated over stable seeds.)*
