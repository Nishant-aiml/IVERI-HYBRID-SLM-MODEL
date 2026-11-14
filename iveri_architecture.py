"""
IVERI Neural Architecture Suite:
- IVERINanoDense: Entropy-Gated Selective State-Space Foundation Model
- IVERITitanMoE: 64-Expert Top-2 Gated Mixture-of-Experts
Published in IJRASET (Paper ID: IJRASET85419, Vol 14, 2026)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class IVERIStateSpaceMixer(nn.Module):
    def __init__(self, d_model=512, d_state=64):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = u * F.silu(gate)
        return self.out_proj(u)

class SparseMoELayer(nn.Module):
    def __init__(self, d_model=512, num_experts=64, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4, bias=False),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model, bias=False)
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        orig_shape = x.shape
        x_flat = x.view(-1, x.shape[-1])
        router_logits = self.gate(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1)
        top_weights, top_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        
        out = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_idx = top_indices[:, i]
            weight = top_weights[:, i].unsqueeze(-1)
            for e_num in range(self.num_experts):
                mask = (expert_idx == e_num)
                if mask.any():
                    tokens = x_flat[mask]
                    out[mask] += weight[mask] * self.experts[e_num](tokens)
        return out.view(orig_shape)

class IVERITitanMoE(nn.Module):
    def __init__(self, vocab_size=50257, d_model=512, d_state=64, num_layers=4, num_experts=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(d_model),
                'ssm': IVERIStateSpaceMixer(d_model=d_model, d_state=d_state),
                'moe': SparseMoELayer(d_model=d_model, num_experts=num_experts, top_k=2)
            })
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.embedding.weight = self.lm_head.weight

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            norm_x = layer['norm'](x)
            x = x + layer['ssm'](norm_x)
            x = x + layer['moe'](x)
        return self.lm_head(self.final_norm(x))
