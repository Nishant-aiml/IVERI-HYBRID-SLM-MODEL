"""Fully instrumented verification script with timestamps and error tracebacks."""
import sys, time, traceback
sys.path.insert(0, '.')

import torch
from configs.base_config import get_base_config
from model.iveri_core import IVERIModel
from training.optimizer import get_optimizer
from training.mixed_precision import PrecisionHandler

def test_run(batch_size, use_checkpointing):
    print(f"\n=======================================================", flush=True)
    print(f"STARTING: batch_size={batch_size}, use_checkpointing={use_checkpointing}", flush=True)
    print(f"=======================================================", flush=True)
    
    t_start = time.perf_counter()
    
    try:
        # Step 1: Config and cache reset
        print("  [Step 1] Loading config and cleaning CUDA cache...", flush=True)
        t_step = time.perf_counter()
        cfg = get_base_config()
        cfg.hardware.mixed_precision = "fp16"
        cfg.hardware.gradient_checkpointing = use_checkpointing
        cfg.training.batch_size = batch_size
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"  [Step 1] Done in {time.perf_counter() - t_step:.2f}s", flush=True)
        
        # Step 2: Initialize model
        print("  [Step 2] Instantiating IVERIModel on CPU...", flush=True)
        t_step = time.perf_counter()
        model = IVERIModel(cfg)
        print(f"  [Step 2] Done in {time.perf_counter() - t_step:.2f}s", flush=True)
        
        # Step 3: Move model to GPU
        print("  [Step 3] Moving model to CUDA...", flush=True)
        t_step = time.perf_counter()
        model.to("cuda")
        torch.cuda.synchronize()
        print(f"  [Step 3] Done in {time.perf_counter() - t_step:.2f}s", flush=True)
        
        # Step 4: Setup optimizer and precision
        print("  [Step 4] Setting up optimizer and precision handler...", flush=True)
        t_step = time.perf_counter()
        optimizer = get_optimizer(model, cfg.training.learning_rate, cfg.training.weight_decay)
        ph = PrecisionHandler(precision="fp16", device="cuda")
        print(f"  [Step 4] Done in {time.perf_counter() - t_step:.2f}s", flush=True)
        
        # Step 5: Generate inputs
        print("  [Step 5] Creating dummy inputs...", flush=True)
        x = torch.randint(0, 256, (batch_size, 512), device="cuda")
        y = torch.randint(0, 256, (batch_size, 512), device="cuda")
        print("  [Step 5] Done", flush=True)
        
        # Step 6: Forward pass
        print("  [Step 6] Starting Forward pass...", flush=True)
        t_step = time.perf_counter()
        with ph.autocast_context():
            outputs = model(x, return_dict=True)
            logits = outputs["logits"]
            flat_logits = logits.view(-1, logits.size(-1))
            flat_targets = y.view(-1)
            loss = torch.nn.functional.cross_entropy(flat_logits, flat_targets)
        torch.cuda.synchronize()
        print(f"  [Step 6] Forward pass complete in {time.perf_counter() - t_step:.2f}s. Loss: {loss.item():.4f}", flush=True)
        
        # Step 7: Backward pass
        print("  [Step 7] Starting Backward pass...", flush=True)
        t_step = time.perf_counter()
        scaled_loss = ph.scale_loss(loss)
        scaled_loss.backward()
        torch.cuda.synchronize()
        print(f"  [Step 7] Backward pass complete in {time.perf_counter() - t_step:.2f}s", flush=True)
        
        # Step 8: Check gradients
        print("  [Step 8] Verifying gradients...", flush=True)
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.requires_grad and p.grad is not None]
        print(f"  [Step 8] Complete: {len(grad_norms)} parameters have gradients. Avg norm: {sum(grad_norms)/len(grad_norms):.6f}", flush=True)
        
        # Step 9: Optimizer step
        print("  [Step 9] Stepping optimizer...", flush=True)
        t_step = time.perf_counter()
        ph.step_optimizer(optimizer)
        torch.cuda.synchronize()
        print(f"  [Step 9] Optimizer step complete in {time.perf_counter() - t_step:.2f}s", flush=True)
        
        # Step 10: VRAM check
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"  [Step 10] Peak VRAM: {peak_vram:.1f} MB", flush=True)
        print(f"SUCCESS: batch_size={batch_size} finished in {time.perf_counter() - t_start:.2f}s total", flush=True)
        
    except Exception as e:
        print(f"\nFAILED: batch_size={batch_size} with exception:", flush=True)
        traceback.print_exc(file=sys.stdout)
        
    finally:
        # Clean up
        print("  Cleaning up CUDA memory...", flush=True)
        if 'model' in locals(): del model
        if 'optimizer' in locals(): del optimizer
        if 'x' in locals(): del x
        if 'y' in locals(): del y
        if 'outputs' in locals(): del outputs
        if 'logits' in locals(): del logits
        if 'loss' in locals(): del loss
        if 'ph' in locals(): del ph
        torch.cuda.empty_cache()

# Run BOTH checkpointed test cases
test_run(batch_size=4, use_checkpointing=True)
test_run(batch_size=8, use_checkpointing=True)
print("\n=== ALL TESTS COMPLETE ===", flush=True)
