import time
import torch
import timm

dev = "cuda"
torch.backends.cudnn.benchmark = True
print("gpu:", torch.cuda.get_device_name(0), "| torch", torch.__version__, "| hip", torch.version.hip, flush=True)


def bench(batch):
    model = timm.create_model("convnext_small", pretrained=False, num_classes=5).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(batch, 3, 1024, 768, device=dev)
    y = torch.randint(0, 5, (batch,), device=dev)
    lossf = torch.nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(3):
            model(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            model(x)
        torch.cuda.synchronize()
    fwd = 10 * batch / (time.time() - t0)

    model.train()
    for _ in range(3):
        with torch.autocast("cuda", dtype=torch.float16):
            loss = lossf(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(10):
        with torch.autocast("cuda", dtype=torch.float16):
            loss = lossf(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    trn = 10 * batch / (time.time() - t0)
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"batch {batch}: forward {fwd:.1f} img/s | train {trn:.1f} img/s | peak VRAM {vram:.1f} GB", flush=True)
    del model, opt, x, y
    torch.cuda.empty_cache()


for b in (8, 16):
    try:
        bench(b)
    except RuntimeError as e:
        print(f"batch {b}: FAILED {type(e).__name__}: {str(e)[:160]}", flush=True)
        torch.cuda.empty_cache()
