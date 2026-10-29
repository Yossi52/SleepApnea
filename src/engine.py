"""Training loop, inference and metrics (same numerics as the paper's training scripts)."""
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch import nn, optim
from tqdm import tqdm

# model_type -> (lr, OneCycle max_lr, grad-clip max_norm, OneCycle final_div_factor, batch size)
HYPERPARAMS = {
    'single': (5e-4, 1e-3, 1.0, 1e4, 16),
    'early': (5e-4, 1e-3, 1.0, 1e4, 32),
    'late': (5e-6, 1e-4, 0.5, 1e5, 32),
    'hybrid': (5e-6, 1e-4, 0.5, 1e5, 32),
    'mb_late': (5e-6, 1e-4, 0.5, 1e5, 32),
    'mb_hybrid': (5e-6, 1e-4, 0.5, 1e5, 32),
}


def prepare(x, device, edit=None):
    x = x.to(device, dtype=torch.float32)
    if edit is not None:
        x = edit(x)
    # Kept from the original pipeline: z-scored batches are divided by 255 whenever max > 1
    # (nearly always). Trained checkpoints expect it; dropping it costs ~30%p accuracy.
    return x / 255.0 if x.max() > 1.0 else x


def logits_of(model, x):
    """fp16 autocast forward, recomputed in fp32 if fp16 overflowed to NaN/Inf."""
    with torch.autocast(x.device.type, enabled=x.device.type == 'cuda'):
        out = model(x)
    if not torch.isfinite(out).all():
        out = model(x)
    return torch.clamp(out.float(), -100, 100)


def fit(model, train_loader, valid_loader, model_type, epochs, patience, class_weights, ckpt_path, device, log=print):
    """Train with early stopping on validation loss; the best epoch is saved to ckpt_path."""
    lr, max_lr, max_norm, final_div, _ = HYPERPARAMS[model_type]
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5, eps=1e-6)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=len(train_loader),
        pct_start=0.05, div_factor=25.0, final_div_factor=final_div, anneal_strategy='cos')
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda')
    history = {k: [] for k in ('train_loss', 'val_loss', 'val_acc', 'val_auc')}
    best, since_best = float('inf'), 0

    for epoch in range(epochs):
        model.train()
        losses = []
        for x, y in tqdm(train_loader, desc=f'epoch {epoch + 1}', leave=False):
            x, y = prepare(x, device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == 'cuda'):
                loss = criterion(torch.clamp(model(x), -100, 100), y)
            if not torch.isfinite(loss):
                log(f'non-finite loss at epoch {epoch + 1}, batch skipped')
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())

        model.eval()
        val_losses, y_val, prob_val = [], [], []
        with torch.no_grad():
            for x, y in valid_loader:
                logits = logits_of(model, prepare(x, device))
                val_losses.append(criterion(logits, y.to(device)).item())
                prob_val.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
                y_val.append(y.numpy())
        val_loss = float(np.mean(val_losses))
        val = metrics(np.concatenate(y_val), np.concatenate(prob_val))
        for k, v in (('train_loss', float(np.mean(losses))), ('val_loss', val_loss),
                     ('val_acc', val['accuracy']), ('val_auc', val['auc'])):
            history[k].append(v)
        log(f'epoch {epoch + 1}: train_loss={history["train_loss"][-1]:.4f} val_loss={val_loss:.4f} '
            f'val_acc={val["accuracy"]:.4f} val_auc={val["auc"]:.4f}')

        if val_loss < best:
            best, since_best = val_loss, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            since_best += 1
            if since_best >= patience:
                break
    history['best_epoch'] = int(np.argmin(history['val_loss']))
    return history


@torch.no_grad()
def predict(model, loader, device, edit=None):
    """Return (y_true, p(class 1), y_pred). `edit(x)` may modify each normalized batch (branch ablation)."""
    model.eval()
    ys, probs = [], []
    for x, y in loader:
        x = prepare(x, device, edit)
        probs.append(torch.softmax(logits_of(model, x), dim=1)[:, 1].cpu().numpy())
        ys.append(y.numpy())
    prob = np.clip(np.concatenate(probs), 1e-7, 1 - 1e-7)
    return np.concatenate(ys), prob, (prob >= 0.5).astype(int)


def metrics(y, prob, pred=None):
    pred = (prob >= 0.5).astype(int) if pred is None else pred
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'accuracy': float((pred == y).mean()), 'auc': float(roc_auc_score(y, prob)),
            'sensitivity': float(tp / max(tp + fn, 1)), 'specificity': float(tn / max(tn + fp, 1)),
            'f1': float(f1_score(y, pred, zero_division=0))}
