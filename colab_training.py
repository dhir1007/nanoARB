# =============================================================================
# NanoARB GPU Training for Google Colab
# =============================================================================
# 
# INSTRUCTIONS:
# 1. Go to https://colab.research.google.com
# 2. Create new notebook
# 3. Go to Runtime → Change runtime type → T4 GPU
# 4. Copy this ENTIRE file into a single cell and run
# 5. Download checkpoints/model.onnx when done
#
# =============================================================================

# Install dependencies
import subprocess
subprocess.run(["pip", "install", "-q", "einops", "onnx", "onnxruntime", "tqdm"])

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from einops import rearrange
from tqdm import tqdm
import numpy as np
import time
import os

print("=" * 60)
print("🚀 NanoARB GPU Training")
print("=" * 60)

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n✅ Using device: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  No GPU found! Go to Runtime → Change runtime type → T4 GPU")

# =============================================================================
# Configuration
# =============================================================================
class Config:
    # Data
    num_samples = 20000
    sequence_length = 100
    num_levels = 10
    features_per_level = 4
    prediction_horizons = [10, 50, 100]
    train_split = 0.7
    val_split = 0.15
    
    # Model  
    input_dim = 40  # num_levels * features_per_level
    hidden_dim = 128
    d_state = 16
    d_conv = 4
    expand = 2
    num_layers = 4
    num_horizons = 3
    num_classes = 3
    dropout = 0.1
    
    # Training
    batch_size = 256
    learning_rate = 1e-4
    weight_decay = 1e-5
    epochs = 50  # More epochs with GPU!
    
config = Config()
print(f"\n📊 Config: {config.epochs} epochs, batch_size={config.batch_size}")

# =============================================================================
# Mamba Model
# =============================================================================
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        
        self.dt_rank = max(d_state // 2, 1)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        batch, seq_len, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :seq_len]
        x = rearrange(x, "b d l -> b l d")
        x = F.silu(x)
        
        y = self.ssm(x)
        y = y * F.silu(z)
        output = self.out_proj(y)
        return self.dropout(output)

    def ssm(self, x):
        batch, seq_len, _ = x.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        
        x_proj = self.x_proj(x)
        dt_proj_input, B, C = x_proj.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt_proj_input))
        
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(-2)
        
        return self.selective_scan(x, deltaA, deltaB, C, D)

    def selective_scan(self, x, deltaA, deltaB, C, D):
        batch, seq_len, d_inner = x.shape
        h = torch.zeros(batch, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            h = deltaA[:, t] * h + deltaB[:, t] * x_t.unsqueeze(-1)
            y = (h * C[:, t].unsqueeze(1)).sum(dim=-1) + D * x_t
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)


class MambaLOBModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        self.layers = nn.ModuleList([
            MambaBlock(config.hidden_dim, config.d_state, config.d_conv, config.expand, config.dropout)
            for _ in range(config.num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(config.hidden_dim) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.heads = nn.ModuleList([
            nn.Linear(config.hidden_dim, config.num_classes) for _ in range(config.num_horizons)
        ])
        
    def forward(self, x):
        x = self.input_proj(x)
        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(norm(x))
        x = self.final_norm(x)
        x = x[:, -1, :]
        return [head(x) for head in self.heads]

# =============================================================================
# Create Synthetic Data
# =============================================================================
def create_synthetic_data(config):
    np.random.seed(42)
    n = config.num_samples
    seq_len = config.sequence_length
    n_features = config.num_levels * config.features_per_level
    
    X = np.zeros((n, seq_len, n_features), dtype=np.float32)
    
    for i in range(n):
        mid_price = 5000 + np.cumsum(np.random.randn(seq_len) * 0.5)
        spread = np.abs(np.random.randn(seq_len) * 0.1) + 0.25
        
        for level in range(config.num_levels):
            bid_price = mid_price - spread * (level + 1)
            ask_price = mid_price + spread * (level + 1)
            bid_qty = np.random.exponential(50, seq_len) * (1 + level * 0.2)
            ask_qty = np.random.exponential(50, seq_len) * (1 + level * 0.2)
            
            base_idx = level * 4
            X[i, :, base_idx] = (bid_price - mid_price) / mid_price
            X[i, :, base_idx + 1] = np.log1p(bid_qty) / 5
            X[i, :, base_idx + 2] = (ask_price - mid_price) / mid_price
            X[i, :, base_idx + 3] = np.log1p(ask_qty) / 5
    
    y = np.zeros((n, len(config.prediction_horizons)), dtype=np.int64)
    for i in range(n):
        for h_idx, horizon in enumerate(config.prediction_horizons):
            future_return = np.random.randn() * 0.001 * np.sqrt(horizon)
            imbalance = np.mean(X[i, -10:, 1]) - np.mean(X[i, -10:, 3])
            future_return += imbalance * 0.01
            
            if future_return > 0.0001:
                y[i, h_idx] = 2
            elif future_return < -0.0001:
                y[i, h_idx] = 0
            else:
                y[i, h_idx] = 1
    
    return X, y

print("\n📦 Creating synthetic data...")
X, y = create_synthetic_data(config)
print(f"   Data shape: X={X.shape}, y={y.shape}")

# Split
n = len(X)
train_end = int(n * config.train_split)
val_end = int(n * (config.train_split + config.val_split))

train_dataset = TensorDataset(torch.from_numpy(X[:train_end]), torch.from_numpy(y[:train_end]))
val_dataset = TensorDataset(torch.from_numpy(X[train_end:val_end]), torch.from_numpy(y[train_end:val_end]))
test_dataset = TensorDataset(torch.from_numpy(X[val_end:]), torch.from_numpy(y[val_end:]))

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# =============================================================================
# Training Functions
# =============================================================================
def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for X, y in tqdm(loader, desc="Training"):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(X)
                loss = sum(criterion(logits[i], y[:, i]) for i in range(len(logits))) / len(logits)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(X)
            loss = sum(criterion(logits[i], y[:, i]) for i in range(len(logits))) / len(logits)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        pred = logits[0].argmax(dim=1)
        correct += (pred == y[:, 0]).sum().item()
        total += y.size(0)
    
    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = [0, 0, 0]
    total = 0
    
    with torch.no_grad():
        for X, y in tqdm(loader, desc="Evaluating"):
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = sum(criterion(logits[i], y[:, i]) for i in range(len(logits))) / len(logits)
            total_loss += loss.item()
            
            for i in range(len(logits)):
                pred = logits[i].argmax(dim=1)
                correct[i] += (pred == y[:, i]).sum().item()
            total += y.size(0)
    
    return total_loss / len(loader), [c / total for c in correct]

# =============================================================================
# Initialize Model
# =============================================================================
print("\n🧠 Initializing model...")
model = MambaLOBModel(config).to(device)
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

# =============================================================================
# Training Loop
# =============================================================================
os.makedirs('checkpoints', exist_ok=True)
best_val_loss = float('inf')
best_model_path = 'checkpoints/best_model.pt'

print(f"\n{'='*60}")
print(f"🚀 Starting training for {config.epochs} epochs...")
print(f"{'='*60}\n")

for epoch in range(config.epochs):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    
    print(f"\nEpoch {epoch+1}/{config.epochs}")
    print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
    print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc[0]:.4f}")
    print(f"  Horizon Acc: [{val_acc[0]:.4f}, {val_acc[1]:.4f}, {val_acc[2]:.4f}]")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'val_accuracy': val_acc[0],
        }, best_model_path)
        print(f"  ✅ Saved best model")

# =============================================================================
# Final Evaluation
# =============================================================================
print(f"\n{'='*60}")
print("📊 Final evaluation on test set...")
print(f"{'='*60}")

checkpoint = torch.load(best_model_path, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
test_loss, test_acc = evaluate(model, test_loader, criterion, device)

print(f"\n🎯 Test Results:")
print(f"   Loss: {test_loss:.4f}")
print(f"   Accuracy: {test_acc[0]:.4f}")
print(f"   Horizon Accuracies: [{test_acc[0]:.4f}, {test_acc[1]:.4f}, {test_acc[2]:.4f}]")

# =============================================================================
# Export to ONNX
# =============================================================================
print(f"\n{'='*60}")
print("📦 Exporting to ONNX...")
print(f"{'='*60}")

model.eval()
dummy_input = torch.randn(1, config.sequence_length, config.input_dim).to(device)
onnx_path = 'checkpoints/model.onnx'

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=['input'],
    output_names=['horizon_0', 'horizon_1', 'horizon_2'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'horizon_0': {0: 'batch_size'},
        'horizon_1': {0: 'batch_size'},
        'horizon_2': {0: 'batch_size'},
    },
    opset_version=14,
)

print(f"✅ ONNX model saved to: {onnx_path}")

# =============================================================================
# Benchmark Latency
# =============================================================================
print(f"\n{'='*60}")
print("⏱️ Benchmarking inference latency...")
print(f"{'='*60}")

test_input = torch.randn(1, config.sequence_length, config.input_dim).to(device)

# Warmup
for _ in range(10):
    with torch.no_grad():
        _ = model(test_input)
if device.type == 'cuda':
    torch.cuda.synchronize()

# Benchmark
latencies = []
for _ in range(100):
    start = time.perf_counter_ns()
    with torch.no_grad():
        _ = model(test_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    latencies.append(time.perf_counter_ns() - start)

latencies = np.array(latencies)
print(f"\n📈 Latency Results:")
print(f"   Mean:   {latencies.mean()/1000:.2f} µs")
print(f"   Median: {np.median(latencies)/1000:.2f} µs")
print(f"   P95:    {np.percentile(latencies, 95)/1000:.2f} µs")
print(f"   P99:    {np.percentile(latencies, 99)/1000:.2f} µs")

# =============================================================================
# Done!
# =============================================================================
print(f"\n{'='*60}")
print("🎉 TRAINING COMPLETE!")
print(f"{'='*60}")
print(f"\n📥 Download your model:")
print(f"   1. Click the Files icon on the left sidebar")
print(f"   2. Navigate to checkpoints/")
print(f"   3. Download: model.onnx")
print(f"\n   Place it in: nanoARB/models/model.onnx")

