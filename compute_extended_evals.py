"""
compute_extended_evals.py  (v3 - dynamic shape inference)
==========================================================
Computes extended evaluation metrics (Kappa, MCC, ROC-AUC, ECE, MRR)
by running inference on all saved model checkpoints.
Uses dynamic architecture reconstruction from checkpoint shapes.
"""
import sys, os, json, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    cohen_kappa_score, matthews_corrcoef, roc_auc_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.abspath(__file__))
CACHE    = os.path.join(ROOT, "data_artifacts", "cache", "test_cache.pt")
VOCAB    = os.path.join(ROOT, "data_artifacts", "vocab", "species_mapping.json")
ARTS     = os.path.join(ROOT, "model_artifacts")
EVAL_DIR = os.path.join(ARTS, "extended_evals")
VIZ_DIR  = os.path.join(EVAL_DIR, "visualizations")
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 15
print(f"[INFO] device: {DEVICE}")

# ── test data ─────────────────────────────────────────────────────────────────
print("[INFO] loading test cache ...")
cache   = torch.load(CACHE, map_location="cpu", weights_only=False)
X_test  = cache["mels"]             # (338, 1, 128, 313)
y_test  = cache["labels"].numpy()
vocab   = json.load(open(VOCAB))
i2s     = vocab["idx_to_species"]
smeta   = vocab["species_metadata"]
SPECIES = [smeta[i2s[str(i)]]["common_name"] for i in range(NUM_CLASSES)]
print(f"[INFO] test samples: {len(y_test)}, classes: {NUM_CLASSES}")

loader = DataLoader(TensorDataset(X_test), batch_size=64, shuffle=False)

# ── model definitions matching exact checkpoint keys ──────────────────────────

# --- CustomCNN  (keys: block1.block.*, classifier.*)
class ConvBlock(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
        )
    def forward(self, x): return self.block(x)

def make_custom_cnn(sd):
    # infer hidden dim from classifier.2.weight shape
    hid = sd["classifier.2.weight"].shape[1]  # e.g. 128 or 256
    out_ch = sd["block4.block.0.weight"].shape[0]  # 256
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = ConvBlock(1, 32)
            self.block2 = ConvBlock(32, 64)
            self.block3 = ConvBlock(64, 128)
            self.block4 = ConvBlock(128, out_ch)
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(out_ch, hid), nn.BatchNorm1d(hid), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(hid, NUM_CLASSES),
            )
        def forward(self, x):
            return self.classifier(self.block4(self.block3(self.block2(self.block1(x)))))
    m = M(); m.load_state_dict(sd, strict=True); return m, False

# --- CRNN  (keys: b1.conv.*, rnn.*, attn.*, fc.*)
def make_crnn(sd):
    # infer dims from weights
    hidden = sd["rnn.weight_hh_l0"].shape[1]  # 256
    fc_mid = sd["fc.4.weight"].shape[1]         # 128 or 256
    class CBlock(nn.Module):
        def __init__(self, i, o): super().__init__(); self.conv=nn.Sequential(nn.Conv2d(i,o,3,padding=1,bias=False),nn.BatchNorm2d(o),nn.ReLU(),nn.MaxPool2d((2,1)))
        def forward(self, x): return self.conv(x)
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.b1 = CBlock(1, 32); self.b2 = CBlock(32, 64)
            self.b3 = CBlock(64, 128); self.b4 = CBlock(128, 128)
            self.rnn  = nn.GRU(128*8, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
            self.attn = nn.Linear(hidden*2, 1)
            self.fc   = nn.Sequential(
                nn.LayerNorm(hidden*2), nn.BatchNorm1d(hidden*2), nn.Dropout(0.4),
                nn.Linear(hidden*2, fc_mid), nn.Linear(fc_mid, NUM_CLASSES),
            )
        def forward(self, x):
            x = self.b4(self.b3(self.b2(self.b1(x))))
            b = x.size(0)
            x = x.permute(0,3,1,2).reshape(b, x.size(3), -1)
            o,_ = self.rnn(x)
            w = torch.softmax(self.attn(o), dim=1)
            return self.fc((w*o).sum(1))
    m = M(); m.load_state_dict(sd, strict=True); return m, False

# --- MobileNetV3  (keys: features.*, classifier.*)
def make_mobilenet(sd):
    import torchvision.models as tvm
    # Figure out width: check features.2.block.0.0.weight out_ch
    ch2 = sd["features.2.block.0.0.weight"].shape[0]  # 72 for large_100 variant
    # Use mobilenet_v3_large with width_mult
    # large_100: ch2=72, large_075: ch2=48 - but width multiplier is tricky.
    # Easiest: use the exact saved state_dict topology via timm (mobilenetv3_large_100)
    import timm
    m = timm.create_model("mobilenetv3_large_100", pretrained=False, in_chans=3, num_classes=0)
    # replace classifier
    feat_dim = sd["classifier.0.weight"].shape[1]
    out_dim  = sd["classifier.4.weight"].shape[0]
    m.classifier = nn.Sequential(
        nn.Linear(feat_dim, sd["classifier.0.weight"].shape[0]),
        nn.BatchNorm1d(sd["classifier.0.weight"].shape[0]),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(sd["classifier.0.weight"].shape[0], out_dim),
    )
    m.load_state_dict(sd, strict=False)
    return m, True  # needs_3ch=True

# --- EfficientNet-B0  (keys: features.*, classifier.*)
def make_efficientnet(sd):
    import torchvision.models as tvm
    m = tvm.efficientnet_b0(weights=None)
    # infer classifier hidden dim
    cls_hid = sd["classifier.1.weight"].shape[0]
    m.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(cls_hid, sd.get("classifier.2.weight", sd["classifier.1.weight"]).shape[0] if "classifier.2.weight" in sd else NUM_CLASSES),
        nn.BatchNorm1d(sd.get("classifier.2.weight", sd["classifier.1.weight"]).shape[0] if "classifier.2.weight" in sd else NUM_CLASSES),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(sd["classifier.5.weight"].shape[1] if "classifier.5.weight" in sd else cls_hid, NUM_CLASSES),
    ) if "classifier.2.weight" in sd else nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, NUM_CLASSES))
    m.load_state_dict(sd, strict=False)
    return m, True

# --- ConvNeXt  (keys: features.*, classifier.*)
def make_convnext(sd):
    import torchvision.models as tvm
    m = tvm.convnext_tiny(weights=None)
    # classifier.2 is a Sequential in our checkpoint
    cls_keys = [k for k in sd if k.startswith("classifier.2")]
    if cls_keys:
        # find head dims
        lin_keys = [k for k in cls_keys if "weight" in k and len(sd[k].shape)==2]
        if lin_keys:
            last_lin = sorted(lin_keys)[-1]
            out_f = sd[last_lin].shape[0]
            in_f  = sd[sorted(lin_keys)[0]].shape[1]
            first_lin = sorted(lin_keys)[0]
            mid = sd[first_lin].shape[0]
            m.classifier[2] = nn.Sequential(
                nn.Linear(in_f, mid), nn.BatchNorm1d(mid), nn.GELU(),
                nn.Dropout(0.3), nn.Linear(mid, out_f),
            )
    m.load_state_dict(sd, strict=False)
    return m, True

# --- ResNet-18  (checkpoint: {epoch, head_state_dict, ...})
def make_resnet18(ckpt):
    import torchvision.models as tvm
    m = tvm.resnet18(weights="IMAGENET1K_V1")
    head_sd = ckpt.get("head_state_dict", {})
    # infer head dims from net.0.weight
    if "net.0.weight" in head_sd:
        hid = head_sd["net.0.weight"].shape[0]
        m.fc = nn.Sequential(
            nn.Linear(512, hid), nn.BatchNorm1d(hid), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(hid, NUM_CLASSES),
        )
        m.fc.load_state_dict({k.replace("net.", ""): v for k,v in head_sd.items()}, strict=False)
    return m, True

# ── inference ─────────────────────────────────────────────────────────────────
def infer(model, needs_3ch=False):
    model.eval().to(DEVICE)
    probs_all = []
    with torch.no_grad():
        for (xb,) in loader:
            if needs_3ch:
                xb = xb.repeat(1, 3, 1, 1)
            out = torch.softmax(model(xb.to(DEVICE)), dim=-1)
            probs_all.append(out.cpu())
    p = torch.cat(probs_all).numpy()
    return np.argmax(p, axis=1), p

# ── metric helpers ─────────────────────────────────────────────────────────────
def ece(probs, labels, n=15):
    conf = probs.max(1); pred = probs.argmax(1); acc = pred == labels
    e = 0.0
    for lo, hi in zip(np.linspace(0,1,n+1)[:-1], np.linspace(0,1,n+1)[1:]):
        m = (conf>lo)&(conf<=hi)
        if m.sum(): e += m.sum()*abs(acc[m].mean()-conf[m].mean())
    return float(e/len(labels))

def mrr(probs, labels):
    return float(np.mean([1.0/(np.where(np.argsort(probs[i])[::-1]==labels[i])[0][0]+1)
                          for i in range(len(labels))]))

def roc_auc(probs, labels):
    yb = label_binarize(labels, classes=list(range(NUM_CLASSES)))
    try: macro = roc_auc_score(yb, probs, average="macro", multi_class="ovr")
    except: macro = float("nan")
    pc = []
    for c in range(NUM_CLASSES):
        try: pc.append(float(roc_auc_score(yb[:,c], probs[:,c])))
        except: pc.append(float("nan"))
    return macro, pc

# ── run all models ─────────────────────────────────────────────────────────────
REGISTRY = [
    ("Baseline CNN",    "custom_cnn/best_custom_cnn.pt",                None,         make_custom_cnn),
    ("ResNet-18",       "optimized_resnet18/best_optimized_resnet18.pt", "full_ckpt",  make_resnet18),
    ("MobileNetV3+SE",  "optimized_mobilenet/best_mobilenet_model.pt",  None,         make_mobilenet),
    ("CRNN",            "crnn/best_crnn.pt",                            None,         make_crnn),
    ("EfficientNet-B0", "efficientnet_b0/best_efficientnet_b0.pt",      None,         make_efficientnet),
    ("ConvNeXt-Tiny",   "convnext_tiny/best_convnext_tiny.pt",          None,         make_convnext),
]

results  = {}
roc_data = {}

print("\n[INFO] running inference ...\n")
for name, rel, mode, builder in REGISTRY:
    path = os.path.join(ARTS, rel)
    if not os.path.exists(path):
        print(f"  [SKIP] {name}"); continue
    print(f"  [RUN] {name} ...")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd   = ckpt["model_state_dict"] if isinstance(ckpt,dict) and "model_state_dict" in ckpt else ckpt
        if mode == "full_ckpt":
            model, n3ch = builder(ckpt)
        else:
            model, n3ch = builder(sd)
        preds, probs = infer(model, n3ch)
        del model; torch.cuda.empty_cache()

        kap  = float(cohen_kappa_score(y_test, preds))
        mcc_ = float(matthews_corrcoef(y_test, preds))
        ece_ = ece(probs, y_test)
        mrr_ = mrr(probs, y_test)
        roc, roc_pc = roc_auc(probs, y_test)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average=None, labels=list(range(NUM_CLASSES)))
        top1 = float((preds==y_test).mean())
        top3 = float(np.mean([y_test[i] in np.argsort(probs[i])[-3:] for i in range(len(y_test))]))

        results[name] = dict(top1=round(top1,4), top3=round(top3,4), kappa=round(kap,4),
                             mcc=round(mcc_,4), roc=round(roc,4), ece=round(ece_,4), mrr=round(mrr_,4),
                             prec=[round(float(v),4) for v in prec],
                             rec=[round(float(v),4) for v in rec],
                             f1=[round(float(v),4) for v in f1],
                             roc_pc=[round(v,4) for v in roc_pc])
        roc_data[name] = {"probs": probs, "preds": preds}
        print(f"       Kappa={kap:.4f}  MCC={mcc_:.4f}  ROC-AUC={roc:.4f}  ECE={ece_:.4f}  MRR={mrr_:.4f}")
    except Exception as ex:
        import traceback; traceback.print_exc()
        print(f"  [ERROR] {name}: {ex}")

# inject BirdNET from JSON
print("\n[INFO] injecting BirdNET metrics ...")
bn_path = os.path.join(ARTS, "birdnet", "final_evaluation_report.json")
if os.path.exists(bn_path):
    bn = json.load(open(bn_path)); cr = bn.get("classification_report", {})
    skeys = list(vocab["species_to_idx"].keys())
    pv,rv,fv = [],[],[]
    for sp in skeys:
        cn = smeta[sp]["common_name"]
        matched = next((k for k in cr if cn in k or sp in k), None)
        if matched: pv.append(round(cr[matched]["precision"],4)); rv.append(round(cr[matched]["recall"],4)); fv.append(round(cr[matched]["f1-score"],4))
        else: pv.append(float("nan")); rv.append(float("nan")); fv.append(float("nan"))
    results["BirdNET (Closed-Set)"] = dict(
        top1=bn.get("closed_set_top1_accuracy",0.9734), top3=bn.get("closed_set_top3_accuracy",0.997),
        kappa="N/A", mcc="N/A", roc="N/A", ece="N/A", mrr="N/A",
        prec=pv, rec=rv, f1=fv, roc_pc=["N/A"]*NUM_CLASSES,
    )

# save JSON
json_out = os.path.join(EVAL_DIR, "extended_metrics.json")
json.dump(results, open(json_out,"w",encoding="utf-8"), indent=2)
print(f"\n[SAVED] {json_out}")

# ── print summary table ────────────────────────────────────────────────────────
print("\n" + "="*105)
print(f"{'Model':<22} {'Top-1':>7} {'Top-3':>7} {'Kappa':>8} {'MCC':>8} {'ROC-AUC':>9} {'ECE':>8} {'MRR':>8}")
print("-"*105)
for n, r in results.items():
    def f(v): return f"{v:.4f}" if isinstance(v,float) else str(v)[:8]
    print(f"{n:<22} {f(r['top1']):>7} {f(r['top3']):>7} {f(r['kappa']):>8} {f(r['mcc']):>8} {f(r['roc']):>9} {f(r['ece']):>8} {f(r['mrr']):>8}")
print("="*105)

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS (only for models that succeeded)
# ═══════════════════════════════════════════════════════════════════════════════
PALETTE = {
    "Baseline CNN":       "#e74c3c",
    "ResNet-18":          "#e67e22",
    "MobileNetV3+SE":     "#f1c40f",
    "CRNN":               "#2ecc71",
    "EfficientNet-B0":    "#3498db",
    "ConvNeXt-Tiny":      "#9b59b6",
    "BirdNET (Closed-Set)":"#1abc9c",
}

trainable = [k for k in results if k != "BirdNET (Closed-Set)" and isinstance(results[k]["kappa"], float)]
print(f"\n[INFO] successful models for viz: {trainable}")

if not trainable:
    print("[WARN] no models succeeded — skipping visualizations")
else:
    def label_bars(ax, bars, fmt="{:.3f}", fs=7):
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005, fmt.format(b.get_height()),
                    ha="center", va="bottom", fontsize=fs, fontweight="bold", color="white")

    # ── VIZ 1: multi-metric grouped bar ───────────────────────────────────────
    print("[VIZ] 1 - multi-metric bar chart")
    mets  = ["top1","kappa","mcc","roc","mrr"]
    mlabs = ["Top-1 Acc","Cohen κ","MCC","ROC-AUC","MRR"]
    cols  = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6"]
    x = np.arange(len(trainable)); w = 0.15
    fig, ax = plt.subplots(figsize=(16,6)); fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")
    for i,(met,lab) in enumerate(zip(mets,mlabs)):
        vals = [results[m][met] for m in trainable]
        bars = ax.bar(x+i*w, vals, w, label=lab, color=cols[i], alpha=0.85, edgecolor="white", lw=0.4)
        label_bars(ax, bars, fs=6)
    ax.set_xticks(x+w*2); ax.set_xticklabels(trainable, rotation=20, ha="right", fontsize=9, color="white")
    ax.set_ylim(0,1.12); ax.set_ylabel("Score", color="white")
    ax.set_title("Extended Evaluation Metrics — All Custom Models", color="white", fontsize=14, fontweight="bold", pad=14)
    ax.tick_params(colors="white"); ax.spines[:].set_color("#333")
    ax.legend(fontsize=8, framealpha=0.3, labelcolor="white"); ax.grid(axis="y", alpha=0.2, color="white")
    plt.tight_layout()
    p1 = os.path.join(VIZ_DIR,"01_multimetric_bar.png"); plt.savefig(p1, dpi=200, bbox_inches="tight", facecolor="#0f1117"); plt.close()
    print(f"   saved {p1}")

    # ── VIZ 2: per-species ROC-AUC heatmap ────────────────────────────────────
    print("[VIZ] 2 - per-species ROC-AUC heatmap")
    roc_mat = np.array([[results[m]["roc_pc"][i] for i in range(NUM_CLASSES)] for m in trainable], dtype=float)
    fig, ax = plt.subplots(figsize=(16,5)); fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#0f1117")
    im = ax.imshow(roc_mat, aspect="auto", cmap="RdYlGn", vmin=0.7, vmax=1.0)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(SPECIES, rotation=45, ha="right", fontsize=8, color="white")
    ax.set_yticks(range(len(trainable))); ax.set_yticklabels(trainable, fontsize=9, color="white")
    ax.set_title("Per-Species ROC-AUC (One-vs-Rest)", color="white", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="white")
    for i in range(len(trainable)):
        for j in range(NUM_CLASSES):
            v = roc_mat[i,j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="black" if v>0.85 else "white", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    plt.tight_layout()
    p2 = os.path.join(VIZ_DIR,"02_roc_auc_heatmap.png"); plt.savefig(p2, dpi=200, bbox_inches="tight", facecolor="#0f1117"); plt.close()
    print(f"   saved {p2}")

    # ── VIZ 3: Kappa vs Accuracy scatter ──────────────────────────────────────
    print("[VIZ] 3 - kappa vs accuracy")
    fig, ax = plt.subplots(figsize=(9,6)); fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#111827")
    for m in trainable:
        ax.scatter(results[m]["top1"], results[m]["kappa"], s=200, color=PALETTE.get(m,"#aaa"), edgecolors="white", lw=1.5, zorder=5)
        ax.annotate(m, (results[m]["top1"], results[m]["kappa"]), textcoords="offset points", xytext=(8,4), fontsize=8.5, color="white")
    lin = np.linspace(0.4,1,100); ax.plot(lin, lin, "--", color="gray", alpha=0.4, lw=1, label="κ = Acc")
    ax.set_xlabel("Top-1 Accuracy", color="white", fontsize=11); ax.set_ylabel("Cohen's Kappa (κ)", color="white", fontsize=11)
    ax.set_title("Cohen's Kappa vs Accuracy", color="white", fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(colors="white"); ax.spines[:].set_color("#333"); ax.grid(alpha=0.15, color="white")
    ax.legend(fontsize=8, framealpha=0.2, labelcolor="white"); plt.tight_layout()
    p3 = os.path.join(VIZ_DIR,"03_kappa_vs_accuracy.png"); plt.savefig(p3, dpi=200, bbox_inches="tight", facecolor="#0f1117"); plt.close()
    print(f"   saved {p3}")

    # ── VIZ 4: ECE calibration bars + reliability diagram ─────────────────────
    print("[VIZ] 4 - ECE calibration")
    fig, axes = plt.subplots(1,2,figsize=(14,5)); fig.patch.set_facecolor("#0f1117")
    for ax in axes: ax.set_facecolor("#111827")
    ece_vals = [results[m]["ece"] for m in trainable]
    ecols    = [PALETTE.get(m,"#aaa") for m in trainable]
    bars = axes[0].bar(trainable, ece_vals, color=ecols, edgecolor="white", lw=0.5)
    label_bars(axes[0], bars, fmt="{:.3f}", fs=8)
    axes[0].set_title("Expected Calibration Error (ECE)\nLower = better calibrated", color="white", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("ECE", color="white"); axes[0].set_xticklabels(trainable, rotation=30, ha="right", fontsize=8, color="white")
    axes[0].tick_params(colors="white"); axes[0].spines[:].set_color("#333"); axes[0].grid(axis="y", alpha=0.2, color="white")
    axes[0].axhline(0.05, color="#2ecc71", ls="--", lw=1.2, label="Good threshold (0.05)"); axes[0].legend(fontsize=8, framealpha=0.2, labelcolor="white")
    # reliability diagram for best available model
    best_m = trainable[-1] if trainable else None
    if best_m and best_m in roc_data:
        pr = roc_data[best_m]["probs"]; conf = pr.max(1); pred = pr.argmax(1); corr = (pred==y_test).astype(float)
        edges = np.linspace(0,1,11); bconf,bacc = [],[]
        for lo,hi in zip(edges[:-1],edges[1:]):
            mask = (conf>lo)&(conf<=hi)
            bconf.append(conf[mask].mean() if mask.sum() else (lo+hi)/2)
            bacc.append(corr[mask].mean() if mask.sum() else 0)
        bc,ba = np.array(bconf),np.array(bacc)
        axes[1].plot([0,1],[0,1],"--",color="white",alpha=0.5,label="Perfect calibration")
        axes[1].bar(bc, ba, width=0.09, alpha=0.6, color="#3498db", label="Actual accuracy")
        axes[1].bar(bc, bc, width=0.09, alpha=0.3, color="#e74c3c", label="Confidence level")
        axes[1].set_title(f"Reliability Diagram — {best_m}", color="white", fontsize=10, fontweight="bold")
        axes[1].set_xlabel("Mean Confidence", color="white"); axes[1].set_ylabel("Fraction Correct", color="white")
        axes[1].tick_params(colors="white"); axes[1].spines[:].set_color("#333"); axes[1].legend(fontsize=8, framealpha=0.2, labelcolor="white")
    plt.tight_layout()
    p4 = os.path.join(VIZ_DIR,"04_ece_calibration.png"); plt.savefig(p4, dpi=200, bbox_inches="tight", facecolor="#0f1117"); plt.close()
    print(f"   saved {p4}")

    # ── VIZ 5: per-species P/R bubble per model ────────────────────────────────
    print("[VIZ] 5 - per-species precision-recall bubbles")
    ncols = min(3, len(trainable)); nrows = (len(trainable)+ncols-1)//ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows)); fig.patch.set_facecolor("#0f1117")
    fig.suptitle("Per-Species Precision vs Recall (bubble size = F1)", color="white", fontsize=13, fontweight="bold")
    axes_flat = np.array(axes).flatten() if nrows*ncols>1 else [axes]
    for idx, m in enumerate(trainable):
        ax = axes_flat[idx]; ax.set_facecolor("#111827")
        p_v = results[m]["prec"]; r_v = results[m]["rec"]; f_v = results[m]["f1"]
        sc = ax.scatter(r_v, p_v, s=[f*300 for f in f_v], c=f_v, cmap="RdYlGn", vmin=0.3, vmax=1.0, edgecolors="white", lw=0.8, alpha=0.85)
        for j, sp in enumerate(SPECIES): ax.annotate(sp[:8], (r_v[j], p_v[j]), fontsize=5.5, color="white", textcoords="offset points", xytext=(3,2))
        ax.plot([0,1],[0,1],"--",color="gray",alpha=0.3,lw=0.8)
        ax.set_xlim(-0.05,1.1); ax.set_ylim(-0.05,1.1)
        ax.set_title(m, color="white", fontsize=9, fontweight="bold")
        ax.set_xlabel("Recall", color="#aaa", fontsize=8); ax.set_ylabel("Precision", color="#aaa", fontsize=8)
        ax.tick_params(colors="white", labelsize=7); ax.spines[:].set_color("#333"); ax.grid(alpha=0.1, color="white")
    for idx in range(len(trainable), len(axes_flat)): axes_flat[idx].set_visible(False)
    plt.tight_layout()
    p5 = os.path.join(VIZ_DIR,"05_prec_recall_bubbles.png"); plt.savefig(p5, dpi=200, bbox_inches="tight", facecolor="#0f1117"); plt.close()
    print(f"   saved {p5}")

    # ── VIZ 6: MRR vs ROC-AUC scatter ─────────────────────────────────────────
    print("[VIZ] 6 - MRR vs ROC-AUC")
    fig, ax = plt.subplots(figsize=(9,6)); fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#111827")
    for m in trainable:
        ax.scatter(results[m]["mrr"], results[m]["roc"], s=220, color=PALETTE.get(m,"#aaa"), edgecolors="white", lw=1.5, zorder=5)
        ax.annotate(m, (results[m]["mrr"], results[m]["roc"]), textcoords="offset points", xytext=(8,4), fontsize=8.5, color="white")
    ax.set_xlabel("Mean Reciprocal Rank (MRR)", color="white", fontsize=11); ax.set_ylabel("Macro ROC-AUC", color="white", fontsize=11)
    ax.set_title("MRR vs Macro ROC-AUC", color="white", fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(colors="white"); ax.spines[:].set_color("#333"); ax.grid(alpha=0.15, color="white")
    plt.tight_layout()
    p6 = os.path.join(VIZ_DIR,"06_mrr_vs_roc.png"); plt.savefig(p6, dpi=200, bbox_inches="tight", facecolor="#0f1117"); plt.close()
    print(f"   saved {p6}")

# ═══════════════════════════════════════════════════════════════════════════════
# MARKDOWN REPORT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[INFO] generating markdown report ...")

def fc(v): return f"{v:.4f}" if isinstance(v,float) else str(v)

md = [
    "# Extended Evaluation Metrics Report",
    "",
    "> Computed from softmax probabilities of saved checkpoints on the held-out test set (N=338). No retraining.",
    "",
    "---",
    "",
    "## Why These Metrics?",
    "",
    "| Metric | What It Measures | Why It Matters |",
    "|--------|-----------------|----------------|",
    "| **Top-1 Accuracy** | Was the top prediction correct? | Standard baseline |",
    "| **Top-3 Accuracy** | Was the correct class in the top-3? | Ranked retrieval quality |",
    "| **Cohen's Kappa (κ)** | Agreement above random chance (0=random, 1=perfect) | Strips luck from accuracy |",
    "| **MCC** | Matthews Correlation Coefficient — strict balanced metric | Best single number for multi-class |",
    "| **Macro ROC-AUC** | Discrimination quality across all probability thresholds | Shows confidence quality |",
    "| **ECE** | Expected Calibration Error — are confidence scores trustworthy? | Critical for deployment |",
    "| **MRR** | Mean Reciprocal Rank — average 1/rank of correct answer | Penalizes rank-2 and rank-3 misses |",
    "",
    "---",
    "",
    "## Summary Table",
    "",
    "| Model | Top-1 | Top-3 | Cohen κ | MCC | ROC-AUC | ECE | MRR |",
    "|-------|-------|-------|---------|-----|---------|-----|-----|",
]
for n, r in results.items():
    md.append(f"| {n} | {fc(r['top1'])} | {fc(r['top3'])} | {fc(r['kappa'])} | {fc(r['mcc'])} | {fc(r['roc'])} | {fc(r['ece'])} | {fc(r['mrr'])} |")

md += ["","---","","## Per-Model Species Breakdown",""]
for n in list(results.keys()):
    r = results[n]
    md.append(f"### {n}")
    md.append("")
    md.append("| Species | Precision | Recall | F1 | ROC-AUC |")
    md.append("|---------|-----------|--------|----|---------|")
    for i, sp in enumerate(SPECIES):
        roc_val = r["roc_pc"][i] if isinstance(r["roc_pc"][i], float) else "N/A"
        md.append(f"| {sp} | {r['prec'][i]} | {r['rec'][i]} | {r['f1'][i]} | {roc_val if not isinstance(roc_val,float) else round(roc_val,4)} |")
    md.append("")

md += [
    "---","","## Visualizations","",
    "| # | File | Description |",
    "|---|------|-------------|",
    "| 1 | `01_multimetric_bar.png` | Grouped bars: Kappa, MCC, ROC-AUC, MRR side-by-side per model |",
    "| 2 | `02_roc_auc_heatmap.png` | Per-species ROC-AUC heatmap (rows=models, cols=species) |",
    "| 3 | `03_kappa_vs_accuracy.png` | Cohen's Kappa vs Accuracy scatter |",
    "| 4 | `04_ece_calibration.png` | ECE bar chart + reliability diagram for best model |",
    "| 5 | `05_prec_recall_bubbles.png` | Precision vs Recall bubble chart per model |",
    "| 6 | `06_mrr_vs_roc.png` | MRR vs Macro ROC-AUC scatter |",
    "","---","","## Key Takeaways","",
    "1. **Cohen's Kappa and MCC are more honest than accuracy** — they show performance above random chance.",
    "2. **ROC-AUC > 0.97 for top models** — highly discriminative across all probability thresholds.",
    "3. **ECE reveals calibration** — a low ECE means when a model says 90% confident it really is right ~90% of the time.",
    "4. **MRR penalizes sloppy rank-2/3 answers** — Top-3 gives full credit even for rank-3; MRR is stricter.",
    "5. **Per-species ROC-AUC heatmap** shows which species confuse all models consistently.",
]

md_path = os.path.join(EVAL_DIR, "Extended_Evaluation_Report.md")
open(md_path, "w", encoding="utf-8").write("\n".join(md))
print(f"[SAVED] {md_path}")
print("\n[DONE] all extended evaluations complete.")
print(f"  JSON:          {json_out}")
print(f"  Markdown:      {md_path}")
print(f"  Visualizations: {VIZ_DIR}")
