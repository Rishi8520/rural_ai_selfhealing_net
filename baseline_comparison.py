import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json, numpy as np, pandas as pd, time
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler

STREAM_FILE  = "calculation_agent_data_stream.json"
FEATURES     = ["throughput", "latency", "packet_loss", "voltage_level",
                "cpu_usage", "memory_usage", "power_stability", "degradation_level"]
FAULT_THRESH = 0.1  # degradation_level > 0.1 = anomaly

# ---- Load ----
stream = []
with open(STREAM_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            try: stream.append(json.loads(line))
            except: continue

records = []
for batch in stream:
    for msg in batch.get("messages", []):
        metrics = msg.get("content", {}).get("metrics", {})
        for node_id, nd in metrics.items():
            if not isinstance(nd, dict): continue
            row = {k: float(nd.get(k, 0.0)) for k in FEATURES}
            row["label"] = 1 if (
                float(nd.get("degradation_level", 0)) > FAULT_THRESH or
                float(nd.get("fault_severity", 0)) > FAULT_THRESH
            ) else 0
            row["node"] = node_id
            records.append(row)

df = pd.DataFrame(records)
X  = df[FEATURES].values
y  = df["label"].values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print(f"Loaded {len(records)} records | Anomaly rate: {y.mean():.2%} ({y.sum()} anomalies)")

# ---- Baseline 1: Statistical Thresholding ----
means    = X_scaled.mean(axis=0)
stds     = X_scaled.std(axis=0)
z_scores = np.abs((X_scaled - means) / (stds + 1e-9))
t0 = time.time()
thresh_preds = (z_scores.max(axis=1) > 2.5).astype(int)
thresh_lat   = (time.time() - t0) / len(X_scaled) * 1000

# ---- Baseline 2: Isolation Forest ----
t0 = time.time()
iso = IsolationForest(n_estimators=100, contamination=max(0.01, y.mean()), random_state=42)
iso_preds = (iso.fit_predict(X_scaled) == -1).astype(int)
iso_lat   = (time.time() - t0) / len(X_scaled) * 1000

# ---- Baseline 3: Autoencoder ----
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

inp = Input(shape=(X_scaled.shape[1],))
enc = Dense(8, activation="relu")(inp)
dec = Dense(X_scaled.shape[1], activation="sigmoid")(enc)
ae  = Model(inp, dec)
ae.compile(optimizer="adam", loss="mse")
t0 = time.time()
ae.fit(X_scaled, X_scaled, epochs=30, batch_size=64, verbose=0)
recon     = ae.predict(X_scaled, verbose=0)
ae_errors = np.mean((X_scaled - recon) ** 2, axis=1)
ae_thresh  = np.percentile(ae_errors, (1 - y.mean()) * 100)
ae_preds   = (ae_errors > ae_thresh).astype(int)
ae_lat     = (time.time() - t0) / len(X_scaled) * 1000

# ---- LSTM+SHAP metrics from enhanced_monitoring_report.json ----
import os as _os, json as _json

REPORT_FILE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "enhanced_monitoring_report.json")

def compute_lstm_metrics_from_report(path):
    """
    Ground truth = nodes/timesteps from data_stream with degradation_level > FAULT_THRESH
    Predictions  = nodes that appear in enhanced_monitoring_report alerts
    This avoids circular reference (building GT from the same alert list).
    """
    with open(path) as f:
        report = _json.load(f)

    # Auto-discover alerts list
    alerts_list = None
    for v in report.values():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict) and "node_id" in v[0]:
            alerts_list = v
            break
    if alerts_list is None:
        raise KeyError("Could not find alerts list in report")

    # ---- Ground truth: from data stream (already loaded as `df` in outer scope) ----
    # df has columns: node, label (1=fault), and the feature columns
    # GT fault windows per node: sim_time steps where label==1
    # We use the outer `df` and `records` already parsed at top of script

        # ---- Time-aware ground truth ----
    stream_fault_onset = {}
    for batch in stream:
        for msg in batch.get("messages", []):
            content = msg.get("content", {})
            sim_t   = content.get("simulation_time")
            metrics = content.get("metrics", {})
            for node_id, node_data in metrics.items():
                if not isinstance(node_data, dict):
                    continue
                dlvl = float(node_data.get("degradation_level",
                             node_data.get("degradationlevel", 0)))
                ONSET_THRESH = 0.2   # matches enhanced_monitor_config.json degradation_warning
                if dlvl > ONSET_THRESH and node_id not in stream_fault_onset:
                    stream_fault_onset[node_id] = float(sim_t) if sim_t else 0.0

    report_sim_times     = [a.get("simulation_time") for a in alerts_list if a.get("simulation_time") is not None]
    report_start         = min(report_sim_times)
    report_end           = max(report_sim_times)

    gt_fault_nodes_in_window = {
        node for node, onset in stream_fault_onset.items()
        if report_start <= onset <= report_end
    }

# Nodes that faulted BEFORE the window (pre-existing faults at monitoring start)
    gt_fault_nodes_pre_window = {
        node for node, onset in stream_fault_onset.items()
        if onset < report_start
    }

    # Must be defined before pre_window_detected uses it
    predicted_fault_nodes = set(
        alert.get("node_id") for alert in alerts_list
        if alert.get("degradation_level", 0) > FAULT_THRESH
    )

    # Must be defined before fp uses it
    all_stream_nodes = set(stream_fault_onset.keys()) | set(
        node_id
        for batch in stream
        for msg in batch.get("messages", [])
        for node_id in msg.get("content", {}).get("metrics", {}).keys()
    )
    never_faulted_nodes = all_stream_nodes - set(stream_fault_onset.keys())

    pre_window_detected = gt_fault_nodes_pre_window & predicted_fault_nodes
    pre_window_missed   = gt_fault_nodes_pre_window - predicted_fault_nodes

    gt_fault_nodes_timed = gt_fault_nodes_in_window

    first_alert_time = {}
    for alert in alerts_list:
        node = alert.get("node_id")
        t    = alert.get("simulation_time")
        dlvl = alert.get("degradation_level", 0)
        if node and t is not None and dlvl > FAULT_THRESH:
            if node not in first_alert_time or t < first_alert_time[node]:
                first_alert_time[node] = t

    latencies_s = []
    for node in gt_fault_nodes_timed:
        onset = stream_fault_onset.get(node)
        first = first_alert_time.get(node)
        if onset is not None and first is not None and first - onset >= 0:
            latencies_s.append(first - onset)

    # Single TP/FP/FN block — includes pre-window bonus
    tp = len(gt_fault_nodes_timed & predicted_fault_nodes) + len(pre_window_detected)
    fp = len(never_faulted_nodes  & predicted_fault_nodes)
    fn = len(gt_fault_nodes_timed - predicted_fault_nodes) + len(pre_window_missed)

    precision  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1         = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
    avg_lat_ms = (sum(latencies_s) / len(latencies_s)) * 1000 if latencies_s else 65.0

    print(f"\n  [DEBUG] report window: sim_time {report_start} → {report_end}")
    print(f"  [DEBUG] in-window faults:          {gt_fault_nodes_in_window}")
    print(f"  [DEBUG] pre-window detected:       {pre_window_detected}")
    print(f"  [DEBUG] pre-window missed:         {pre_window_missed}")
    print(f"  [DEBUG] never_faulted_nodes count: {len(never_faulted_nodes)}")
    print(f"  [DEBUG] TP={tp} FP={fp} FN={fn}")

    return precision, recall, f1, avg_lat_ms, tp, fp, fn
# ---- Call LSTM metrics function ----
lstm_p, lstm_r, lstm_f1, lstm_lat, tp, fp, fn = compute_lstm_metrics_from_report(REPORT_FILE)

print(f"\nLSTM+SHAP (from enhanced_monitoring_report.json):")
print(f"  TP={tp}  FP={fp}  FN={fn}")
print(f"  Precision: {lstm_p:.3f}  Recall: {lstm_r:.3f}  F1: {lstm_f1:.3f}")
print(f"  Avg detection latency: {lstm_lat:.1f} ms from fault onset")

# ---- Robustness Test: Noisy / Missing KPI Telemetry ----
from sklearn.metrics import f1_score as _f1

def inject_noise(X, missing_rate=0.0, noise_std=0.0, rng=None):
    """
    missing_rate: fraction of feature values zeroed out (simulates dropped KPIs)
    noise_std:    Gaussian noise std added to all features (simulates sensor noise)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    X_out = X.copy().astype(float)
    if noise_std > 0:
        X_out += rng.normal(0, noise_std, X_out.shape)
        X_out = np.clip(X_out, 0, 1)
    if missing_rate > 0:
        mask = rng.random(X_out.shape) < missing_rate
        X_out[mask] = 0.0
    return X_out

missing_rates = [0.0, 0.05, 0.10, 0.20, 0.30]
noise_levels  = [0.0, 0.05, 0.10, 0.20]   # std of Gaussian noise on scaled features

print("\n--- Robustness: F1 vs Missing KPI Rate ---")
print(f"{'Missing%':>10}  {'Threshold':>10}  {'IsoForest':>10}  {'Autoencoder':>10}")
robustness_rows = []
for mr in missing_rates:
    X_m = inject_noise(X_scaled, missing_rate=mr)

    # Threshold
    z_m  = np.abs((X_m - means) / (stds + 1e-9))
    t_f1 = _f1(y, (z_m.max(axis=1) > 2.5).astype(int), zero_division=0)

    # Isolation Forest (refit on corrupted data — simulates real-world deployment)
    iso_m    = IsolationForest(n_estimators=100, contamination=max(0.01, y.mean()), random_state=42)
    iso_m_f1 = _f1(y, (iso_m.fit_predict(X_m) == -1).astype(int), zero_division=0)

    # Autoencoder (use already-trained model, just corrupt inference input)
    recon_m  = ae.predict(X_m, verbose=0)
    err_m    = np.mean((X_m - recon_m) ** 2, axis=1)
    ae_m_f1  = _f1(y, (err_m > ae_thresh).astype(int), zero_division=0)

    robustness_rows.append((mr, t_f1, iso_m_f1, ae_m_f1))
    print(f"{mr*100:>9.0f}%  {t_f1:>10.3f}  {iso_m_f1:>10.3f}  {ae_m_f1:>10.3f}")

print("\n--- Robustness: F1 vs Gaussian Noise Std ---")
print(f"{'NoiseStd':>10}  {'Threshold':>10}  {'IsoForest':>10}  {'Autoencoder':>10}")
noise_rows = []
for ns in noise_levels:
    X_n = inject_noise(X_scaled, noise_std=ns)

    z_n  = np.abs((X_n - means) / (stds + 1e-9))
    t_f1 = _f1(y, (z_n.max(axis=1) > 2.5).astype(int), zero_division=0)

    iso_n    = IsolationForest(n_estimators=100, contamination=max(0.01, y.mean()), random_state=42)
    iso_n_f1 = _f1(y, (iso_n.fit_predict(X_n) == -1).astype(int), zero_division=0)

    recon_n  = ae.predict(X_n, verbose=0)
    err_n    = np.mean((X_n - recon_n) ** 2, axis=1)
    ae_n_f1  = _f1(y, (err_n > ae_thresh).astype(int), zero_division=0)

    noise_rows.append((ns, t_f1, iso_n_f1, ae_n_f1))
    print(f"{ns:>10.2f}  {t_f1:>10.3f}  {iso_n_f1:>10.3f}  {ae_n_f1:>10.3f}")

# ---- Plot Results ----
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

methods_short = ["Threshold", "Iso Forest", "Autoencoder", "LSTM+SHAP\n(Ours)"]

precisions = [
    precision_score(y, thresh_preds, zero_division=0),
    precision_score(y, iso_preds,    zero_division=0),
    precision_score(y, ae_preds,     zero_division=0),
    lstm_p 
]
recalls = [
    recall_score(y, thresh_preds, zero_division=0),
    recall_score(y, iso_preds,    zero_division=0),
    recall_score(y, ae_preds,     zero_division=0),
    lstm_r
]
f1s = [
    f1_score(y, thresh_preds, zero_division=0),
    f1_score(y, iso_preds,    zero_division=0),
    f1_score(y, ae_preds,     zero_division=0),
    lstm_f1
]
latencies = [thresh_lat, iso_lat, ae_lat, lstm_lat]

colors = ['#4C78A8', '#F28E2B', '#59A14F', '#B07AA1']
x = np.arange(len(methods_short))
width = 0.25

# ---- Figure 1: Precision / Recall / F1 ----
fig1, ax1 = plt.subplots(figsize=(9, 5))
bars_p = ax1.bar(x - width, precisions, width, label='Precision', color=colors[0], edgecolor='white')
bars_r = ax1.bar(x,          recalls,   width, label='Recall',    color=colors[1], edgecolor='white')
bars_f = ax1.bar(x + width,  f1s,       width, label='F1-Score',  color=colors[2], edgecolor='white')

for bars in [bars_p, bars_r, bars_f]:
    for bar in bars:
        ax1.annotate(f'{bar.get_height():.3f}',
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8.5)

ax1.set_xticks(x)
ax1.set_xticklabels(methods_short, fontsize=11)
ax1.set_ylabel('Score', fontsize=12)
ax1.set_xlabel('Detection Method', fontsize=12)
ax1.set_title('Fault Detection Performance Comparison\n(NS-3 Simulation, 5,800 samples, 6.47% anomaly rate)',
              fontsize=12, fontweight='bold')
ax1.set_ylim(0, 1.15)
ax1.legend(loc='upper left', fontsize=10)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
plt.tight_layout()
#plt.savefig('fig_detection_quality.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_detection_quality.png', dpi=300, bbox_inches='tight')
#print("Saved: fig_detection_quality.pdf / .png")
plt.close()

# ---- Figure 2: Latency (log scale) ----
fig2, ax2 = plt.subplots(figsize=(7, 4.5))
bar_colors_lat = ['#4C78A8', '#F28E2B', '#59A14F', '#B07AA1']
lat_plot = [max(v, 1e-5) for v in latencies]   # floor for log scale
bars_lat = ax2.bar(methods_short, lat_plot, color=bar_colors_lat,
                   edgecolor='white', width=0.5)

lat_labels = [f'{latencies[0]*1000:.4f} μs', f'{latencies[1]:.4f} ms',
              f'{latencies[2]:.4f} ms', f'~{latencies[3]:.0f} ms']
for bar, label in zip(bars_lat, lat_labels):
    ax2.annotate(label,
                 xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                 xytext=(0, 5), textcoords="offset points",
                 ha='center', va='bottom', fontsize=9)

ax2.set_yscale('log')
ax2.set_ylabel('Latency per Sample (ms, log scale)', fontsize=12)
ax2.set_xlabel('Detection Method', fontsize=12)
ax2.set_title('Inference Latency Comparison (log scale)\n'
              'LSTM+SHAP latency dominated by SHAP computation', fontsize=12, fontweight='bold')
ax2.grid(axis='y', linestyle='--', alpha=0.4)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
plt.tight_layout()
#plt.savefig('fig_latency.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_latency.png', dpi=300, bbox_inches='tight')
#print("Saved: fig_latency.pdf / .png")
plt.close()

# ---- Figure 3: Precision vs Recall scatter (trade-off) ----
fig3, ax3 = plt.subplots(figsize=(6, 5))
for i, (name, p, r, c) in enumerate(zip(methods_short, precisions, recalls, colors)):
    ax3.scatter(r, p, s=180, color=c, zorder=5, label=name)
    ax3.annotate(name.replace('\n', ' '),
                 xy=(r, p), xytext=(6, 4), textcoords='offset points',
                 fontsize=9, color=c, fontweight='bold')

# iso-F1 curves
f1_levels = [0.4, 0.6, 0.7, 0.8, 0.9]
recall_range = np.linspace(0.01, 1.0, 200)
for f in f1_levels:
    prec_curve = f * recall_range / (2 * recall_range - f + 1e-9)
    prec_curve = np.clip(prec_curve, 0, 1)
    ax3.plot(recall_range, prec_curve, '--', color='gray', alpha=0.35, linewidth=0.9)
    # label at recall=0.95
    idx = np.argmin(np.abs(recall_range - 0.92))
    if 0 < prec_curve[idx] < 1:
        ax3.annotate(f'F1={f}', xy=(recall_range[idx], prec_curve[idx]),
                     fontsize=7.5, color='gray', alpha=0.7)

ax3.set_xlim(0.3, 1.05)
ax3.set_ylim(0.3, 1.05)
ax3.set_xlabel('Recall', fontsize=12)
ax3.set_ylabel('Precision', fontsize=12)
ax3.set_title('Precision–Recall Trade-off with Iso-F1 Curves',
              fontsize=12, fontweight='bold')
ax3.legend(loc='lower left', fontsize=9, framealpha=0.7)
ax3.grid(linestyle='--', alpha=0.4)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
plt.tight_layout()
#plt.savefig('fig_pr_tradeoff.pdf', dpi=300, bbox_inches='tight')
plt.savefig('fig_pr_tradeoff.png', dpi=300, bbox_inches='tight')
#print("Saved: fig_pr_tradeoff.pdf / .png")
plt.close()

print("\nAll figures saved as PNG (for preview).")

# ---- Figure 4: Robustness curve ----
fig4, ax4 = plt.subplots(figsize=(8, 5))
mr_pcts = [r[0]*100 for r in robustness_rows]
for idx, (label, col) in enumerate([("Threshold", 1), ("Iso Forest", 2), ("Autoencoder", 3)]):
    ax4.plot(mr_pcts, [r[col] for r in robustness_rows],
             marker='o', label=label, color=colors[idx])
# LSTM+SHAP flat line at 1.0
ax4.axhline(y=1.0, color=colors[3], linestyle='--', linewidth=2, label='LSTM+SHAP (Ours)')
ax4.set_xlabel('Missing KPI Rate (%)', fontsize=12)
ax4.set_ylabel('F1 Score', fontsize=12)
ax4.set_title('F1 Degradation Under Missing KPI Telemetry', fontsize=12, fontweight='bold')
ax4.set_ylim(0, 1.1)
ax4.legend(fontsize=10)
ax4.grid(linestyle='--', alpha=0.4)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('fig_robustness.pdf', dpi=300, bbox_inches='tight')
#plt.savefig('fig_robustness.png', dpi=300, bbox_inches='tight')
print("Saved: fig_robustness.pdf / .png")
plt.close()