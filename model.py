import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error, mean_absolute_error, confusion_matrix
from scipy.signal import savgol_filter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.gridspec as gridspec
import warnings
import joblib
import itertools

warnings.filterwarnings('ignore')
np.set_printoptions(precision=5, suppress=True)
torch.set_printoptions(precision=5)
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "BATCH_SIZE": 32,
    "EPOCHS": 300,
    "LR": 1e-3,
    "TEST_SIZE": 0.2,
    "RANDOM_SEED": 42,
    "PATIENCE": 50,
    "N_SPLITS": 5,
    "HMS_COLS": ["As", "Cr", "Pb"],
    "PFAS_COLS": ["PFBA", "PFOA", "PFOS"],
    "PREPROCESS_X": True,
    "X_MODE": "standard",
    "PREPROCESS_Y": True,
    "HMS_Y_MODE": "maxabs",
    "PFAS_Y_MODE": "maxabs",
    "SAVE_DIR": "./results",
    "INTERPRET_DIR": "./results/interpretability",
    "EPSILON": 1e-4,
    "SPEC_PATH": "./data/simulated_absorbance.xlsx",
    "REG_PATH": "./data/concentration.xlsx",
    "PARAM_GRID": {
        "LR": [1e-3, 5e-4],
        "BATCH_SIZE": [16, 32],
        "D_MODEL": [64, 128],
        "NUM_LAYERS": [1, 2]
    }
}

os.makedirs(CONFIG["SAVE_DIR"], exist_ok=True)
os.makedirs(CONFIG["INTERPRET_DIR"], exist_ok=True)
print(f"Results saved to: {CONFIG['SAVE_DIR']}")
print(f"Interpretability results saved to: {CONFIG['INTERPRET_DIR']}")
print(f"Current device: {DEVICE}")

class SpecDataset(Dataset):
    def __init__(self, X, y_cls, y_hms, y_pfas):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_cls = torch.tensor(y_cls, dtype=torch.float32)
        self.y_hms = torch.tensor(y_hms, dtype=torch.float32)
        self.y_pfas = torch.tensor(y_pfas, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].unsqueeze(-1), self.y_cls[idx], self.y_hms[idx], self.y_pfas[idx]

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None, return_attention=False):
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask, need_weights=True)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if return_attention:
            return src, attn
        return src

class CustomTransformerEncoder(nn.TransformerEncoder):
    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        self.attn_weights = []
        for mod in self.layers:
            output, attn = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, return_attention=True)
            self.attn_weights.append(attn)
        if self.norm is not None:
            output = self.norm(output)
        return output, torch.stack(self.attn_weights)

class SpectralExpertModel(nn.Module):
    def __init__(self, wave_num, out_dim, task="reg", d_model=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.task = task
        self.wave_num = wave_num
        self.d_model = d_model
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, self.d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.GELU(),
            nn.Dropout(0.15)
        )
        self.pos_emb = nn.Parameter(torch.randn(1, wave_num, self.d_model))
        encoder_layer = CustomTransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=256,
            batch_first=True,
            dropout=dropout
        )
        self.transformer = CustomTransformerEncoder(encoder_layer, num_layers=num_layers)
        self.flatten = nn.Flatten()
        self.shared_mlp = nn.Sequential(
            nn.Linear(self.d_model * wave_num, 256),
            nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.2)
        )
        if self.task == "cls":
            self.head = nn.Sequential(nn.Linear(64, out_dim))
        else:
            self.head = nn.Sequential(nn.Linear(64, out_dim), nn.ReLU())
            self.physics_decoder = nn.Sequential(
                nn.Linear(out_dim, 64),
                nn.GELU(),
                nn.Linear(64, 256),
                nn.GELU(),
                nn.Linear(256, wave_num)
            )

    def forward(self, x):
        x_perm = x.permute(0, 2, 1)
        feat = self.cnn(x_perm)
        feat = feat.permute(0, 2, 1)
        feat = feat + self.pos_emb
        feat_transformer, self.attn_weights = self.transformer(feat)
        feat_flat = self.flatten(feat_transformer)
        shared_feat = self.shared_mlp(feat_flat)
        pred = self.head(shared_feat)
        if self.task == "reg":
            x_recon = self.physics_decoder(pred)
            return pred, x_recon
        return pred

class LightweightSpectralExpertModel(nn.Module):
    def __init__(self, wave_num, out_dim, task="reg", d_model=64, num_layers=1, dropout=0.3):
        super().__init__()
        self.task = task
        self.wave_num = wave_num
        self.d_model = d_model
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, self.d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.pos_emb = nn.Parameter(torch.randn(1, wave_num, self.d_model))
        encoder_layer = CustomTransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=128,
            batch_first=True,
            dropout=dropout
        )
        self.transformer = CustomTransformerEncoder(encoder_layer, num_layers=num_layers)
        self.flatten = nn.Flatten()
        self.shared_mlp = nn.Sequential(
            nn.Linear(self.d_model * wave_num, 128),
            nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.2)
        )
        if self.task == "cls":
            self.head = nn.Sequential(nn.Linear(64, out_dim))
        else:
            self.head = nn.Sequential(nn.Linear(64, out_dim), nn.ReLU())
            self.physics_decoder = nn.Sequential(
                nn.Linear(out_dim, 64),
                nn.GELU(),
                nn.Linear(64, 256),
                nn.GELU(),
                nn.Linear(256, wave_num)
            )

    def forward(self, x):
        x_perm = x.permute(0, 2, 1)
        feat = self.cnn(x_perm)
        feat = feat.permute(0, 2, 1)
        feat = feat + self.pos_emb
        feat_transformer, self.attn_weights = self.transformer(feat)
        feat_flat = self.flatten(feat_transformer)
        shared_feat = self.shared_mlp(feat_flat)
        pred = self.head(shared_feat)
        if self.task == "reg":
            x_recon = self.physics_decoder(pred)
            return pred, x_recon
        return pred

def get_scaler(mode):
    if mode == "standard":
        return StandardScaler()
    elif mode == "minmax":
        return MinMaxScaler()
    elif mode == "maxabs":
        return MaxAbsScaler()
    return None

def load_raw_data():
    spec_df = pd.read_excel(CONFIG["SPEC_PATH"], index_col=0)
    X = savgol_filter(spec_df.values.astype(float), window_length=15, polyorder=2, deriv=1, axis=1)
    reg_df = pd.read_excel(CONFIG["REG_PATH"], header=0, index_col=0)
    hms_cols = [c for c in CONFIG["HMS_COLS"] if c in reg_df.columns]
    pfas_cols = [c for c in CONFIG["PFAS_COLS"] if c in reg_df.columns]
    all_cols = hms_cols + pfas_cols
    y_reg_all = reg_df[all_cols].values.astype(float)
    y_cls_all = (y_reg_all > CONFIG["EPSILON"]).astype(float)
    X_trainval, X_test, y_cls_trainval, y_cls_test, y_reg_trainval, y_reg_test = train_test_split(
        X, y_cls_all, y_reg_all, test_size=CONFIG["TEST_SIZE"], random_state=CONFIG["RANDOM_SEED"]
    )
    return X_trainval, X_test, y_cls_trainval, y_cls_test, y_reg_trainval, y_reg_test, hms_cols, pfas_cols

def build_dataloaders(X_train, X_val, y_cls_train, y_cls_val, y_reg_train, y_reg_val, hms_cols, pfas_cols, batch_size):
    num_hms = len(hms_cols)
    y_hms_train, y_hms_val = y_reg_train[:, :num_hms], y_reg_val[:, :num_hms]
    y_pfas_train, y_pfas_val = y_reg_train[:, num_hms:], y_reg_val[:, num_hms:]

    scaler_X = None
    if CONFIG["PREPROCESS_X"]:
        scaler_X = get_scaler(CONFIG["X_MODE"])
        X_train = scaler_X.fit_transform(X_train)
        X_val = scaler_X.transform(X_val)

    scaler_hms, scaler_pfas = None, None
    if CONFIG["PREPROCESS_Y"]:
        scaler_hms = get_scaler(CONFIG["HMS_Y_MODE"])
        y_hms_train = scaler_hms.fit_transform(y_hms_train)
        y_hms_val = scaler_hms.transform(y_hms_val)

        scaler_pfas = get_scaler(CONFIG["PFAS_Y_MODE"])
        y_pfas_train = scaler_pfas.fit_transform(y_pfas_train)
        y_pfas_val = scaler_pfas.transform(y_pfas_val)

    train_set = SpecDataset(X_train, y_cls_train, y_hms_train, y_pfas_train)
    val_set = SpecDataset(X_val, y_cls_val, y_hms_val, y_pfas_val)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, scaler_X, scaler_hms, scaler_pfas

def train_expert(model, train_loader, val_loader, epochs, lr, task, save_path=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_reg = nn.SmoothL1Loss(beta=0.1)
    criterion_phys = nn.MSELoss()
    ALPHA_PHYSICS = 0.1

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for batch_X, batch_y_cls, batch_y_hms, batch_y_pfas in train_loader:
            batch_X = batch_X.to(DEVICE)
            if task == "cls":
                batch_y = batch_y_cls.to(DEVICE)
            elif task == "reg_hms":
                batch_y = batch_y_hms.to(DEVICE)
            elif task == "reg_pfas":
                batch_y = batch_y_pfas.to(DEVICE)

            optimizer.zero_grad()
            if task == "cls":
                out = model(batch_X)
                loss = criterion_cls(out, batch_y)
            else:
                out, x_recon = model(batch_X)
                loss_data = criterion_reg(out, batch_y)
                loss_phys = criterion_phys(x_recon, batch_X.squeeze(-1))
                loss = loss_data + ALPHA_PHYSICS * loss_phys

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch_X.size(0)

        train_loss = total_train_loss / len(train_loader.dataset)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y_cls, batch_y_hms, batch_y_pfas in val_loader:
                batch_X = batch_X.to(DEVICE)
                if task == "cls":
                    batch_y = batch_y_cls.to(DEVICE)
                    out = model(batch_X)
                    val_loss_batch = criterion_cls(out, batch_y)
                else:
                    batch_y = batch_y_hms.to(DEVICE) if task == "reg_hms" else batch_y_pfas.to(DEVICE)
                    out, x_recon = model(batch_X)
                    loss_data = criterion_reg(out, batch_y)
                    loss_phys = criterion_phys(x_recon, batch_X.squeeze(-1))
                    val_loss_batch = loss_data + ALPHA_PHYSICS * loss_phys
                total_val_loss += val_loss_batch.item() * batch_X.size(0)

        val_loss = total_val_loss / len(val_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= CONFIG["PATIENCE"]:
            break

    if save_path and best_state is not None:
        torch.save(best_state, save_path)
        model.load_state_dict(best_state)

    return model, best_val_loss

def inverse_transform_target(preds, scaler, mode):
    if mode == "log":
        return np.expm1(preds)
    elif scaler is not None:
        return scaler.inverse_transform(preds)
    return preds

def grid_search_kfold(X_trainval, y_cls_trainval, y_reg_trainval, hms_cols, pfas_cols, task, model_class, out_dim):
    kf = KFold(n_splits=CONFIG["N_SPLITS"], shuffle=True, random_state=CONFIG["RANDOM_SEED"])
    param_grid = CONFIG["PARAM_GRID"]
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_score = float('inf')
    best_params = None
    print(f"Grid search started for task: {task}, total combinations: {len(param_combinations)}")

    for params in param_combinations:
        fold_val_losses = []
        print(f"  Testing params: {params}")

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_trainval)):
            X_train_fold, X_val_fold = X_trainval[train_idx], X_trainval[val_idx]
            y_cls_train_fold, y_cls_val_fold = y_cls_trainval[train_idx], y_cls_trainval[val_idx]
            y_reg_train_fold, y_reg_val_fold = y_reg_trainval[train_idx], y_reg_trainval[val_idx]

            train_loader, val_loader, _, _, _ = build_dataloaders(
                X_train_fold, X_val_fold,
                y_cls_train_fold, y_cls_val_fold,
                y_reg_train_fold, y_reg_val_fold,
                hms_cols, pfas_cols,
                params["BATCH_SIZE"]
            )

            wave_num = X_train_fold.shape[1]
            model_task = task.split("_")[0] if "reg" in task else task
            model = model_class(
                wave_num=wave_num,
                out_dim=out_dim,
                task=model_task,
                d_model=params["D_MODEL"],
                num_layers=params["NUM_LAYERS"]
            ).to(DEVICE)

            _, val_loss = train_expert(
                model, train_loader, val_loader,
                epochs=CONFIG["EPOCHS"],
                lr=params["LR"],
                task=task
            )
            fold_val_losses.append(val_loss)
            print(f"    Fold {fold+1}/{CONFIG['N_SPLITS']} val loss: {val_loss:.5f}")

        avg_val_loss = np.mean(fold_val_losses)
        print(f"  Average val loss: {avg_val_loss:.5f}")

        if avg_val_loss < best_score:
            best_score = avg_val_loss
            best_params = params

    print(f"Best params for {task}: {best_params}, best avg val loss: {best_score:.5f}")
    return best_params

def evaluate_pipeline(model_cls, model_hms, model_pfas, test_loader, hms_cols, pfas_cols,
                      scaler_hms, scaler_pfas, y_cls_test, y_reg_test_original, save_dir):
    model_cls.eval()
    model_hms.eval()
    model_pfas.eval()
    all_cols = hms_cols + pfas_cols
    num_hms, num_pfas = len(hms_cols), len(pfas_cols)
    num_total = num_hms + num_pfas
    y_cls_pred_list = []
    y_reg_pred_processed_list = []

    with torch.no_grad():
        for batch_X, _, _, _ in test_loader:
            batch_X = batch_X.to(DEVICE)
            out_cls = model_cls(batch_X)
            pred_cls_binary = (torch.sigmoid(out_cls) > 0.5).int()
            y_cls_pred_list.extend(pred_cls_binary.cpu().numpy())

            batch_reg_pred = torch.zeros((batch_X.size(0), num_total)).to(DEVICE)
            hms_mask = pred_cls_binary[:, :num_hms].sum(dim=1) > 0
            if hms_mask.any():
                pred_hms, _ = model_hms(batch_X[hms_mask])
                batch_reg_pred[hms_mask, :num_hms] = pred_hms
            pfas_mask = pred_cls_binary[:, num_hms:].sum(dim=1) > 0
            if pfas_mask.any():
                pred_pfas, _ = model_pfas(batch_X[pfas_mask])
                batch_reg_pred[pfas_mask, num_hms:] = pred_pfas
            y_reg_pred_processed_list.extend(batch_reg_pred.cpu().numpy())

    y_cls_pred_final = np.array(y_cls_pred_list)
    y_reg_pred_processed = np.array(y_reg_pred_processed_list)

    y_reg_pred_hms_orig = inverse_transform_target(y_reg_pred_processed[:, :num_hms], scaler_hms, CONFIG["HMS_Y_MODE"])
    y_reg_pred_pfas_orig = inverse_transform_target(y_reg_pred_processed[:, num_hms:], scaler_pfas, CONFIG["PFAS_Y_MODE"])
    y_reg_pred_original = np.hstack([y_reg_pred_hms_orig, y_reg_pred_pfas_orig])
    y_reg_pred_original = np.maximum(y_reg_pred_original, 0.0)
    y_reg_pred_filtered = y_reg_pred_original * y_cls_pred_final

    pred_result_df = pd.DataFrame()
    rel_error_df = pd.DataFrame()
    r2_scores, rmse_scores, mae_scores = [], [], []

    print("Classification Evaluation:")
    print(f"Full match accuracy: {accuracy_score(y_cls_test, y_cls_pred_final):.5f}")
    for i, name in enumerate(all_cols):
        acc = accuracy_score(y_cls_test[:, i], y_cls_pred_final[:, i])
        print(f"  [{name:6s}]: Acc = {acc:.5f}")

    print("Regression Evaluation:")
    for i, name in enumerate(all_cols):
        y_true_all = y_reg_test_original[:, i]
        y_pred_all = y_reg_pred_filtered[:, i]
        mask = y_true_all > CONFIG["EPSILON"]
        y_true, y_pred = y_true_all[mask], y_pred_all[mask]
        re = np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-6) * 100
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2_scores.append(r2)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        print(f"  [{name:6s}]: R²={r2:.5f} | RMSE={rmse:.5f} | MAE={mae:.5f}")
        pred_result_df[f"True_{name}"] = pd.Series(y_true)
        pred_result_df[f"Pre_{name}"] = pd.Series(y_pred)
        rel_error_df[f"{name}_RE(%)"] = pd.Series(re)

    pred_csv_path = os.path.join(save_dir, "test_predictions.csv")
    pred_result_df.to_csv(pred_csv_path, index_label="ID", encoding="utf-8-sig")
    re_xlsx_path = os.path.join(save_dir, "test_relative_error.xlsx")
    rel_error_df.to_excel(re_xlsx_path, index_label="Sample ID")

    fig_box = plt.figure(figsize=(10, 6))
    sns.boxplot(data=rel_error_df, palette="Set2", showfliers=True)
    plt.title("Relative Error of Each Component", fontsize=15, fontweight='bold')
    plt.ylabel("Relative Error (%)", fontsize=12)
    plt.yscale('symlog')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_box_path = os.path.join(save_dir, "relative_error_boxplot.png")
    fig_box.savefig(save_box_path, dpi=300, bbox_inches="tight")
    plt.close(fig_box)

    n_cols = 2
    n_rows = (num_total + 1) // 2
    fig, gs = plt.figure(figsize=(n_cols * 6, n_rows * 5)), gridspec.GridSpec(n_rows, n_cols)
    fig.suptitle("Concentration Prediction Scatter Plot", fontsize=16, fontweight='bold', y=1.02)
    for i, name in enumerate(all_cols):
        ax = fig.add_subplot(gs[i])
        y_true, y_pred = y_reg_test_original[:, i], y_reg_pred_filtered[:, i]
        ax.scatter(y_true, y_pred, alpha=0.6, s=25, color=sns.color_palette("Set1")[i % 7], edgecolors='k', linewidth=0.5)
        min_v, max_v = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=2, label="1:1 line")
        metrics_text = f"$R^2$: {r2_scores[i]:.4f}\nRMSE: {rmse_scores[i]:.4f}\nMAE: {mae_scores[i]:.4f}"
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
        ax.set_title(f"{name}", fontsize=14)
        ax.set_xlabel("True", fontsize=11)
        ax.set_ylabel("Pred", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "regression_scatter.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    n_cols_cm = 3
    n_rows_cm = (num_total + 2) // 3
    fig_cm = plt.figure(figsize=(n_cols_cm * 4, n_rows_cm * 3.5))
    fig_cm.suptitle("Confusion Matrix", fontsize=16, fontweight='bold', y=1.05)
    gs_cm = gridspec.GridSpec(n_rows_cm, n_cols_cm, figure=fig_cm)
    for i, name in enumerate(all_cols):
        ax = fig_cm.add_subplot(gs_cm[i])
        cm = confusion_matrix(y_cls_test[:, i], y_cls_pred_final[:, i])
        if cm.shape != (2, 2):
            new_cm = np.zeros((2, 2), dtype=int)
            unique_true = np.unique(y_cls_test[:, i])
            for idx, val in enumerate(unique_true):
                if val == 0:
                    new_cm[0, 0] = cm[idx, 0] if cm.shape == (1, 1) else cm[idx, 0]
                if val == 1:
                    new_cm[1, 1] = cm[0, 0] if (cm.shape == (1, 1) and unique_true[0] == 1) else (cm[idx, 1] if cm.shape[1] > 1 else 0)
            cm = new_cm
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False, annot_kws={"size": 13})
        ax.set_title(f"{name}", fontsize=13)
        ax.set_xlabel("Predicted (0:absent, 1:present)", fontsize=11)
        ax.set_ylabel("True (0:absent, 1:present)", fontsize=11)
        ax.set_xticks([0.5, 1.5])
        ax.set_yticks([0.5, 1.5])
        ax.set_xticklabels(['0', '1'])
        ax.set_yticklabels(['0', '1'])
    fig_cm.tight_layout()
    save_cm_path = os.path.join(save_dir, "confusion_matrix.png")
    fig_cm.savefig(save_cm_path, dpi=300, bbox_inches="tight")
    plt.close(fig_cm)

    return y_cls_pred_final, y_reg_pred_filtered

class GradCAM1D(nn.Module):
    def __init__(self, model, target_layer):
        super().__init__()
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def forward(self, x, target_idx=None):
        if self.model.task == "reg":
            pred, _ = self.model(x)
        else:
            pred = self.model(x)
        self.model.zero_grad()
        if target_idx is not None:
            loss = pred[:, target_idx].sum()
        else:
            loss = pred.sum()
        loss.backward()
        gradients = self.gradients
        activations = self.activations
        weights = torch.mean(gradients, dim=2, keepdim=True)
        cam = torch.sum(activations * weights, dim=1)
        cam = F.relu(cam)
        cam = (cam - cam.min(dim=1, keepdim=True)[0]) / (cam.max(dim=1, keepdim=True)[0] + 1e-8)
        return cam, pred

def plot_spectral_importance(model, test_loader, target_names, model_name, save_dir, device):
    model.eval()
    if isinstance(model, SpectralExpertModel):
        target_layer = model.cnn[9]
    elif isinstance(model, LightweightSpectralExpertModel):
        target_layer = model.cnn[3]
    else:
        raise ValueError("Unsupported model type")

    grad_cam = GradCAM1D(model, target_layer)
    batch_X, _, _, _ = next(iter(test_loader))
    batch_X = batch_X.to(device)
    wave_num = batch_X.shape[1]
    wavelengths = np.linspace(190, 780, wave_num)

    all_cam_data = pd.DataFrame({"Wavelength": wavelengths})
    fig, axes = plt.subplots(len(target_names), 1, figsize=(12, 2.5 * len(target_names)))
    fig.suptitle(f"{model_name} - Spectral Feature Importance (Grad-CAM)", fontsize=14, fontweight="bold")

    for idx, name in enumerate(target_names):
        cam, _ = grad_cam(batch_X, target_idx=idx)
        cam_mean = cam.mean(dim=0).detach().cpu().numpy()
        all_cam_data[f"{name}_importance"] = cam_mean
        ax = axes[idx] if len(target_names) > 1 else axes
        ax.plot(wavelengths, cam_mean, color="#e74c3c", linewidth=2)
        ax.fill_between(wavelengths, 0, cam_mean, alpha=0.3, color="#e74c3c")
        ax.set_title(f"Component: {name}", fontsize=12)
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Feature Importance")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    save_fig_path = os.path.join(save_dir, f"{model_name}_spectral_importance.png")
    save_data_path = os.path.join(save_dir, f"{model_name}_GradCAM_weights.csv")
    fig.savefig(save_fig_path, dpi=300, bbox_inches="tight")
    all_cam_data.to_csv(save_data_path, index=False, encoding="utf-8-sig")
    plt.close(fig)
    print(f"{model_name} spectral importance analysis completed")
    print(f"  Figure: {save_fig_path}")
    print(f"  Data: {save_data_path}")

def plot_attention_heatmap(model, test_loader, model_name, save_dir, device):
    model.eval()
    batch_X, _, _, _ = next(iter(test_loader))
    batch_X = batch_X.to(device)
    if model.task == "reg":
        _, _ = model(batch_X)
    else:
        _ = model(batch_X)

    attn = model.transformer.attn_weights[-1][5].cpu().detach().numpy()
    wave_num = attn.shape[0]
    attn_matrix = model.transformer.attn_weights[-1][0].cpu().detach().numpy()
    wave_num = attn_matrix.shape[0]
    wavelengths = np.linspace(190, 780, wave_num)
    wavelengths_rounded = np.round(wavelengths, 1)
    attn_df = pd.DataFrame(
        attn_matrix,
        index=wavelengths_rounded,
        columns=wavelengths_rounded
    )
    save_csv_path = os.path.join(save_dir, f"{model_name}_attention_matrix.csv")
    attn_df.to_csv(save_csv_path, index_label="Key_Query_nm", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 8))
    wavelengths = np.linspace(190, 780, wave_num)
    tick_indices = np.linspace(0, wave_num - 1, 6, dtype=int)
    tick_labels = [f"{wavelengths[i]:.0f}" for i in tick_indices]
    cbar_kws = {'shrink': 1.0}
    heatmap_ax = sns.heatmap(attn, cmap="viridis", ax=ax, xticklabels=False, yticklabels=False, cbar_kws=cbar_kws)
    cbar = heatmap_ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    ax.set_xticks(tick_indices + 0.5)
    ax.set_xticklabels(tick_labels, rotation=0, fontsize=18)
    ax.set_yticks(tick_indices + 0.5)
    ax.set_yticklabels(tick_labels, rotation=0, fontsize=18)
    ax.set_title(f"{model_name} Attention Heat Map", fontsize=22)
    ax.set_xlabel("Query Wavelength (nm)", fontsize=24)
    ax.set_ylabel("Key Wavelength (nm)", fontsize=24)
    plt.tight_layout()
    save_fig_path = os.path.join(save_dir, f"{model_name}_attention_heatmap.png")
    fig.savefig(save_fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"{model_name} attention weight analysis completed")
    print(f"  Figure: {save_fig_path}")

def analyze_pinn_physical_constraint(model_hms, model_pfas, test_loader, scaler_hms, scaler_pfas, save_dir, device):
    model_hms.eval()
    model_pfas.eval()
    recon_loss_list = []
    pred_error_list = []
    sample_idx_list = []

    with torch.no_grad():
        for batch_idx, (batch_X, _, batch_y_hms, batch_y_pfas) in enumerate(test_loader):
            batch_X = batch_X.to(device)
            batch_y_hms = batch_y_hms.to(device)
            batch_y_pfas = batch_y_pfas.to(device)
            pred_hms, x_recon_hms = model_hms(batch_X)
            pred_pfas, x_recon_pfas = model_pfas(batch_X)

            recon_loss_hms = F.mse_loss(x_recon_hms, batch_X.squeeze(-1), reduction='none').mean(dim=1).cpu().numpy()
            recon_loss_pfas = F.mse_loss(x_recon_pfas, batch_X.squeeze(-1), reduction='none').mean(dim=1).cpu().numpy()
            recon_loss_batch = (recon_loss_hms + recon_loss_pfas) / 2
            recon_loss_list.extend(recon_loss_batch)

            pred_hms_orig = inverse_transform_target(pred_hms.cpu().numpy(), scaler_hms, CONFIG["HMS_Y_MODE"])
            true_hms_orig = inverse_transform_target(batch_y_hms.cpu().numpy(), scaler_hms, CONFIG["HMS_Y_MODE"])
            pred_pfas_orig = inverse_transform_target(pred_pfas.cpu().numpy(), scaler_pfas, CONFIG["PFAS_Y_MODE"])
            true_pfas_orig = inverse_transform_target(batch_y_pfas.cpu().numpy(), scaler_pfas, CONFIG["PFAS_Y_MODE"])

            mae_hms = np.mean(np.abs(pred_hms_orig - true_hms_orig), axis=1)
            mae_pfas = np.mean(np.abs(pred_pfas_orig - true_pfas_orig), axis=1)
            pred_error_batch = (mae_hms + mae_pfas) / 2
            pred_error_list.extend(pred_error_batch)

            start_idx = batch_idx * CONFIG["BATCH_SIZE"]
            sample_idx_list.extend(range(start_idx, start_idx + len(batch_X)))

    pinn_data = pd.DataFrame({
        "Sample ID": sample_idx_list,
        "Physical Reconstruction Loss (MSE)": recon_loss_list,
        "Prediction MAE": pred_error_list
    })
    save_data_path = os.path.join(save_dir, "PINN_physical_constraint_analysis.csv")
    pinn_data.to_csv(save_data_path, index=False, encoding="utf-8-sig")

    corr = np.corrcoef(recon_loss_list, pred_error_list)[0, 1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(recon_loss_list, pred_error_list, alpha=0.6, color="#2ecc71", edgecolors="k", s=30)
    ax.set_title(f"PINN Reconstruction Loss vs Prediction Error (corr={corr:.4f})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Physical Reconstruction Loss (MSE)")
    ax.set_ylabel("Prediction MAE")
    ax.grid(alpha=0.3)
    save_fig1_path = os.path.join(save_dir, "PINN_recon_loss_pred_error_correlation.png")
    fig.savefig(save_fig1_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    batch_X, _, _, _ = next(iter(test_loader))
    batch_X = batch_X.to(device)
    _, x_recon_hms = model_hms(batch_X)
    _, x_recon_pfas = model_pfas(batch_X)
    original_spec = batch_X[0].squeeze(-1).cpu().numpy()
    recon_spec_hms = x_recon_hms[0].cpu().detach().numpy()
    recon_spec_pfas = x_recon_pfas[0].cpu().detach().numpy()
    wave_num = original_spec.shape[0]
    wavelengths = np.linspace(190, 780, wave_num)

    spectra_data = pd.DataFrame({
        "Wavelength (nm)": wavelengths,
        "Original Spectrum": original_spec,
        "HMS Reconstructed Spectrum": recon_spec_hms,
        "PFAS Reconstructed Spectrum": recon_spec_pfas
    })
    save_spectra_path = os.path.join(save_dir, "PINN_spectrum_reconstruction_comparison.csv")
    spectra_data.to_csv(save_spectra_path, index=False, encoding="utf-8-sig")

    ax2.plot(wavelengths, original_spec, label="Original Spectrum", color="black", linewidth=2)
    ax2.plot(wavelengths, recon_spec_hms, label="HMS Reconstructed Spectrum", color="#e74c3c", linestyle="--", alpha=0.8)
    ax2.plot(wavelengths, recon_spec_pfas, label="PFAS Reconstructed Spectrum", color="#3498db", linestyle="--", alpha=0.8)
    ax2.set_title("Original vs PINN Reconstructed Spectrum (First Test Sample)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Spectral Intensity")
    ax2.legend()
    ax2.grid(alpha=0.3)
    save_fig2_path = os.path.join(save_dir, "PINN_spectrum_reconstruction_comparison.png")
    fig2.savefig(save_fig2_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    print("PINN physical constraint analysis completed")
    print(f"  Correlation figure: {save_fig1_path}")
    print(f"  Spectrum comparison figure: {save_fig2_path}")
    print(f"  Analysis data: {save_data_path}")

if __name__ == "__main__":
    try:
        X_trainval, X_test, y_cls_trainval, y_cls_test, y_reg_trainval, y_reg_test, hms_cols, pfas_cols = load_raw_data()
        wave_num = X_trainval.shape[1]
        num_hms = len(hms_cols)
        num_pfas = len(pfas_cols)

        print("=" * 60)
        print("Starting grid search with K-fold cross validation")
        print("=" * 60)

        best_params_cls = grid_search_kfold(
            X_trainval, y_cls_trainval, y_reg_trainval,
            hms_cols, pfas_cols,
            task="cls",
            model_class=SpectralExpertModel,
            out_dim=num_hms + num_pfas
        )

        best_params_hms = grid_search_kfold(
            X_trainval, y_cls_trainval, y_reg_trainval,
            hms_cols, pfas_cols,
            task="reg_hms",
            model_class=SpectralExpertModel,
            out_dim=num_hms
        )

        best_params_pfas = grid_search_kfold(
            X_trainval, y_cls_trainval, y_reg_trainval,
            hms_cols, pfas_cols,
            task="reg_pfas",
            model_class=LightweightSpectralExpertModel,
            out_dim=num_pfas
        )

        print("\n" + "=" * 60)
        print("Training final models with best parameters")
        print("=" * 60)

        train_loader_full, test_loader, scaler_X_full, scaler_hms_full, scaler_pfas_full = build_dataloaders(
            X_trainval, X_test,
            y_cls_trainval, y_cls_test,
            y_reg_trainval, y_reg_test,
            hms_cols, pfas_cols,
            best_params_cls["BATCH_SIZE"]
        )
        joblib.dump(scaler_X_full, os.path.join(CONFIG["SAVE_DIR"], 'scaler_X.pkl'))
        joblib.dump(scaler_hms_full, os.path.join(CONFIG["SAVE_DIR"], 'scaler_hms.pkl'))
        joblib.dump(scaler_pfas_full, os.path.join(CONFIG["SAVE_DIR"], 'scaler_pfas.pkl'))

        model_cls = SpectralExpertModel(
            wave_num=wave_num,
            out_dim=num_hms + num_pfas,
            task="cls",
            d_model=best_params_cls["D_MODEL"],
            num_layers=best_params_cls["NUM_LAYERS"]
        ).to(DEVICE)
        model_cls, _ = train_expert(
            model_cls, train_loader_full, test_loader,
            epochs=CONFIG["EPOCHS"],
            lr=best_params_cls["LR"],
            task="cls",
            save_path=os.path.join(CONFIG["SAVE_DIR"], "Model_Cls.pth")
        )

        model_hms = SpectralExpertModel(
            wave_num=wave_num,
            out_dim=num_hms,
            task="reg",
            d_model=best_params_hms["D_MODEL"],
            num_layers=best_params_hms["NUM_LAYERS"]
        ).to(DEVICE)
        model_hms, _ = train_expert(
            model_hms, train_loader_full, test_loader,
            epochs=CONFIG["EPOCHS"],
            lr=best_params_hms["LR"],
            task="reg_hms",
            save_path=os.path.join(CONFIG["SAVE_DIR"], "Model_HMS.pth")
        )

        model_pfas = LightweightSpectralExpertModel(
            wave_num=wave_num,
            out_dim=num_pfas,
            task="reg",
            d_model=best_params_pfas["D_MODEL"],
            num_layers=best_params_pfas["NUM_LAYERS"]
        ).to(DEVICE)
        model_pfas, _ = train_expert(
            model_pfas, train_loader_full, test_loader,
            epochs=CONFIG["EPOCHS"],
            lr=best_params_pfas["LR"],
            task="reg_pfas",
            save_path=os.path.join(CONFIG["SAVE_DIR"], "Model_PFAs.pth")
        )

        y_reg_test_original = y_reg_test.copy()
        y_cls_pred_final, y_reg_pred_filtered = evaluate_pipeline(
            model_cls, model_hms, model_pfas, test_loader,
            hms_cols, pfas_cols, scaler_hms_full, scaler_pfas_full,
            y_cls_test, y_reg_test_original, CONFIG["SAVE_DIR"]
        )

        print("\n" + "=" * 60)
        print("Starting interpretability analysis")
        print("=" * 60)

        plot_spectral_importance(model_hms, test_loader, hms_cols, "HMS Model", CONFIG["INTERPRET_DIR"], DEVICE)
        plot_spectral_importance(model_pfas, test_loader, pfas_cols, "PFAS Model", CONFIG["INTERPRET_DIR"], DEVICE)
        plot_attention_heatmap(model_hms, test_loader, "HMS Model", CONFIG["INTERPRET_DIR"], DEVICE)
        plot_attention_heatmap(model_pfas, test_loader, "PFAS Model", CONFIG["INTERPRET_DIR"], DEVICE)
        analyze_pinn_physical_constraint(model_hms, model_pfas, test_loader, scaler_hms_full, scaler_pfas_full,
                                         CONFIG["INTERPRET_DIR"], DEVICE)

        print("\nAll tasks completed!")
        print(f"Results saved to: {CONFIG['SAVE_DIR']}")
        print(f"Interpretability results saved to: {CONFIG['INTERPRET_DIR']}")

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()