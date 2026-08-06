"""Transformer时序模型 - PyTorch实现，作为LightGBM的备选"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from smartalpha.config import MODEL_SAVE_DIR as MODEL_DIR

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch未安装，Transformer模型不可用")


class PositionalEncoding(nn.Module if HAS_TORCH else object):
    def __init__(self, d_model, max_len=5000):
        if HAS_TORCH:
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        if HAS_TORCH:
            return x + self.pe[:, :x.size(1), :]
        return x


class TransformerModel(nn.Module if HAS_TORCH else object):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        if HAS_TORCH:
            super().__init__()
            self.input_proj = nn.Linear(input_dim, d_model)
            self.pos_encoder = PositionalEncoding(d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_layer = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1))

    def forward(self, x):
        if HAS_TORCH:
            x = self.input_proj(x)
            x = self.pos_encoder(x)
            # 因果mask: 防止当前时间步看到未来信息
            seq_len = x.size(1)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
            x = self.transformer_encoder(x, mask=causal_mask)
            x = x[:, -1, :]
            x = self.output_layer(x)
            return x.squeeze(-1)
        return None


class TransformerTrainer:
    """Transformer时序模型训练器"""

    def __init__(self, input_dim, seq_len=20):
        if not HAS_TORCH:
            raise ImportError("PyTorch未安装，无法使用Transformer")
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        logger.info(f"Transformer将使用: {self.device}")

    def prepare_sequences(self, X, y, ts_codes=None):
        """将截面数据转为时序序列，按ts_code分组防止跨股票拼接"""
        X_np = X.values if hasattr(X, 'values') else np.array(X)
        y_np = y.values if hasattr(y, 'values') else np.array(y)

        X_seq, y_seq = [], []
        if ts_codes is not None:
            # 按股票分组切序列
            unique_codes = np.unique(ts_codes)
            for code in unique_codes:
                idx = np.where(ts_codes == code)[0]
                if len(idx) < self.seq_len:
                    continue
                for i in range(0, len(idx) - self.seq_len + 1, max(1, self.seq_len // 2)):
                    window_idx = idx[i:i+self.seq_len]
                    X_seq.append(X_np[window_idx])
                    y_seq.append(y_np[idx[i+self.seq_len-1]])
        else:
            # 无分组信息时按行顺序切（调用方需确保单股票）
            n = len(X_np) - self.seq_len + 1
            for i in range(0, n, max(1, self.seq_len // 2)):
                X_seq.append(X_np[i:i+self.seq_len])
                y_seq.append(y_np[i+self.seq_len-1])

        if not X_seq:
            return torch.FloatTensor([]).to(self.device), torch.FloatTensor([]).to(self.device)

        X_tensor = torch.FloatTensor(np.array(X_seq)).to(self.device)
        y_tensor = torch.FloatTensor(np.array(y_seq)).to(self.device)
        return X_tensor, y_tensor

    def train(self, X, y, ts_codes=None, dates=None, epochs=10, lr=1e-3, batch_size=256):
        """训练Transformer：支持mini-batch + 验证集早停"""
        X_seq, y_seq = self.prepare_sequences(X, y, ts_codes=ts_codes)
        if len(X_seq) == 0:
            logger.error("无法构建时序序列，数据不足")
            return {"ic": 0}

        # 时序切分：80%训练，20%验证
        n_total = len(X_seq)
        split = int(n_total * 0.8)
        X_train, X_val = X_seq[:split], X_seq[split:]
        y_train, y_val = y_seq[:split], y_seq[split:]

        self.model = TransformerModel(self.input_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        patience, patience_counter = 5, 0

        for epoch in range(epochs):
            self.model.train()
            # Mini-batch训练
            indices = torch.randperm(len(X_train))
            for i in range(0, len(X_train), batch_size):
                batch_idx = indices[i:i+batch_size]
                pred = self.model(X_train[batch_idx])
                loss = criterion(pred, y_train[batch_idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 验证集评估
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val)
                val_loss = criterion(val_pred, y_val).item()
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss:.6f}")

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # 最终评估
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X_seq).cpu().numpy()
            y_true = y_seq.cpu().numpy()
            ic = np.corrcoef(pred, y_true)[0, 1] if len(pred) > 2 else 0
            logger.info(f"Transformer IC: {ic:.4f}")

        self._save()
        return {"ic": ic}

    def predict(self, X, ts_codes=None):
        if self.model is None:
            return None
        X_seq, _ = self.prepare_sequences(X, np.zeros(len(X)), ts_codes=ts_codes)
        if len(X_seq) == 0:
            return np.array([])
        self.model.eval()
        with torch.no_grad():
            return self.model(X_seq).cpu().numpy()

    def _save(self):
        path = MODEL_DIR / "transformer_model.pt"
        torch.save(self.model.state_dict(), path)
        # 同时保存超参数，加载时需要
        import json
        hyperparams = {"input_dim": self.input_dim, "seq_len": self.seq_len}
        with open(MODEL_DIR / "transformer_hyperparams.json", "w") as f:
            json.dump(hyperparams, f)
        logger.info(f"Transformer保存: {path}")
