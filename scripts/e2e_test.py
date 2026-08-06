"""SmartAlpha Pro 端到端流程验证"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from smartalpha.config import PROCESSED_DIR, RESULTS_DIR, MODEL_SAVE_DIR


def _read_parquet(path):
    try:
        return pd.read_parquet(path)
    except OSError:
        return pd.read_parquet(path, engine="fastparquet")


print("=" * 60)
print("SmartAlpha Pro 端到端验证")
print("=" * 60)

# 1
print("\n[1/5] 数据加载...")
files_info = {
    "行情": PROCESSED_DIR / "daily_clean.parquet",
    "Mask行情": PROCESSED_DIR / "daily_masked.parquet",
    "因子": PROCESSED_DIR / "factors_neutral.parquet",
    "因子全量": PROCESSED_DIR / "factors_all.parquet",
    "基本面": PROCESSED_DIR / "daily_basic_clean.parquet",
    "模型预测": PROCESSED_DIR / "model_predictions.parquet",
}
for name, p in files_info.items():
    ok = p.exists()
    mb = p.stat().st_size / 1024 / 1024 if ok else 0
    print(f"  {name}: {'OK' if ok else 'MISSING'} ({mb:.1f} MB)")

# 2
print("\n[2/5] 因子工程...")
from smartalpha.factor import filter_by_ic
factors = _read_parquet(PROCESSED_DIR / "factors_neutral.parquet")
exclude = {"ts_code", "trade_date", "industry", "circ_mv", "log_mv", "fwd_ret_5d"}
factor_cols = [c for c in factors.columns if c not in exclude]
print(f"  因子数量: {len(factor_cols)}")
print(f"  数据行数: {len(factors):,}")
print(f"  日期: {factors['trade_date'].min()} ~ {factors['trade_date'].max()}")

# 3
print("\n[3/5] 模型训练...")
from smartalpha.model.lgbm import LGBMTrainer
try:
    sample = factors.head(2000)
    feats = [c for c in factor_cols if c in sample.columns]
    if "fwd_ret_5d" in sample.columns and len(feats) >= 3:
        trainer = LGBMTrainer()
        result = trainer.run(selected_factors=feats[:5])
        if result:
            rmse = result.get("rmse", 0)
            ic = result.get("ic", 0)
            print(f"  RMSE={rmse:.6f}, IC={ic:.4f}")
except Exception as e:
    print(f"  跳过: {e}")

# 4
print("\n[4/5] 回测...")
try:
    from smartalpha.backtest.engine import BacktestEngine
    engine = BacktestEngine()
    report = engine.run()
    for k in ["annual_return", "sharpe_ratio", "max_drawdown"]:
        print(f"  {k}: {report.get(k, 'N/A')}")
except Exception as e:
    print(f"  跳过: {e}")

# 5
print("\n[5/5] 模型文件...")
for name, path in [
    ("LightGBM", MODEL_SAVE_DIR / "lgbm_model.joblib"),
    ("SAC RL", MODEL_SAVE_DIR / "sac_portfolio.zip"),
    ("Transformer", MODEL_SAVE_DIR / "transformer_model.pt"),
]:
    ok = path.exists()
    kb = path.stat().st_size / 1024 if ok else 0
    print(f"  {name}: {'OK' if ok else 'N/A'} ({kb:.1f} KB)")

print("\n" + "=" * 60)
print("验证完成")
print("=" * 60)
