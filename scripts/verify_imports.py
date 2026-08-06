"""快速导入验证"""
from smartalpha.config import DATA_DIR, PROCESSED_DIR, RESULTS_DIR, MODEL_SAVE_DIR, RL_TOTAL_TIMESTEPS
from smartalpha.factor import neutralize, filter_by_ic, remove_correlated, cross_sectional_corr_filter, select_factors, build_limit_mask, apply_mask, industry_neutralize, market_cap_neutralize
from smartalpha.model.ensemble import EnsembleModel
from smartalpha.model.auto_retrain import RetrainTrigger
from smartalpha.model.monitor import ModelMonitor

print("Config imports: OK")
print(f"  PROJECT_ROOT: {DATA_DIR.parent}")
print(f"  RL_TOTAL_TIMESTEPS: {RL_TOTAL_TIMESTEPS}")
print(f"  MODEL_SAVE_DIR: {MODEL_SAVE_DIR}")
print("Factor imports: OK")
print(f"  __all__ exports: {len([neutralize, filter_by_ic, remove_correlated, cross_sectional_corr_filter, select_factors, build_limit_mask, apply_mask, industry_neutralize, market_cap_neutralize])}")
print("Model imports: OK")
print("All imports passed!")
