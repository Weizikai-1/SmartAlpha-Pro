from smartalpha.model.lgbm import LightGBMPredictor, LGBMTrainer

assert LGBMTrainer is LightGBMPredictor

p = LGBMTrainer()
assert hasattr(p, "train"), "train missing"
assert hasattr(p, "run"), "run missing"
assert hasattr(p, "prepare_data"), "prepare_data missing"
assert hasattr(p, "model"), "model property missing"

# Test prepare_data
X, y, dates, cols = p.prepare_data(selected_factors=None)
assert X is not None and len(X) > 0, "X empty"
assert y is not None and len(y) > 0, "y empty"
assert len(cols) > 0, "no factor columns"
print(f"prepare_data OK: X shape={X.shape}, y shape={y.shape}, {len(cols)} factors")

# Test run (quick, small sample)
result = p.run(selected_factors=cols[:5])
assert result is not None, "run returned None"
assert "rmse" in result, "rmse missing"
assert "ic" in result, "ic missing"
print(f"run OK: rmse={result['rmse']:.6f}, ic={result['ic']:.4f}")

# Test train
p2 = LGBMTrainer()
p2.train(X.head(500), y.head(500), dates.head(500), test_ratio=0.3)
assert p2.model is not None, "train did not set model"
assert p2.is_trained, "model not trained"
print(f"train OK: model trained, best_iteration={p2.model.best_iteration}")

print("\nAll checks passed!")
