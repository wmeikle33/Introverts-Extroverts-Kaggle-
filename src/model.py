lgb = LGBMClassifier(
    n_estimators=3000, learning_rate=0.02,
    max_depth=-1, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42
)
