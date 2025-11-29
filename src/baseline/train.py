import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from . import config, constants
from .features import add_aggregate_features, handle_missing_values
from .temporal_split import get_split_date_from_ratio, temporal_split_by_date

def prepare_training_data(train_split: pd.DataFrame, use_unread_for_features: bool = True):
    if use_unread_for_features:
        features_train = train_split.copy()
        target_mask = features_train[constants.COL_HAS_READ] == 1
        
        X_train = features_train[target_mask].copy()
        y_train = X_train[config.TARGET]
        
        agg_reference = train_split.copy()
    else:
        X_train = train_split[train_split[constants.COL_HAS_READ] == 1].copy()
        y_train = X_train[config.TARGET]
        agg_reference = X_train.copy()
    
    return X_train, y_train, agg_reference

def temporal_cross_validate(df: pd.DataFrame, n_splits: int = 3):
    df = df.copy()
    df[constants.COL_TIMESTAMP] = pd.to_datetime(df[constants.COL_TIMESTAMP])
    
    dates = df[constants.COL_TIMESTAMP].sort_values().unique()
    split_dates = [dates[int(len(dates) * i / n_splits)] for i in range(1, n_splits)]
    
    scores = []
    for i, split_date in enumerate(split_dates):
        train_mask, val_mask = temporal_split_by_date(df, split_date, constants.COL_TIMESTAMP)
        train_split = df[train_mask].copy()
        val_split = df[val_mask].copy()
        
        X_train, y_train, agg_reference = prepare_training_data(
            train_split, use_unread_for_features=config.USE_READING_BEHAVIOR_FEATURES
        )
        
        if len(X_train) < 100 or len(val_split) < 50:
            continue
            
        X_train_with_agg = add_aggregate_features(X_train.copy(), agg_reference)
        val_split_with_agg = add_aggregate_features(val_split.copy(), agg_reference)
        
        X_train_final = handle_missing_values(X_train_with_agg, agg_reference)
        val_split_final = handle_missing_values(val_split_with_agg, agg_reference)
        
        exclude_cols = [
            constants.COL_SOURCE, config.TARGET, constants.COL_PREDICTION, 
            constants.COL_TIMESTAMP, constants.COL_HAS_READ
        ]
        features = [col for col in X_train_final.columns if col not in exclude_cols]
        
        non_feature_object_cols = X_train_final[features].select_dtypes(include=["object"]).columns.tolist()
        features = [f for f in features if f not in non_feature_object_cols]
        
        X_train_features = X_train_final[features]
        X_val_features = val_split_final[features]
        y_val = val_split_final[config.TARGET]
        
        model = lgb.LGBMRegressor(**config.LGB_PARAMS)
        fit_params = config.LGB_FIT_PARAMS.copy()
        fit_params["callbacks"] = [lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS, verbose=False)]
        
        model.fit(
            X_train_features,
            y_train,
            eval_set=[(X_val_features, y_val)],
            eval_metric=fit_params["eval_metric"],
            callbacks=fit_params["callbacks"],
            verbose=False
        )
        
        val_preds = model.predict(X_val_features)
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        mae = mean_absolute_error(y_val, val_preds)
        
        range_width = 10.0
        mae_norm = mae / range_width
        rmse_norm = rmse / range_width
        score = 1 - (0.5 * rmse_norm + 0.5 * mae_norm)
        
        scores.append(score)
    
    return scores

def train() -> None:
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME

    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_path}. "
            "Please run 'poetry run python -m src.baseline.prepare_data' first."
        )

    featured_df = pd.read_parquet(processed_path, engine="pyarrow")

    train_set = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    if constants.COL_TIMESTAMP not in train_set.columns:
        raise ValueError(
            f"Timestamp column '{constants.COL_TIMESTAMP}' not found in train set. "
            "Make sure data was prepared with timestamp preserved."
        )

    if not pd.api.types.is_datetime64_any_dtype(train_set[constants.COL_TIMESTAMP]):
        train_set[constants.COL_TIMESTAMP] = pd.to_datetime(train_set[constants.COL_TIMESTAMP])

    cv_scores = temporal_cross_validate(train_set, n_splits=3)

    split_date = get_split_date_from_ratio(train_set, config.TEMPORAL_SPLIT_RATIO, constants.COL_TIMESTAMP)

    train_mask, val_mask = temporal_split_by_date(train_set, split_date, constants.COL_TIMESTAMP)

    train_split = train_set[train_mask].copy()
    val_split = train_set[val_mask].copy()

    max_train_timestamp = train_split[constants.COL_TIMESTAMP].max()
    min_val_timestamp = val_split[constants.COL_TIMESTAMP].min()

    if min_val_timestamp <= max_train_timestamp:
        raise ValueError(
            f"Temporal split validation failed: min validation timestamp ({min_val_timestamp}) "
            f"is not greater than max train timestamp ({max_train_timestamp})."
        )

    X_train, y_train, agg_reference = prepare_training_data(
        train_split, use_unread_for_features=config.USE_READING_BEHAVIOR_FEATURES
    )

    X_train_with_agg = add_aggregate_features(X_train.copy(), agg_reference)
    val_split_with_agg = add_aggregate_features(val_split.copy(), agg_reference)

    X_train_final = handle_missing_values(X_train_with_agg, agg_reference)
    val_split_final = handle_missing_values(val_split_with_agg, agg_reference)

    exclude_cols = [
        constants.COL_SOURCE, config.TARGET, constants.COL_PREDICTION, 
        constants.COL_TIMESTAMP, constants.COL_HAS_READ
    ]
    features = [col for col in X_train_final.columns if col not in exclude_cols]

    non_feature_object_cols = X_train_final[features].select_dtypes(include=["object"]).columns.tolist()
    features = [f for f in features if f not in non_feature_object_cols]

    X_train_features = X_train_final[features]
    X_val_features = val_split_final[features]
    y_val = val_split_final[config.TARGET]

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model = lgb.LGBMRegressor(**config.LGB_PARAMS)

    fit_params = config.LGB_FIT_PARAMS.copy()
    fit_params["callbacks"] = [lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS, verbose=True)]

    model.fit(
        X_train_features,
        y_train,
        eval_set=[(X_val_features, y_val)],
        eval_metric=fit_params["eval_metric"],
        callbacks=fit_params["callbacks"],
    )

    val_preds = model.predict(X_val_features)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    mae = mean_absolute_error(y_val, val_preds)
    
    range_width = 10.0
    mae_norm = mae / range_width
    rmse_norm = rmse / range_width
    score = 1 - (0.5 * rmse_norm + 0.5 * mae_norm)

    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    model_path = config.MODEL_DIR / config.MODEL_FILENAME
    model.booster_.save_model(str(model_path))

    importance_path = config.MODEL_DIR / "feature_importance.csv"
    feature_importance.to_csv(importance_path, index=False)

if __name__ == "__main__":
    train()