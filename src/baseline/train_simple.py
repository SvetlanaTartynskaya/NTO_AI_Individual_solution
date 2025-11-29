import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.baseline import config, constants
from src.baseline.features import add_aggregate_features, handle_missing_values
from src.baseline.temporal_split import get_split_date_from_ratio, temporal_split_by_date

def train_simple() -> None:
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME

    if not processed_path.exists():
        raise FileNotFoundError(f"Processed data not found at {processed_path}")

    featured_df = pd.read_parquet(processed_path, engine="pyarrow")

    train_set = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()
    train_set = train_set[train_set[constants.COL_HAS_READ] == 1].copy()

    if not pd.api.types.is_datetime64_any_dtype(train_set[constants.COL_TIMESTAMP]):
        train_set[constants.COL_TIMESTAMP] = pd.to_datetime(train_set[constants.COL_TIMESTAMP])

    split_date = get_split_date_from_ratio(train_set, config.TEMPORAL_SPLIT_RATIO, constants.COL_TIMESTAMP)
    train_mask, val_mask = temporal_split_by_date(train_set, split_date, constants.COL_TIMESTAMP)
    train_split = train_set[train_mask].copy()
    val_split = train_set[val_mask].copy()

    train_with_agg = add_aggregate_features(train_split.copy(), train_split)
    val_with_agg = add_aggregate_features(val_split.copy(), train_split)

    train_final = safe_handle_missing_values(train_with_agg, train_split)
    val_final = safe_handle_missing_values(val_with_agg, train_split)

    exclude_cols = [
        constants.COL_SOURCE, 
        config.TARGET, 
        constants.COL_PREDICTION, 
        constants.COL_TIMESTAMP, 
        constants.COL_HAS_READ
    ]
    
    features = [col for col in train_final.columns if col not in exclude_cols]
    object_cols = train_final[features].select_dtypes(include=["object"]).columns.tolist()
    features = [f for f in features if f not in object_cols]

    X_train = train_final[features]
    y_train = train_final[config.TARGET]
    X_val = val_final[features] 
    y_val = val_final[config.TARGET]

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    simple_params = {
        "objective": "rmse",
        "metric": "rmse", 
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
        "random_state": config.RANDOM_STATE,
    }
    
    model = lgb.LGBMRegressor(**simple_params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)],
    )

    val_preds = model.predict(X_val)
    
    try:
        rmse = mean_squared_error(y_val, val_preds, squared=False)
    except TypeError:
        mse = mean_squared_error(y_val, val_preds)
        rmse = np.sqrt(mse)
        
    mae = mean_absolute_error(y_val, val_preds)
    
    range_width = 10.0
    mae_norm = mae / range_width
    rmse_norm = rmse / range_width
    score = 1 - (0.5 * rmse_norm + 0.5 * mae_norm)

    model_path = config.MODEL_DIR / config.MODEL_FILENAME
    model.booster_.save_model(str(model_path))

    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

def safe_handle_missing_values(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    read_books = train_df[train_df[constants.COL_HAS_READ] == 1]
    global_mean = read_books[config.TARGET].mean() if not read_books.empty else 5.0

    basic_features = {
        constants.COL_AGE: df[constants.COL_AGE].median(),
        constants.COL_AVG_RATING: global_mean,
    }
    
    for col, fill_value in basic_features.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)

    aggregate_features = {
        constants.F_USER_MEAN_RATING: global_mean,
        constants.F_BOOK_MEAN_RATING: global_mean, 
        constants.F_AUTHOR_MEAN_RATING: global_mean,
        constants.F_USER_RATINGS_COUNT: 0,
        constants.F_BOOK_RATINGS_COUNT: 0,
        constants.F_BOOK_GENRES_COUNT: 0,
    }
    
    for col, fill_value in aggregate_features.items():
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)

    tfidf_cols = [col for col in df.columns if col.startswith("tfidf_")]
    for col in tfidf_cols:
        df[col] = df[col].fillna(0.0)
            
    bert_cols = [col for col in df.columns if col.startswith("bert_")]
    for col in bert_cols:
        df[col] = df[col].fillna(0.0)

    new_features = [
        constants.F_USER_BOOKS_READ, constants.F_USER_TOTAL_INTERACTIONS, 
        constants.F_USER_READ_RATIO, constants.F_BOOK_WANT_COUNT
    ]
    
    for feature in new_features:
        if feature in df.columns:
            if 'count' in feature:
                df[feature] = df[feature].fillna(0)
            else:
                df[feature] = df[feature].fillna(global_mean)

    return df

if __name__ == "__main__":
    train_simple()