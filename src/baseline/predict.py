import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config, constants
from .features import add_aggregate_features, handle_missing_values

def prepare_prediction_data(test_set: pd.DataFrame, train_set: pd.DataFrame, use_unread_for_features: bool = True):
    if use_unread_for_features:
        agg_reference = train_set.copy()
    else:
        agg_reference = train_set[train_set[constants.COL_HAS_READ] == 1].copy()
    
    test_set_with_agg = add_aggregate_features(test_set.copy(), agg_reference)
    test_set_final = safe_handle_missing_values_predict(test_set_with_agg, agg_reference)
    
    return test_set_final

def safe_handle_missing_values_predict(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
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

    return df

def predict() -> None:
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME

    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_path}. "
            "Please run 'poetry run python -m src.baseline.prepare_data' first."
        )

    featured_df = pd.read_parquet(processed_path, engine="pyarrow")

    train_set = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()
    test_set = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TEST].copy()

    test_set_final = prepare_prediction_data(
        test_set, train_set, use_unread_for_features=config.USE_READING_BEHAVIOR_FEATURES
    )

    exclude_cols = [
        constants.COL_SOURCE, config.TARGET, constants.COL_PREDICTION,
        constants.COL_TIMESTAMP, constants.COL_HAS_READ
    ]
    features = [col for col in test_set_final.columns if col not in exclude_cols]

    non_feature_object_cols = test_set_final[features].select_dtypes(include=["object"]).columns.tolist()
    features = [f for f in features if f not in non_feature_object_cols]

    X_test = test_set_final[features]

    model_path = config.MODEL_DIR / config.MODEL_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. " 
            "Please run 'poetry run python -m src.baseline.train' first."
        )

    model = lgb.Booster(model_file=str(model_path))

    model_features = model.feature_name()
    missing_features = set(model_features) - set(features)
    extra_features = set(features) - set(model_features)
    
    X_test_aligned = X_test.reindex(columns=model_features, fill_value=0)

    test_preds = model.predict(X_test_aligned)

    clipped_preds = np.clip(test_preds, constants.PREDICTION_MIN_VALUE, constants.PREDICTION_MAX_VALUE)

    submission_df = test_set[[constants.COL_USER_ID, constants.COL_BOOK_ID]].copy()
    submission_df[constants.COL_PREDICTION] = clipped_preds

    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = config.SUBMISSION_DIR / constants.SUBMISSION_FILENAME

    submission_df.to_csv(submission_path, index=False)

if __name__ == "__main__":
    predict()