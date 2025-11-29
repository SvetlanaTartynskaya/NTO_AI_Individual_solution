import sys
import os
import traceback
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from src.baseline import config, constants

def add_safe_advanced_features(df, train_df):
    df = df.copy()
    train_df = train_df.copy()
    
    user_book_counts = train_df.groupby([constants.COL_USER_ID, constants.COL_BOOK_ID]).size().reset_index(name='user_book_interactions')
    df = df.merge(user_book_counts, on=[constants.COL_USER_ID, constants.COL_BOOK_ID], how='left')
    df['user_book_interactions'] = df['user_book_interactions'].fillna(0)
    
    df[constants.COL_TIMESTAMP] = pd.to_datetime(df[constants.COL_TIMESTAMP])
    df['days_since_first'] = (df[constants.COL_TIMESTAMP] - df.groupby(constants.COL_USER_ID)[constants.COL_TIMESTAMP].transform('min')).dt.days
    df['user_activity_recency'] = (df[constants.COL_TIMESTAMP].max() - df[constants.COL_TIMESTAMP]).dt.days
    
    if 'user_mean_rating' in df.columns and 'book_mean_rating' in df.columns:
        df['rating_diff_user_book'] = df['user_mean_rating'] - df['book_mean_rating']
        global_mean = train_df[config.TARGET].mean()
        df['rating_diff_user_global'] = df['user_mean_rating'] - global_mean
    
    return df

def get_safe_features(df):
    base_features = [
        'user_id', 'book_id', 'gender', 'age', 'author_id', 
        'publication_year', 'language', 'publisher', 'avg_rating'
    ]
    
    aggregate_features = [
        'user_mean_rating', 'user_ratings_count', 'book_mean_rating', 
        'book_ratings_count', 'author_mean_rating', 'book_genres_count'
    ]
    
    advanced_features = [
        'user_book_interactions', 'days_since_first', 'user_activity_recency',
        'rating_diff_user_book', 'rating_diff_user_global'
    ]
    
    all_possible_features = base_features + aggregate_features + advanced_features
    existing_features = [f for f in all_possible_features if f in df.columns]
    
    return existing_features

def run_improved_pipeline():
    try:
        from src.baseline.data_processing import load_and_merge_data
        
        merged_df, book_genres_df, _, descriptions_df = load_and_merge_data()
        
        train_df = merged_df[
            (merged_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN) & 
            (merged_df[constants.COL_HAS_READ] == 1)
        ].copy()
        
        test_df = merged_df[merged_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TEST].copy()
        
        from src.baseline.features import add_aggregate_features, add_genre_features
        
        train_with_agg = add_aggregate_features(train_df.copy(), train_df)
        test_with_agg = add_aggregate_features(test_df.copy(), train_df)
        
        train_with_genres = add_genre_features(train_with_agg, book_genres_df)
        test_with_genres = add_genre_features(test_with_agg, book_genres_df)
        
        train_final = add_safe_advanced_features(train_with_genres, train_df)
        test_final = add_safe_advanced_features(test_with_genres, train_df)
        
        train_final[constants.COL_TIMESTAMP] = pd.to_datetime(train_final[constants.COL_TIMESTAMP])
        train_final = train_final.sort_values(constants.COL_TIMESTAMP)
        
        split_idx = int(len(train_final) * 0.8)
        train_split = train_final.iloc[:split_idx].copy()
        val_split = train_final.iloc[split_idx:].copy()
        
        features = get_safe_features(train_split)
        
        exclude_cols = [constants.COL_SOURCE, config.TARGET, constants.COL_PREDICTION, 
                       constants.COL_TIMESTAMP, constants.COL_HAS_READ]
        features = [f for f in features if f not in exclude_cols]
        
        features = [f for f in features if train_split[f].dtype != 'object']
        
        X_train = train_split[features]
        y_train = train_split[config.TARGET]
        X_val = val_split[features]
        y_val = val_split[config.TARGET]
        X_test = test_final[features]
        
        lgb_optimized = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.02,
            num_leaves=63,
            max_depth=8,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=100
        )
        
        lgb_optimized.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
        )
        
        val_preds = lgb_optimized.predict(X_val)
        
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        mae = mean_absolute_error(y_val, val_preds)
        rmse_norm = rmse / 10.0
        mae_norm = mae / 10.0
        score = 1 - (rmse_norm + mae_norm) / 2
        
        test_preds = lgb_optimized.predict(X_test)
        
        config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
        submission_df = test_df[[constants.COL_USER_ID, constants.COL_BOOK_ID]].copy()
        submission_df[constants.COL_PREDICTION] = np.clip(test_preds, 0, 10)
        
        submission_path = config.SUBMISSION_DIR / "submission.csv"
        submission_df.to_csv(submission_path, index=False)
        
        importance = pd.DataFrame({
            'feature': features,
            'importance': lgb_optimized.feature_importances_
        }).sort_values('importance', ascending=False)
        
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    run_improved_pipeline()