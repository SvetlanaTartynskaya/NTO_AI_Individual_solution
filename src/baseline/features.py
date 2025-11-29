import time
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from . import config, constants

def add_advanced_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    train_df = train_df.copy()
    
    df[constants.COL_TIMESTAMP] = pd.to_datetime(df[constants.COL_TIMESTAMP])
    train_df[constants.COL_TIMESTAMP] = pd.to_datetime(train_df[constants.COL_TIMESTAMP])
    
    user_book_interactions = train_df.groupby([constants.COL_USER_ID, constants.COL_BOOK_ID]).size().reset_index(name='user_book_interaction_count')
    df = df.merge(user_book_interactions, on=[constants.COL_USER_ID, constants.COL_BOOK_ID], how='left')
    df['user_book_interaction_count'] = df['user_book_interaction_count'].fillna(0)
    
    df['days_since_first_activity'] = (df[constants.COL_TIMESTAMP] - df.groupby(constants.COL_USER_ID)[constants.COL_TIMESTAMP].transform('min')).dt.days
    overall_max = df[constants.COL_TIMESTAMP].max()
    df['days_since_last_activity'] = (overall_max - df[constants.COL_TIMESTAMP]).dt.days
    
    global_mean = train_df[config.TARGET].mean()
    df['user_global_diff'] = df[constants.F_USER_MEAN_RATING] - global_mean
    df['book_global_diff'] = df[constants.F_BOOK_MEAN_RATING] - global_mean
    df['user_book_diff'] = df[constants.F_USER_MEAN_RATING] - df[constants.F_BOOK_MEAN_RATING]
    
    recent_cutoff = overall_max - pd.Timedelta(days=180)
    recent_ratings = train_df[train_df[constants.COL_TIMESTAMP] > recent_cutoff]
    
    if not recent_ratings.empty:
        book_recent = recent_ratings.groupby(constants.COL_BOOK_ID)[config.TARGET].agg(['mean', 'count']).reset_index()
        book_recent.columns = [constants.COL_BOOK_ID, 'book_recent_mean', 'book_recent_count']
        df = df.merge(book_recent, on=constants.COL_BOOK_ID, how='left')
        
        user_recent = recent_ratings.groupby(constants.COL_USER_ID)[config.TARGET].agg(['mean', 'count']).reset_index()
        user_recent.columns = [constants.COL_USER_ID, 'user_recent_mean', 'user_recent_count']
        df = df.merge(user_recent, on=constants.COL_USER_ID, how='left')
    
    new_features = ['user_book_interaction_count', 'days_since_first_activity', 'days_since_last_activity',
                   'user_global_diff', 'book_global_diff', 'user_book_diff',
                   'book_recent_mean', 'book_recent_count', 'user_recent_mean', 'user_recent_count']
    
    for feature in new_features:
        if feature in df.columns:
            if 'count' in feature:
                df[feature] = df[feature].fillna(0)
            elif 'mean' in feature or 'diff' in feature:
                df[feature] = df[feature].fillna(0)
            else:
                df[feature] = df[feature].fillna(-1)
    
    return df

def add_genre_features(df: pd.DataFrame, book_genres_df: pd.DataFrame) -> pd.DataFrame:
    genre_counts = book_genres_df.groupby(constants.COL_BOOK_ID)[constants.COL_GENRE_ID].count().reset_index()
    genre_counts.columns = [
        constants.COL_BOOK_ID,
        constants.F_BOOK_GENRES_COUNT,
    ]
    
    result_df = df.merge(genre_counts, on=constants.COL_BOOK_ID, how="left")
    return result_df

def add_aggregate_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    user_agg = train_df.groupby(constants.COL_USER_ID)[config.TARGET].agg(["mean", "count"]).reset_index()
    user_agg.columns = [
        constants.COL_USER_ID,
        constants.F_USER_MEAN_RATING,
        constants.F_USER_RATINGS_COUNT,
    ]

    book_agg = train_df.groupby(constants.COL_BOOK_ID)[config.TARGET].agg(["mean", "count"]).reset_index()
    book_agg.columns = [
        constants.COL_BOOK_ID,
        constants.F_BOOK_MEAN_RATING,
        constants.F_BOOK_RATINGS_COUNT,
    ]

    author_agg = train_df.groupby(constants.COL_AUTHOR_ID)[config.TARGET].agg(["mean"]).reset_index()
    author_agg.columns = [constants.COL_AUTHOR_ID, constants.F_AUTHOR_MEAN_RATING]

    df = df.merge(user_agg, on=constants.COL_USER_ID, how="left")
    df = df.merge(book_agg, on=constants.COL_BOOK_ID, how="left")
    return df.merge(author_agg, on=constants.COL_AUTHOR_ID, how="left")

def add_reading_behavior_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    if not config.USE_READING_BEHAVIOR_FEATURES:
        return df
        
    user_reading_stats = train_df.groupby(constants.COL_USER_ID).agg({
        constants.COL_HAS_READ: ['sum', 'count', 'mean']
    }).reset_index()
    user_reading_stats.columns = [
        constants.COL_USER_ID,
        constants.F_USER_BOOKS_READ,
        constants.F_USER_TOTAL_INTERACTIONS, 
        constants.F_USER_READ_RATIO
    ]
    
    want_to_read = train_df[train_df[constants.COL_HAS_READ] == 0].copy()
    actually_read = train_df[train_df[constants.COL_HAS_READ] == 1].copy()
    
    if not want_to_read.empty and not actually_read.empty:
        merged_readings = actually_read.merge(
            want_to_read[[constants.COL_USER_ID, constants.COL_BOOK_ID, constants.COL_TIMESTAMP]], 
            on=[constants.COL_USER_ID, constants.COL_BOOK_ID],
            suffixes=('_read', '_want'),
            how='inner'
        )
        
        if not merged_readings.empty:
            merged_readings[constants.COL_TIMESTAMP + '_read'] = pd.to_datetime(merged_readings[constants.COL_TIMESTAMP + '_read'])
            merged_readings[constants.COL_TIMESTAMP + '_want'] = pd.to_datetime(merged_readings[constants.COL_TIMESTAMP + '_want'])
            
            merged_readings['days_to_read'] = (
                merged_readings[constants.COL_TIMESTAMP + '_read'] - merged_readings[constants.COL_TIMESTAMP + '_want']
            ).dt.days
            
            user_reading_times = merged_readings.groupby(constants.COL_USER_ID)['days_to_read'].agg([
                'mean', 'median', 'min', 'max'
            ]).reset_index()
            user_reading_times.columns = [constants.COL_USER_ID] + [
                constants.F_USER_DAYS_TO_READ_MEAN,
                constants.F_USER_DAYS_TO_READ_MEDIAN, 
                constants.F_USER_DAYS_TO_READ_MIN,
                constants.F_USER_DAYS_TO_READ_MAX
            ]
            
            df = df.merge(user_reading_stats, on=constants.COL_USER_ID, how='left')
            df = df.merge(user_reading_times, on=constants.COL_USER_ID, how='left')
            return df
    
    df = df.merge(user_reading_stats, on=constants.COL_USER_ID, how='left')
    return df

def add_unread_book_features(df: pd.DataFrame, train_df: pd.DataFrame, book_genres_df: pd.DataFrame) -> pd.DataFrame:
    if not config.USE_UNREAD_BOOK_FEATURES:
        return df
        
    want_to_read_df = train_df[train_df[constants.COL_HAS_READ] == 0].copy()
    
    if not want_to_read_df.empty:
        book_want_popularity = want_to_read_df.groupby(constants.COL_BOOK_ID).agg({
            constants.COL_USER_ID: 'count'
        }).reset_index()
        book_want_popularity.columns = [constants.COL_BOOK_ID, constants.F_BOOK_WANT_COUNT]
        
        user_want_genres = want_to_read_df.merge(
            book_genres_df, on=constants.COL_BOOK_ID, how='left'
        ).groupby([constants.COL_USER_ID, constants.COL_GENRE_ID]).agg({
            constants.COL_BOOK_ID: 'count'
        }).reset_index()
        user_want_genres.columns = [constants.COL_USER_ID, constants.COL_GENRE_ID, constants.F_USER_WANT_GENRE_COUNT]
        
        user_top_want_genre = user_want_genres.loc[
            user_want_genres.groupby(constants.COL_USER_ID)[constants.F_USER_WANT_GENRE_COUNT].idxmax()
        ][[constants.COL_USER_ID, constants.COL_GENRE_ID]]
        user_top_want_genre.columns = [constants.COL_USER_ID, constants.F_USER_TOP_WANT_GENRE]
        
        df = df.merge(book_want_popularity, on=constants.COL_BOOK_ID, how='left')
        df = df.merge(user_top_want_genre, on=constants.COL_USER_ID, how='left')
    
    return df

def enhanced_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    if not config.USE_ENHANCED_TEMPORAL_FEATURES:
        return df
        
    df = df.copy()
    df[constants.COL_TIMESTAMP] = pd.to_datetime(df[constants.COL_TIMESTAMP])
    
    df[constants.F_TIMESTAMP_YEAR] = df[constants.COL_TIMESTAMP].dt.year
    df[constants.F_TIMESTAMP_MONTH] = df[constants.COL_TIMESTAMP].dt.month  
    df[constants.F_TIMESTAMP_DAY] = df[constants.COL_TIMESTAMP].dt.day
    df[constants.F_TIMESTAMP_DAYOFWEEK] = df[constants.COL_TIMESTAMP].dt.dayofweek
    df[constants.F_TIMESTAMP_HOUR] = df[constants.COL_TIMESTAMP].dt.hour
    df[constants.F_TIMESTAMP_WEEKEND] = df[constants.F_TIMESTAMP_DAYOFWEEK].isin([5, 6]).astype(int)
    
    df[constants.F_TIMESTAMP_SEASON] = (df[constants.F_TIMESTAMP_MONTH] % 12 + 3) // 3
    
    user_first_activity = df.groupby(constants.COL_USER_ID)[constants.COL_TIMESTAMP].transform('min')
    df[constants.F_DAYS_SINCE_FIRST_ACTIVITY] = (df[constants.COL_TIMESTAMP] - user_first_activity).dt.days
    
    overall_max_time = df[constants.COL_TIMESTAMP].max()
    df[constants.F_DAYS_SINCE_LAST_ACTIVITY] = (overall_max_time - df[constants.COL_TIMESTAMP]).dt.days
    
    return df

def add_interaction_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    if not config.USE_INTERACTION_FEATURES:
        return df
        
    read_books_df = train_df[train_df[constants.COL_HAS_READ] == 1].copy()
    
    gender_popularity = read_books_df.groupby([constants.COL_BOOK_ID, constants.COL_GENDER])[config.TARGET].agg(['mean', 'count']).reset_index()
    gender_popularity.columns = [
        constants.COL_BOOK_ID, constants.COL_GENDER, 
        constants.F_BOOK_GENDER_MEAN_RATING, constants.F_BOOK_GENDER_COUNT
    ]
    
    read_books_temp = read_books_df.copy()
    read_books_temp[constants.F_AGE_GROUP] = pd.cut(
        read_books_temp[constants.COL_AGE], 
        bins=[0, 18, 25, 35, 50, 100], 
        labels=['teen', 'young', 'adult', 'middle', 'senior']
    )
    
    age_popularity = read_books_temp.groupby([constants.COL_BOOK_ID, constants.F_AGE_GROUP])[config.TARGET].agg(['mean', 'count']).reset_index()
    age_popularity.columns = [
        constants.COL_BOOK_ID, constants.F_AGE_GROUP, 
        constants.F_BOOK_AGE_MEAN_RATING, constants.F_BOOK_AGE_COUNT
    ]
    
    df[constants.F_AGE_GROUP] = pd.cut(
        df[constants.COL_AGE], 
        bins=[0, 18, 25, 35, 50, 100], 
        labels=['teen', 'young', 'adult', 'middle', 'senior']
    )
    
    df = df.merge(gender_popularity, on=[constants.COL_BOOK_ID, constants.COL_GENDER], how='left')
    df = df.merge(age_popularity, on=[constants.COL_BOOK_ID, constants.F_AGE_GROUP], how='left')
    
    return df

def add_text_features(df: pd.DataFrame, train_df: pd.DataFrame, descriptions_df: pd.DataFrame) -> pd.DataFrame:
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    vectorizer_path = config.MODEL_DIR / constants.TFIDF_VECTORIZER_FILENAME

    train_books = train_df[constants.COL_BOOK_ID].unique()

    train_descriptions = descriptions_df[descriptions_df[constants.COL_BOOK_ID].isin(train_books)].copy()
    train_descriptions[constants.COL_DESCRIPTION] = train_descriptions[constants.COL_DESCRIPTION].fillna("")

    if vectorizer_path.exists():
        vectorizer = joblib.load(vectorizer_path)
    else:
        vectorizer = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            min_df=config.TFIDF_MIN_DF,
            max_df=config.TFIDF_MAX_DF,
            ngram_range=config.TFIDF_NGRAM_RANGE,
            analyzer=config.TFIDF_ANALYZER,
            sublinear_tf=config.TFIDF_SUBLINEAR_TF,
        )
        vectorizer.fit(train_descriptions[constants.COL_DESCRIPTION])
        joblib.dump(vectorizer, vectorizer_path)

    all_descriptions = descriptions_df[[constants.COL_BOOK_ID, constants.COL_DESCRIPTION]].copy()
    all_descriptions[constants.COL_DESCRIPTION] = all_descriptions[constants.COL_DESCRIPTION].fillna("")
    description_map = dict(
        zip(all_descriptions[constants.COL_BOOK_ID], all_descriptions[constants.COL_DESCRIPTION], strict=False)
    )
    df_descriptions = df[constants.COL_BOOK_ID].map(description_map).fillna("")
    tfidf_matrix = vectorizer.transform(df_descriptions)

    tfidf_feature_names = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=tfidf_feature_names,
        index=df.index,
    )

    df_with_tfidf = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

    return df_with_tfidf

def create_features(
    df: pd.DataFrame, book_genres_df: pd.DataFrame, descriptions_df: pd.DataFrame, include_aggregates: bool = False
) -> pd.DataFrame:
    train_df = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    df = add_reading_behavior_features(df, train_df)
    df = add_unread_book_features(df, train_df, book_genres_df)
    df = enhanced_temporal_features(df)
    df = add_interaction_features(df, train_df)

    for col in config.CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df

def handle_missing_values(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    read_books = train_df[train_df[constants.COL_HAS_READ] == 1]
    global_mean = read_books[config.TARGET].mean() if not read_books.empty else 5.0

    age_median = df[constants.COL_AGE].median()
    df[constants.COL_AGE] = df[constants.COL_AGE].fillna(age_median)

    if constants.F_USER_MEAN_RATING in df.columns:
        df[constants.F_USER_MEAN_RATING] = df[constants.F_USER_MEAN_RATING].fillna(global_mean)
    if constants.F_BOOK_MEAN_RATING in df.columns:
        df[constants.F_BOOK_MEAN_RATING] = df[constants.F_BOOK_MEAN_RATING].fillna(global_mean)
    if constants.F_AUTHOR_MEAN_RATING in df.columns:
        df[constants.F_AUTHOR_MEAN_RATING] = df[constants.F_AUTHOR_MEAN_RATING].fillna(global_mean)

    if constants.F_USER_RATINGS_COUNT in df.columns:
        df[constants.F_USER_RATINGS_COUNT] = df[constants.F_USER_RATINGS_COUNT].fillna(0)
    if constants.F_BOOK_RATINGS_COUNT in df.columns:
        df[constants.F_BOOK_RATINGS_COUNT] = df[constants.F_BOOK_RATINGS_COUNT].fillna(0)

    df[constants.COL_AVG_RATING] = df[constants.COL_AVG_RATING].fillna(global_mean)

    df[constants.F_BOOK_GENRES_COUNT] = df[constants.F_BOOK_GENRES_COUNT].fillna(0)

    tfidf_cols = [col for col in df.columns if col.startswith("tfidf_")]
    for col in tfidf_cols:
        df[col] = df[col].fillna(0.0)

    bert_cols = [col for col in df.columns if col.startswith("bert_")]
    for col in bert_cols:
        df[col] = df[col].fillna(0.0)

    new_numeric_features = [
        constants.F_USER_BOOKS_READ, constants.F_USER_TOTAL_INTERACTIONS, constants.F_USER_READ_RATIO,
        constants.F_USER_DAYS_TO_READ_MEAN, constants.F_USER_DAYS_TO_READ_MEDIAN,
        constants.F_USER_DAYS_TO_READ_MIN, constants.F_USER_DAYS_TO_READ_MAX,
        constants.F_BOOK_WANT_COUNT, constants.F_BOOK_GENDER_MEAN_RATING,
        constants.F_BOOK_GENDER_COUNT, constants.F_BOOK_AGE_MEAN_RATING, constants.F_BOOK_AGE_COUNT
    ]
    
    for feature in new_numeric_features:
        if feature in df.columns:
            if 'mean' in feature or 'ratio' in feature:
                df[feature] = df[feature].fillna(global_mean)
            elif 'count' in feature:
                df[feature] = df[feature].fillna(0)
            else:
                df[feature] = df[feature].fillna(constants.MISSING_NUM_VALUE)

    for col in config.CAT_FEATURES:
        if col in df.columns:
            if df[col].dtype.name == "category":
                if constants.MISSING_CAT_VALUE not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories([constants.MISSING_CAT_VALUE])
                df[col] = df[col].fillna(constants.MISSING_CAT_VALUE)
            elif df[col].dtype.name == "object":
                df[col] = df[col].fillna(constants.MISSING_CAT_VALUE)
            elif pd.api.types.is_numeric_dtype(df[col].dtype) and df[col].isna().any():
                df[col] = df[col].fillna(constants.MISSING_NUM_VALUE)

    new_categorical_features = [constants.F_USER_TOP_WANT_GENRE, constants.F_AGE_GROUP]
    for feature in new_categorical_features:
        if feature in df.columns:
            if df[feature].dtype.name == "category":
                if constants.MISSING_CAT_VALUE not in df[feature].cat.categories:
                    df[feature] = df[feature].cat.add_categories([constants.MISSING_CAT_VALUE])
                df[feature] = df[feature].fillna(constants.MISSING_CAT_VALUE)
            else:
                df[feature] = df[feature].fillna(constants.MISSING_CAT_VALUE)
                df[feature] = df[feature].astype("category")

    return df