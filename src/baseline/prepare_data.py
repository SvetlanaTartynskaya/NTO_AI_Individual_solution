from . import config, constants
from .data_processing import load_and_merge_data
from .features import create_features

def prepare_data() -> None:
    merged_df, book_genres_df, _, descriptions_df = load_and_merge_data()

    featured_df = create_features(merged_df, book_genres_df, descriptions_df, include_aggregates=False)

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME

    featured_df.to_parquet(processed_path, index=False, engine="pyarrow", compression="snappy")

    train_data = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN]
    test_data = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TEST]
    
    train_has_read_1 = len(train_data[train_data[constants.COL_HAS_READ] == 1])
    train_has_read_0 = len(train_data[train_data[constants.COL_HAS_READ] == 0])
    
    total_features = len(featured_df.columns)
    
    new_features = [
        f for f in featured_df.columns 
        if any(pattern in f for pattern in [
            'user_books_read', 'user_total_interactions', 'user_read_ratio',
            'days_to_read', 'book_want_count', 'user_want_genre', 
            'timestamp_', 'days_since_', 'book_gender_', 'book_age_', 'age_group'
        ])
    ]

if __name__ == "__main__":
    prepare_data()