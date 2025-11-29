from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from . import constants

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "models"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"

N_SPLITS = 5
RANDOM_STATE = 42
TARGET = constants.COL_TARGET

TEMPORAL_SPLIT_RATIO = 0.8

EARLY_STOPPING_ROUNDS = 50
MODEL_FILENAME = "lgb_model.txt"

TFIDF_MAX_FEATURES = 1000
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.9
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_ANALYZER = "char_wb"
TFIDF_SUBLINEAR_TF = True

BERT_MODEL_NAME = constants.BERT_MODEL_NAME
BERT_BATCH_SIZE = 16
BERT_MAX_LENGTH = 512
BERT_EMBEDDING_DIM = 768
BERT_DEVICE = "cuda" if torch and torch.cuda.is_available() else "cpu"
BERT_GPU_MEMORY_FRACTION = 0.75

CAT_FEATURES = [
    constants.COL_USER_ID,
    constants.COL_BOOK_ID,
    constants.COL_GENDER,
    constants.COL_AGE,
    constants.COL_AUTHOR_ID,
    constants.COL_PUBLICATION_YEAR,
    constants.COL_LANGUAGE,
    constants.COL_PUBLISHER,
    'timestamp_season',
    'age_group',
]

LGB_PARAMS = {
    "objective": "rmse",
    "metric": "rmse",
    "n_estimators": 3000,
    "learning_rate": 0.005,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.2,
    "lambda_l2": 0.3,
    "num_leaves": 127,
    "min_data_in_leaf": 50,
    "max_depth": -1,
    "verbose": -1,
    "n_jobs": -1,
    "seed": RANDOM_STATE,
    "boosting_type": "gbdt",
}

LGB_FIT_PARAMS = {
    "eval_metric": "rmse",
    "callbacks": [],
}

LGB_OPTIMIZED_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 1500,
    "learning_rate": 0.01,
    "num_leaves": 127,
    "max_depth": 10,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": RANDOM_STATE,
}

ENSEMBLE_WEIGHTS = {
    "lgb": 0.6,
    "xgb": 0.3, 
    "rf": 0.1
}

USE_READING_BEHAVIOR_FEATURES = True
USE_UNREAD_BOOK_FEATURES = True  
USE_ENHANCED_TEMPORAL_FEATURES = True
USE_INTERACTION_FEATURES = True