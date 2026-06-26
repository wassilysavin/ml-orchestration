from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

KAGGLE_DATASET_SLUG = "jessicali9530/kuc-hackathon-winter-2018"
KAGGLE_TRAIN_FILENAME = "drugsComTrain_raw.csv"
KAGGLE_TEST_FILENAME = "drugsComTest_raw.csv"

TRAIN_RAW_FILE = RAW_DIR / "drugsComTrain_raw.csv"
TEST_RAW_FILE = RAW_DIR / "drugsComTest_raw.csv"
TEST_RAW_PARQUET_FILE = RAW_DIR / "drugsComTest_raw.parquet"

COMBINED_PROCESSED_FILE = PROCESSED_DIR / "drug_reviews.parquet"
REFERENCE_PROCESSED_FILE = PROCESSED_DIR / "drug_reviews_reference.parquet"
CURRENT_PROCESSED_FILE = PROCESSED_DIR / "drug_reviews_current.parquet"
TRAINING_PROCESSED_FILE = PROCESSED_DIR / "drug_reviews_training.parquet"
SUMMARY_FILE = PROCESSED_DIR / "quality_summary.json"

INCOMING_EVAL_FRACTION = 0.3
INCOMING_SPLIT_SALT = "incoming-holdout"

DATE_FORMAT = "%d-%b-%y"
REFERENCE_SPLIT_LABEL = "reference"
CURRENT_SPLIT_LABEL = "current"

UCI_DATASET_NAME = "UCI Drug Review Dataset"
UCI_DATASET_PAGE = "https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018"

MAX_CONDITION_MISSINGNESS = 0.01
MAX_RATING_KS_STATISTIC = 0.02
MAX_CONDITION_TVD = 0.05

MODELS_DIR = PROJECT_ROOT / "models"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
MODEL_FILENAME = "model.onnx"
MODEL_METADATA_FILENAME = "metadata.json"
MLFLOW_EXPERIMENT_NAME = "drug-review-sentiment"

POSITIVE_RATING_FLOOR = 7
NEGATIVE_RATING_CEILING = 4
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0

MIN_TRAINING_ROWS = 1000
RANDOM_SEED = 0

MIN_MACRO_F1_MARGIN_OVER_BASELINE = 0.20
MIN_INVARIANCE_AGREEMENT = 0.95
INVARIANCE_SAMPLE_SIZE = 500
MIN_NEGATION_PROBABILITY_SHIFT = 0.15
NEGATION_SAMPLE_SIZE = 200

NEGATION_PREFIXES = (
    "I disagree, ",
    "Not really - ",
    "On the contrary, ",
)

FLOW_VERSION_TAG = "flow_version_id"
FLOW_NAME_TAG = "flow_name"
TRAINING_FLOW_NAME = "drug-review-training"

INCOMING_DATASET_ENV = "INCOMING_DATASET"

COMPONENT_NAME_TAG = "component_name"
COMPONENT_VERSION_TAG = "component_version_id"
DATASET_VERSION_TAG = "dataset_version_id"
COMPONENTS_EXPERIMENT_NAME = "drug-review-components"
DATASET_PULL_COMPONENT = "dataset-pull"
DATA_PREP_COMPONENT = "data-prep"

MONITORING_EXPERIMENT_NAME = "drug-review-monitoring"
DRIFT_FEATURE = "review_length"
PSI_BINS = 10
PSI_MODERATE_THRESHOLD = 0.10
PSI_SIGNIFICANT_THRESHOLD = 0.25

SELECTION_EXPERIMENT_NAME = "drug-review-selection"
SELECTION_WINNER_ARTIFACT = "selection/winner"

AB_EXPERIMENT_NAME = "drug-review-abtest"
AB_SPLIT_ATTRIBUTE = "unique_id"
AB_VARIANT_A = "A"
AB_VARIANT_B = "B"
AB_WINNER_ARTIFACT = "ab/winner"

ADAPTATION_EXPERIMENT_NAME = "drug-review-adaptation"
CHAMPION_FILENAME = "champion.json"
ADAPTATION_STATE_FILENAME = "adaptation_state.json"
PROMOTION_MIN_MACRO_F1_MARGIN = 0.01
RETRAIN_COOLDOWN_SECONDS = 3600
