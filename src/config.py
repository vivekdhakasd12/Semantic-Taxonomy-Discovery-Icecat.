import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = "/Users/dev/Downloads/icecat_data_train.json"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

LABEL_COL = "Category.Name.Value"  # Corrected based on file inspection
TEXT_COLS = ["Title", "ProductName", "Brand", "Description", "LongDesc"]

EMBEDDING_MODEL = "all-mpnet-base-v2"  # Upgraded from all-MiniLM-L6-v2 for higher quality (768 dims)
RANDOM_SEED = 42
MAX_ROWS = None  # None = Full dataset (~500k rows), set to 50000 for quick testing
