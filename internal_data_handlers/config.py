from pathlib import Path

# deprecated paths
# PROJECT_ROOT = Path(__file__).parent.parent
# DATASETS_DIR = PROJECT_ROOT / 'datasets'

# defining paths to datasets
SOCCER_FOLDER_DIR = Path(__file__).parent.parent.parent
DATASETS_DIR = SOCCER_FOLDER_DIR / 'data'
WYSCOUT_DATA = DATASETS_DIR / 'wyscout_data'
ESPN_DATA = DATASETS_DIR / 'espn_data'