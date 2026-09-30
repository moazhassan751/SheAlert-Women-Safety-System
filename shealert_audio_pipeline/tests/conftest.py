import sys
from pathlib import Path

# Add shealert_audio_pipeline root to sys.path
pipeline_root = Path(__file__).resolve().parent.parent
if str(pipeline_root) not in sys.path:
    sys.path.insert(0, str(pipeline_root))
