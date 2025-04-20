from pathlib import Path

class Config:

    PROJECT_ROOT = Path(__file__).parent
    MODEL_PATH = PROJECT_ROOT / 'models' / 'best.pt'
    VIDEO_SOURCE = PROJECT_ROOT / 'data' / 'gen_fire.mp4'
