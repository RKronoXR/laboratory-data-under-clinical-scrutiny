"""Recorded input and geometry helpers used by the portable benchmark."""
from pathlib import Path
import pandas as pd
def run_dir(project, arch, seed, fold):
    return Path(project)/'runs'/arch/f'seed_{seed}'/f'fold_{fold}'

def read_frame(path):
    return pd.read_csv(path, dtype={'sample_id':str,'bitewing_id':str,'group_id':str},keep_default_na=False, float_precision='round_trip')
