"""
Script meant to be imported at the beginning of experimental notebooks.
Sets random seeds and moves the notebook's working directory to the project root.
"""
import os
import random
from pathlib import Path

import numpy as np
import torch


# Configure file system
notebook_path = Path(__file__).parent
project_root = notebook_path.parent
data_path = project_root / "data"
os.chdir(project_root)

# Configure PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(0)