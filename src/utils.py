"""
utils.py
Helper functions for the project.
"""

import joblib
import numpy as np
import random
import os

def save_model(model, path):
	"""Save a model to disk using joblib."""
	joblib.dump(model, path)

def load_model(path):
	"""Load a model from disk using joblib."""
	return joblib.load(path)

def save_results(results, path):
	"""Save results (dict) to a CSV file."""
	import pandas as pd
	pd.DataFrame([results]).to_csv(path, index=False)

def set_seed(seed=42):
	"""Set random seed for reproducibility."""
	np.random.seed(seed)
	random.seed(seed)
	try:
		import torch
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	except ImportError:
		pass
