# %%
import numpy as np

# %%

def l1_distance(x : np.ndarray, y : np.ndarray) -> np.ndarray:
    return np.sum(np.abs(x - y))

def l2_distance(x : np.ndarray, y : np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.power(x - y, 2)))

def l2_squared_distance(x : np.ndarray, y : np.ndarray) -> np.ndarray:
    return np.sum(np.power(x - y, 2))

def lp_distance(x : np.ndarray, y : np.ndarray, p : int = 2) -> np.ndarray:
    return np.power(np.sum(np.power(x - y, p)), 1/p)

def inf_distance(x : np.ndarray, y : np.ndarray) -> np.ndarray:
    return np.max(np.abs(x - y))

# %%
