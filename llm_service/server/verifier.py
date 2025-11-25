import numpy as np
import random

def accept_or_resample(draft_id, x, y, eos_token_id=None):
    """
    x, y: numpy arrays of probs (same length)
    returns (accepted, token_id)
    """
    x_d, y_d = x[draft_id], y[draft_id]
    if eos_token_id is not None and draft_id == eos_token_id:
        print(f"[VERIFIER DEBUG] Force-accepting EOS token: draft_id={draft_id}, eos_token_id={eos_token_id}")
        return True, draft_id
    print(f"[VERIFIER DEBUG] Non-EOS token: draft_id={draft_id}, eos_token_id={eos_token_id}, x_d={x_d:.4f}, y_d={y_d:.4f}")
    if x_d <= y_d:
        return True, draft_id

    # reject with prob 1 - y_d / x_d
    if random.random() < y_d / x_d:
        return True, draft_id
    else:
        delta = np.maximum(y - x, 0)
        if np.sum(delta) == 0:
            # fallback uniform
            return False, int(np.argmax(y))
        delta /= np.sum(delta)
        token_id = int(np.random.choice(len(delta), p=delta))
        return False, token_id
