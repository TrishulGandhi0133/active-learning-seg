import numpy as np

def margin_sampling(predictions, n_samples):
    margins = []
    for pred in predictions:
        sorted_probs = np.sort(pred)
        margin = sorted_probs[-1] - sorted_probs[-2]
        margins.append(margin)
    
    query_indices = np.argsort(margins)[:n_samples]
    return query_indices

if __name__ == "__main__":
    # This is a placeholder. Integrate this with the main script.
    pass
