import numpy as np
from fastdtw import dtw

def compute_alingment(pho1, pho2):
    pho1 = np.array(pho1)
    pho2 = np.array(pho2)
    _, path = dtw(pho1, pho2)#, keep_internals = True)
    return path

def analyze_alingment(pho1, pho2):
    matches = []
    mismatches = []
    path = compute_alingment(pho1, pho2)
    for i, j in path:
        if pho1[i] == pho2[j]:
            matches.append((pho1[i], pho2[j]))
        else:
            mismatches.append((pho1[i], pho2[j]))
    return matches, mismatches

def calculate_confidence(pho1, pho2):
    path = compute_alingment(pho1, pho2)
    matches, mismatches = analyze_alingment(path, pho1, pho2)
    max_distance = max(len(matches), len(mismatches))
    confidence = 1 - (len(mismatches) / max_distance)
    return confidence