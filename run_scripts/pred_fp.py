""" Predict fingerprints.
"""
from mist import pred_fp
import time

if __name__=="__main__":
    start_time = time.time()
    pred_fp.run_fp_pred()
    end_time = time.time()
    print(f"Program finished in: {end_time - start_time} seconds")
