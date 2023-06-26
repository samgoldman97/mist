""" run hyperopt.
"""
from mist import hyperopt_contrastive
import time

if __name__=="__main__":
    start_time = time.time()
    hyperopt_contrastive.run_hyperopt()
    end_time = time.time()
    print(f"Program finished in: {end_time - start_time} seconds")
