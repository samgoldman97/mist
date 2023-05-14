""" run hyperopt.
"""
from mist import hyperopt_ffn_binned
import time

if __name__=="__main__":
    start_time = time.time()
    hyperopt_ffn_binned.run_hyperopt()
    end_time = time.time()
    print(f"Program finished in: {end_time - start_time} seconds")
