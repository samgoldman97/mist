""" run hyperopt
"""
from mist import hyperopt_mist
import time

if __name__=="__main__":
    start_time = time.time()
    hyperopt_mist.run_hyperopt()
    end_time = time.time()
    print(f"Program finished in: {end_time - start_time} seconds")
