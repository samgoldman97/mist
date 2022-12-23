""" run hyperopt
"""
from mist.forward import ffn_hyperopt
import time

if __name__=="__main__":
    start_time = time.time()
    ffn_hyperopt.run_hyperopt()
    end_time = time.time()
    print(f"Program finished in: {end_time - start_time} seconds")
