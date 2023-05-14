""" Train forward model.
"""
from mist.forward import ffn_train
import time

if __name__=="__main__":
    start_time = time.time()
    ffn_train.train_model()
    end_time = time.time()
    print(f"Program finished in: {end_time - start_time} seconds")
