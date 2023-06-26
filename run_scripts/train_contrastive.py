""" Train contrastive model.
"""
from mist import train_contrastive
import time

if __name__=="__main__": 
    start_time = time.time()
    train_contrastive.run_training()
    end_time = time.time()
    print(f"Program finished in: {end_time - start_time} seconds")
