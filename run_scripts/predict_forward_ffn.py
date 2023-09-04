""" Predict forward specs with ffn.
"""
from mist.forward import ffn_predict
import time

if __name__=="__main__": 
    start_time = time.time()
    ffn_predict.predict()
    end_time = time.time()
    print(f"Program finished in: {end_time - start_time} seconds")
