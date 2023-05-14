""" Run contrastive embedding.

"""
from mist import embed_contrast
import time

if __name__=="__main__": 
    start_time = time.time()
    embed_contrast.embed_specs()
    end_time = time.time()
    print(f"Program finished in: {end_time - start_time} seconds")
