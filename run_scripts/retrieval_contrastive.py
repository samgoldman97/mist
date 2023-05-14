""" Run retrieval hdf contrast

"""
from mist import retrieval_contrast
import time

if __name__=="__main__":
    start_time = time.time()
    retrieval_contrast.run_contrastive_retrieval()
    end_time = time.time()
    print(f"Program finished in: {end_time - start_time} seconds")
