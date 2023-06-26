""" Embed smiles.

"""
from mist import embed_smis
import time

if __name__=="__main__":
    start_time = time.time()
    embed_smis.embed_smis()
    end_time = time.time()
    print(f"Program finished in: {end_time - start_time} seconds")
