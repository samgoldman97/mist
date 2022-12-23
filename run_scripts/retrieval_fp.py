""" retrieval_fp.py
"""
from mist import retrieval_fp
import time

if __name__=="__main__":
    start_time = time.time()
    retrieval_fp.run_retrieval()
    end_time = time.time()
    print(f"Program finished in: {end_time - start_time} seconds")
