"""parallel_utils.py"""
import logging
from multiprocess.context import TimeoutError
from pathos import multiprocessing as mp
from tqdm import tqdm


def simple_parallel(
    input_list,
    function,
    max_cpu=16,
    timeout=4000,
    max_retries=3,
    override_cpu=None,
):
    """Simple parallelization.

    Use map async and retries in case we get odd stalling behavior.

    input_list: Input list to op on
    function: Fn to apply
    max_cpu: Num cpus
    timeout: Length of timeout
    max_retries: Num times to retry this

    """

    cpus = min(mp.cpu_count(), max_cpu)
    if override_cpu is not None:
        cpus = override_cpu
    pool = mp.Pool(processes=cpus)
    async_results = [pool.apply_async(function, args=(i,)) for i in input_list]
    pool.close()

    retries = 0
    while True:
        try:
            list_outputs = []
            for async_result in tqdm(async_results, total=len(input_list)):
                result = async_result.get(timeout)
                list_outputs.append(result)
            pool.join()
            break
        except TimeoutError:
            retries += 1
            logging.info(f"Timeout Error (s > {timeout})")
            pool.terminate()
            if retries <= max_retries:
                pool = mp.Pool(processes=cpus)
                async_results = [
                    pool.apply_async(function, args=(i,)) for i in input_list
                ]
                pool.close()
                logging.info(f"Retry attempt: {retries}")
            else:
                raise ValueError()

    return list_outputs


def chunked_parallel(
    input_list, function, chunks=100, max_cpu=16, timeout=4000, max_retries=3
):
    """chunked_parallel.

    Args:
        input_list : list of objects to apply function
        function : Callable with 1 input and returning a single value
        chunks: number of hcunks
        max_cpu: Max num cpus
        timeout: Length of timeout
        max_retries: Num times to retry this
    """

    # Adding it here fixes somessetting disrupted elsewhere

    def batch_func(list_inputs):
        outputs = []
        for i in list_inputs:
            outputs.append(function(i))
        return outputs

    list_len = len(input_list)
    num_chunks = min(list_len, chunks)
    step_size = len(input_list) // num_chunks + 1

    chunked_list = [
        input_list[i : i + step_size] for i in range(0, len(input_list), step_size)
    ]

    list_outputs = simple_parallel(
        chunked_list,
        batch_func,
        max_cpu=max_cpu,
        timeout=timeout,
        max_retries=max_retries,
    )
    # Unroll
    full_output = [j for i in list_outputs for j in i]

    return full_output
