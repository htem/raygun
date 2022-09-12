import daisy
import logging
from time import sleep
logging.basicConfig(level=logging.INFO)

def base_worker(options):
    logger = logging.getLogger('worker')
    client = daisy.Client()
    worker_id = client.worker_id
    logger.info(f"I am {worker_id} with options {options}")
    while True:
        with client.acquire_block() as block:
            if block is None:
                break
            sleep(0.01)

if __name__ == '__main__':
    task = daisy.Task(
        'DummyTask',
        daisy.Roi((0,), (1000,)),
        daisy.Roi((0,), (10,)),
        daisy.Roi((1,), (8,)),
        process_function=lambda: base_worker("[fancy options]"),
        num_workers=23
    )
    daisy.run_blockwise([task])
    