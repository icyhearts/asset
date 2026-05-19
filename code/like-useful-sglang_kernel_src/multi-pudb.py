import multiprocessing as mp
import time

def worker(port):
    from pudb.remote import set_trace
    print(f"Process {port} waiting for debugger...")
    set_trace(port=port)
    print(f"Process {port} resumed")
    time.sleep(2)

if __name__ == "__main__":
    p1 = mp.Process(target=worker, args=(6899,))
    p2 = mp.Process(target=worker, args=(6900,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
