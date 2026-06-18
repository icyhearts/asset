import threading

counter = 0
lock = threading.Lock()

def worker():
    global counter
    for _ in range(100000):
        with lock:
            counter += 1

threads = []
for _ in range(40):
    t = threading.Thread(target=worker)
    t.start()
    threads.append(t)

for t in threads:
    t.join()

print(counter)
