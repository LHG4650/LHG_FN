import time

def time_stamp():
    time_stamp = time.strftime('%y.%m.%d_%H.%M.%S',time.localtime())
    return time_stamp