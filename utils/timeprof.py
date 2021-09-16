import time

def tick():
    return time.time()

def tock(st):
    delay = time.time()-st
    return delay

if __name__=="__main__":
    x=tick()
    time.sleep(2)
    y=tock(x)
    print(y)