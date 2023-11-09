import datetime 
import timeit

# clock function 
def clock(func):
    def clocked(*args, **kwargs):
        name = func.__name__
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'----------------------------Start {name:s} at {now:s}. -------------------------------') 
        t0 = timeit.default_timer()
        result = func(*args, **kwargs)
        elapsed = timeit.default_timer() - t0
        print(f'----------------------------Finish {name:s} in {elapsed: .6f}s! -------------------------------') 
        return result
    return clocked