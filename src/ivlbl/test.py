from multiprocessing import Pool
import random

def change(x):
    
    x *= 2
    return x 

def y(n):
    pool = Pool(2)
    z = n
    for x in pool.imap(change, [1,2,3]):
        print x
        
if __name__=='__main__':
    y(5)