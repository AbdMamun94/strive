import numpy as np
def moving_average(a,n) : 

    test = np.cumsum(a, dtype=float) 
    test[n:] = test[n:] - test[:-n]
    
    return test[n-1:] /n
    
res=moving_average([1,2,3,4,5,6,7,8,9,10] ,5 ) 
print(res)