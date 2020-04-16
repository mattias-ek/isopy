import isopy
import numpy as np
import datetime


if __name__ == "__main__":

    data = {'pd102': [1,2,3,4], 'ge74': [10,20,30,40]}

    array = isopy.IsotopeArray(data)
    array2 = np.array([5,6,7,8])
    array2 = [1,2,3,4]

    np.mean(array, axis=1)
    np.append(array,array2)
    np.concatenate(array)





    #print(isopy.IsotopeArray(d, ndim = 0))






