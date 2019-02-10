import isopy

class a:
    def __getitem__(self, **item):
        print(item.keys())

if __name__ == "__main__":
    a = isopy.dtypes.IsotopeArray([[1,2,3],[4,5,6]], keys =['105pd', '106pd'])
    b = isopy.dtypes.IsotopeArray([2,1,3], keys=['ru101', 'pd105', '106pd'])
    c = {'105pd': 1.5, '106Pd': 2}
    d = isopy.dtypes.IsopyDict(c)

    print(a)
    print(a*d)
