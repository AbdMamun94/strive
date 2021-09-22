# Your functions
def main():
    """
    a = [2, 4, 6, 12, 15, 99, 100]
    100
    2
    4
    """
    return


a = [2, 4, 6, 12, 15, 99, 100]

max_number = max(a)
print (max_number)

min_number = min(a)
print (min_number)

div = []
for i in a :
    if i % 3 == 0:
       div.append(i)
print (len (div))