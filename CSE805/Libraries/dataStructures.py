#always ensure to hava an empty new line after indented code

#tuples: read-only collections of items
a = (1, 2, 3)
print(a)

#list
mylist = [1, 2, 3,4]
print("Value: %d" % mylist[3])
mylist.append(99)
print(mylist)
print("Length of the list: %d" % len(mylist))
for element in mylist:
    print(element)

#dictionary: mapping of names to values like key value pairs
mydict1 = {'a': 1, 'b':2, 'c':3}
print("A value: %d" %mydict1['c'])
mydict1['b']=25
print("A value: %d" % mydict1['b'])
print("Keys: %s" % mydict1.keys())
print("values: %s" % mydict1.values())
for key in mydict1.keys():
    print(mydict1[key])

#functions
#sum function
def sum1(x, y):
    return x+y

#test sum function
result = sum1(1978, 900)
print(result)