from collections import Counter

elements = ['apple', 'orange', 'banana', 'apple', 'orange', 'banana', 'banana']
count = Counter(elements)

print(count)  # Output: Counter({'banana': 3, 'apple': 2, 'orange': 2})


from collections import defaultdict

counts = defaultdict(int)
fruits = ['apple', 'banana', 'apple', 'orange', 'banana']

for fruit in fruits:
    counts[fruit] += 1

print(counts)  # Output: defaultdict(<class 'int'>, {'apple': 2, 'banana': 2, 'orange': 1})


from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])
p = Point(10, 20)

print(p.x, p.y)  # Output: 10 20
