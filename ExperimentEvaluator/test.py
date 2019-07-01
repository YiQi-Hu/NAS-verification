import sys

for line in sys.stdin:
    for value in line.split():
        print(int(value))