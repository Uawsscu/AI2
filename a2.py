import random
import csv


def select(fs):
    p = random.uniform(0, sum(fs))
    for i, f in enumerate(fs):
        p -= f
        if p <= 0:
            break
    return i


prob = 0.2
V = ['a', 'b']
E = []
E.append(['a', 'b', 50.4, 0.5])  # เส้นทาง1
E.append(['a', 'b', 30.5, 0.5])  # เส้นทางที่2
E.append(['a', 'b', 100.0, 0.5])
E.append(['a', 'b', 70.5, 0.5])
E.append(['a', 'b', 500, 0.5])
solution = 1

p = [0, 0, 0, 0, 0]
N = 5
ant = []
loop = 500

sump = 0
for i in range(len(E)):
    sump = sump + E[i][3]

fp = open('test1.csv', 'w')
write = csv.writer(fp, delimiter=',')

for iter in range(loop):
    ant = []
    for i in range(N):
        for j in range(len(E)):
            p[j] = E[j][3] / sump
        choice = select(p)
        ant.append(choice)

    #################
    ok = 0.0
    for path in ant:
        if path == solution:
            ok += 1
    print iter, ok / N * 100
    write.writerows([[ok / N * 100]])
    #################

    for i in range(len(E)):
        deposit = 0.0
        for edge in ant:
            if edge == i:
                deposit += 1.0 / E[i][2]
        E[i][3] = (1.0 - prob) * E[i][3] + deposit * prob




