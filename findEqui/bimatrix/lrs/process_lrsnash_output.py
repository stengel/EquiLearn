import os
import sys
import json

file = 'lrs/lrsnash_output'

f= open(file, 'r')
x={}
i=1
number_of_equilibria = 0

for line in f.readlines():
    if 'Number of equilibria found' in line:
        number_of_equilibria = int(line.split()[4])
        break
    else:
        x[i] = line.split()
        i+=1

# store mixed strategies as arrays of string probabilities
e1 = {}
e2 = {}

# store payoffs
p1 = {}
p2 = {}


dict1 = {}
dict2 = {}

# store indices for mixed strategies for input to clique algorithm
index1 = {}
index2 = {}

# next index for input to clique algorithm
c1 = 1
c2 = 1

eq = -1 # array index of current equilibrium
# (shared by e1,e2,p1,p2,index1,index2)

count = 0 # how many equilibria of II to match with one

for j in range(2,len(x)):
    if not x[j]:
        count = 0 # reset count, ready for next set of II's strategies
        continue
    elif x[j][0] == "2":
        processII = True
        count += 1 # one more of II's strategies to pair with I's
        eq += 1
    elif x[j][0] == "1":
        processII = False

    l = len(x[j])

    if processII :
        e2[eq] = x[j][1:l-1]
        p1[eq] = x[j][l-1] # payoffs swapped in lrs output

        e2string = ','.join(e2[eq])

        if e2string not in dict2.keys():
            dict2[e2string] = c2
            c2 += 1
        index2[eq] = dict2[e2string]
    else:
        e1[eq] = x[j][1:l-1]
        p2[eq] = x[j][l-1] # payoffs swapped in lrs output

        e1string = ','.join(e1[eq])

        if e1string not in dict1.values():
            dict1[e1string] = c1
            c1 += 1
        index1[eq] = dict1[e1string]

        for i in range(1,count):
            e1[eq-i] = e1[eq]
            p2[eq-i] = p2[eq]
            index1[eq-i] = index1[eq]

result = []
for i in range(number_of_equilibria):
    result.append([{},{}])
    result[i][0]['number'] = index1[i]
    result[i][0]['distribution'] = e1[i]
    result[i][0]['payoff'] = p1[i]
    result[i][1]['number'] = index2[i]
    result[i][1]['distribution'] = e2[i]
    result[i][1]['payoff'] = p2[i]

with open('index_input', 'w') as file:
    file.write(json.dumps(result))

with open('clique/clique_input', 'w') as file:
    for i in range(number_of_equilibria):
        file.write("{0} {1}\n".format(index1[i],index2[i]))







