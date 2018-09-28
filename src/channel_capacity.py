import math


def log2(x):
    return math.log(x, 2)


def discrete_entropy(prob):
    return sum(map(lambda p: -p * log2(p), prob))


# four symbols each with their prob. at source
px = [1/4] * 4

# prob. of correct transmission
p = 1021/1024
# prob. of incorrect transmission
q = 1/1024

# prob. of each symbol at Rx
py = [0.0] * 4

for j in range(4):
    for k in range(4):
        if j == k:  # at correct transmission
            py[j] += px[k] * p
        else:  # at incorrect transmission
            py[j] += px[k] * q

# while if fact
py = px.copy()


ent = -(p * log2(p) + 3 * q * log2(q))
info_lost = sum(map(lambda x: x*ent, py))
info_src = discrete_entropy(px)

print(info_lost)
print(info_src)
print(discrete_entropy([p, q, q, q]))
print(info_src - info_lost)
