import numpy as np
from numba import njit, int32, float64
from numba.experimental import jitclass


spec = [
    ('nodes', int32[:]),
    ('keys', float64[:]),
    ('size', int32),
]


@jitclass(spec)
class MinHeap:
    def __init__(self, n):
        self.nodes = np.empty(n, dtype=np.int32)
        self.keys = np.empty(n, dtype=np.float64)
        self.size = 0

    def push(self, node, key):
        i = self.size
        self.nodes[i] = node
        self.keys[i] = key
        self.size += 1

        while i > 0:
            p = (i - 1) // 2
            if self.keys[p] <= self.keys[i]:
                break

            tmp_node = self.nodes[i]
            tmp_key = self.keys[i]

            self.nodes[i] = self.nodes[p]
            self.keys[i] = self.keys[p]

            self.nodes[p] = tmp_node
            self.keys[p] = tmp_key

            i = p

    def pop(self):
        node = self.nodes[0]
        key = self.keys[0]

        self.size -= 1
        self.nodes[0] = self.nodes[self.size]
        self.keys[0] = self.keys[self.size]

        i = 0
        while True:
            l = 2 * i + 1
            r = 2 * i + 2
            s = i

            if l < self.size and self.keys[l] < self.keys[s]:
                s = l
            if r < self.size and self.keys[r] < self.keys[s]:
                s = r

            if s == i:
                break

            tmp_node = self.nodes[i]
            tmp_key = self.keys[i]

            self.nodes[i] = self.nodes[s]
            self.keys[i] = self.keys[s]

            self.nodes[s] = tmp_node
            self.keys[s] = tmp_key

            i = s

        return node, key