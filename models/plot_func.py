#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 20:33:31 2025

@author: sheng
"""

import matplotlib.pyplot as plt
from collections import deque

def moving_average(xs, k=50):
    if k <= 1 or len(xs) == 0:
        return xs
    out = []
    q = deque(maxlen=k)
    s = 0.0
    for v in xs:
        if len(q) == k:
            s -= q[0]
        q.append(v)
        s += v
        out.append(s / len(q))
    return out