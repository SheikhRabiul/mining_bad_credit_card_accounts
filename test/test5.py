#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 23:44:44 2017

@author: sheikh
"""
def scale_a_number(inpt, to_min, to_max, from_min, from_max):
    return (to_max-to_min)*(inpt-from_min)/(from_max-from_min)+to_min

def scale_a_list(l, to_min, to_max):
    return [scale_a_number(i, to_min, to_max, min(l), max(l)) for i in l]

print(scale_a_list([1,3,4,5], 0, 1))