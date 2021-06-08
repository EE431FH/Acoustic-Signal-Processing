#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import matplotlib.pyplot as plt
import numpy as np
import torch

class GetLoader(torch.utils.data.Dataset):
        def __init__(self, data_root, data_label, transform=None):
            self.data = data_root
            self.label = data_label
            self.transform = transform
        def __getitem__(self, index):
            data = self.data[index]
            labels = self.label[index]
            if self.transform:
               data = self.transform(data)
            return data, labels
        def __len__(self):
            return len(self.data)

def clistrain(value1, value2, value3, value4, value5, value6, value7, length):
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    list6 = []
    list7 = []
    for i in range(length):
        list1.append(value1)
        list2.append(value2)
        list3.append(value3)
        list4.append(value4)
        list5.append(value5)
        list6.append(value6)
        list7.append(value7)
    return list1, list2, list3, list4, list5, list6, list7

def clistest(value1, value2, value3, value4, value5, length):
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    for i in range(length):
        list1.append(value1)
        list2.append(value2)
        list3.append(value3)
        list4.append(value4)
        list5.append(value5)
    return list1, list2, list3, list4, list5
