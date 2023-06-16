# -*- coding: utf-8 -*-
"""
DESCRIPTION
------------
This module contains all the common variables to the modules of the package.

NOTES
------
Notation used in the variables
  - 'm': month of the year index, in range [1, m]
  - 'f' : time-of-use tariff timeslot index, in range [1, nf]
  - 'j' : day-type index, in [0, nj)
  - 'h' : time step index during one day, in range [0, nh)
  - 'i' : time step index in the typical load profile, in range [0, ni)

INFO
----
Author : G. Lorenti (gianmarco.lorenti@polito.it)
"""

import numpy as np
import pandas as pd
from os import path

# Path for the folder of common data
basepath = path.dirname(path.abspath(__file__))
folder = path.join(basepath, "data")

# Input data

# ARERA's day-types depending on subdivision into tariff time-slots
arera = pd.read_csv(path.join(folder, "arera.csv"), sep=';', index_col=0)\
    .values

# Length of the time step for calculation (in hours)
dt = 1

# Months of the year (index m)
nm = 12
ms = range(nm)
months = ['january', 'february', 'march', 'april', 'may', 'june',
          'july', 'august', 'september', 'october', 'november', 'december']

# Number and list of tariff time-slots (index f)
fs = list(np.unique(arera))
nf = len(fs)

# Number of day-types (index j)
# NOTE : j : 0 - work-days (monday-friday)
#            1 - saturdays
#            2 - sundays and holydays
nj = np.size(arera, axis=0)
js = list(range(nj))
day_types = ['work-day', 'saturday', 'holiday']

# Number of time-steps during each day (index h)
nh = np.size(arera, axis=1)

# Number of time-steps in the typical load profiles (index i)
ni = arera.size
