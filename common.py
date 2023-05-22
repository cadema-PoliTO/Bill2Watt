# -*- coding: utf-8 -*-
"""
Module 'common.py'
____________
DESCRIPTION
______
NOTES
Notation used in the variables
  - 'f' : tariff timeslot index, in range [1, nf]
  - 'j' : day-type index, in [0, n_j)
  - 'h' : time step index during one day, in range [0, n_h)
  - 'h' : time step index in the typical load profile, in range [0, n_i)
_____
INFO
Author : G. Lorenti (gianmarco.lorenti@polito.it)
Date : 20.05.2023
"""
# ----------------------------------------------------------------------------
# Libs, packages, modules
import numpy as np
import pandas as pd
# ----------------------------------------------------------------------------
# Common
# months of the year
nm = 12
ms = range(nm)
months = ['january', 'february', 'march', 'april', 'may', 'june',
          'july', 'august', 'september', 'october', 'november', 'december']
# ARERA's day-types depending on subdivision into tariff time-slots
# NOTE : f : 1 - tariff time-slot F1, central hours of work-days
#            2 - tariff time-slot F2, evening of work-days, and saturdays
#            3 - tariff times-lot F2, night, sundays and holidays
arera = pd.read_csv("Common\\arera.csv", sep=';', index_col=0).values
# total number and list of tariff time-slots (index f)
fs = list(np.unique(arera))
nf = len(fs)
# number of day-types (index j)
# NOTE : j : 0 - work-days (monday-friday)
#            1 - saturdays
#            2 - sundays and holydays
nj = np.size(arera, axis=0)
js = list(range(nj))
# number of time-steps during each day (index h)
nh = np.size(arera, axis=1)
# total number of time-steps (index i)
ni = arera.size
# reference profiles from GSE
y_ref_gse = pd.read_csv("Common\\y_ref_gse.csv", sep=';', index_col=0)
y_ref_gse = {m: row.values
             for m, row in y_ref_gse.set_index(['type', 'month']).iterrows()}

