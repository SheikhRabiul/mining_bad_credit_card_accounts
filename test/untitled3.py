# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 00:50:54 2017

@author: SheikhRabiul
"""

import time
import datetime
date1 = "2015-12-31"
date2 = "2016-01-01"

newdate1 = time.strptime(date1, "%Y-%m-%d")

newdate2 = time.strptime(date2, "%Y-%m-%d")

if newdate1 > newdate2:
    print("1")
else:
    print("2")

payment_due_date=str('2017-07-01')



now = str(datetime.datetime.now())

#today = str(datetime.date.today());
curr_year = str(now[0:4]);
print(curr_year)

curr_month = str(now[4:2]);
print(curr_month)

#import datetime
now=str(datetime.datetime.now())
yr_mn=  now[0:7]
print(yr_mn)
