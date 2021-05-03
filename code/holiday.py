import datetime
import itertools
import numpy as np
import pandas as pd

# Holidays in Turkey
## National Holidays - non-changing date
holiday_non_changing_dates = [
#   (MM, DD)  # Explanation
    ( 1, 1 ), # New Years
    ( 4, 23), # National Sovereignty and Children's Day
    ( 5, 1 ), # Labor and Solidarity Day
    ( 5, 19), # Commemoration of Atatürk, Youth and Sports Day
    ( 7, 15), # Democracy and National Unity Day
    ( 8, 30), # Victory Day
    (10, 29)  # Republic Day
]
holiday_national = [
    datetime.date(year, *mm_dd)
    for year, mm_dd in itertools.product([2018,2019,2020,2021], holiday_non_changing_dates)
]

## Religious holidays - different each year
holiday_changing_dates = [
#   (Start, End) # Explanation
    ('2018-06-15', '2018-06-17'), # Ramadan Feast
    ('2018-08-20', '2018-08-24'), # Sacrifice Feast incl. Government Announced Extension
    ('2019-07-03', '2019-07-07'), # Ramadan Feast incl. Government Announced Extension
    ('2019-08-11', '2019-08-14'), # Sacrifice Feast
    ('2020-05-24', '2020-05-26'), # Ramadan Feast
    ('2020-07-31', '2020-08-03'), # Sacrifice Feast
]
for (start, end) in holiday_changing_dates:
    holiday_national += pd.date_range(start, end).date.tolist()

## half-day holidays
### Non-Changing
holiday_half_day = [
    datetime.date(year, *mm_dd)
    for year, mm_dd in itertools.product([2018,2019,2020,2021], [(10, 28)])
    # Republic Day Eve
]
### Changing
holiday_half_day += [
    datetime.date(*date) for date in [
        (2018, 6, 14), # Ramadan Feast Eve
        (2019, 8, 10), # Sacrifice Feast Eve
        (2020, 5, 23), # Ramadan Feast Eve
        (2020, 7, 30), # Sacrifice Feast Eve
    ]
]

# Function for creating a holiday column with 0 and 1 values
get_holiday = lambda df: df.assign(
    holiday_full_day = np.isin(df.index.date, holiday_national),
    holiday_half_day = np.isin(df.index.date, holiday_half_day) & (df.index.hour > 12),
    Holiday = lambda df: (df.holiday_full_day | df.holiday_half_day).astype(int)
) \
.drop(columns=['holiday_full_day', 'holiday_half_day'])
