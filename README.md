# iitr_capstone
Repository for AI/ML course from IIT Roorkee
# Data findings

All numbers below are copied from the output files in `outputs/`.

## 1. Most delayed service type
**Standard service type** is delayed most often, with a delay rate of **20%**
(132 of 642 shipments).
Source: `part_a_delay_by_group.csv`.

## 2. Delay rate from Tier 1 to Tier 3
Delay rate goes from **4%** in Tier 1 to **21.%** in Tier 2 and
**44%** in Tier 3
Source: `part_a_delay_by_group.csv`, rows `hub_tier / 1, 2, 3`.

## 3. Delay rate across distance bands
Delay rate moves from **2.2%** (0-200) to **13.6%** (201-400),
**43.6%** (401-600) and **85.2%** (601+), i.e. <one-line description of the trend>.
Source: `part_a_delay_by_group.csv`, rows `distance_band / ...`.

## 4. Columns most correlated with `delivery_hours`
Ranked by absolute correlation in `outputs/part_a_correlations.csv`
(`delivery_hours` itself excluded):

| Rank | Column | Correlation |
|------|--------|-------------|
| 1 | `promised_hours` | 0.8372 |
| 2 | `distance_km` | 0.4013 |
| 3 | `n_stops` | 0.3247 |

<Optional one sentence interpreting the sign and strength.>

## 5. Why `declared_value_inr` was log-transformed
Before the transform the mean was **1533.0542** and the median **1121.5**;
after it, the mean is **7.0277** and the median **7.0224** (filled column).
<Since np.log shrinks the larger value more than the smaller values, the numbers are replaced by their log values. Hence we see. such a difference between mean and median before and after log-transformed>