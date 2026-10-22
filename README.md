# iitr_capstone
Repository for AI/ML course from IIT Roorkee
# Data findings

All numbers below are copied from the output files in `outputs/`.

## 1. Most delayed service type
**<Service type>** is delayed most often, with a delay rate of **<0.0000>**
(<delayed> of <shipments> shipments).
Source: `<file name>`, row `service_type / <group>`.

## 2. Delay rate from Tier 1 to Tier 3
Delay rate goes from **<0.0000>** in Tier 1 to **<0.0000>** in Tier 2 and
**<0.0000>** in Tier 3, i.e. <rises / falls> <steadily / not steadily> with tier.
Source: `<file name>`, rows `hub_tier / 1, 2, 3`.

## 3. Delay rate across distance bands
Delay rate moves from **<0.0000>** (0-200) to **<0.0000>** (201-400),
**<0.0000>** (401-600) and **<0.0000>** (601+), i.e. <one-line description of the trend>.
Source: `<file name>`, rows `distance_band / ...`.

## 4. Columns most correlated with `delivery_hours`
Ranked by absolute correlation in `outputs/part_a_correlations.csv`
(`delivery_hours` itself excluded):

| Rank | Column | Correlation |
|------|--------|-------------|
| 1 | `<column>` | <±0.0000> |
| 2 | `<column>` | <±0.0000> |
| 3 | `<column>` | <±0.0000> |

<Optional one sentence interpreting the sign and strength.>

## 5. Why `declared_value_inr` was log-transformed
Before the transform the mean was **<value>** and the median **<value>**;
after it, the mean is **<value>** and the median **<value>** (filled column).
<One sentence: the large gap between mean and median shows right skew from a few very
large values, and the log brings them close together / makes the distribution more symmetric.>