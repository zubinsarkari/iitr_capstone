# iitr_capstone
Repository for AI/ML course from IIT Roorkee

## PART A

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

## PART B

## Model results (regression)

**Baseline.**
R² = -0.0212. Since the value of r2 is low, it means that the model does not explain any variability in the dependent variable

**Validation winner.**
LinearRegression won with RMSE = 4.8912 hours (R² 0.9313).

**Bias vs variance by depth** (from `part_b_tree_depth.csv`):
- High bias: depths <(1 to 4)>. Evidence: <11.6334 to 6.0999, 0.5 to 1.29>.
- High variance: depths <8,12, unlimited>. Evidence: <(3.336,1.36,0), (6.83,7.41,7.77), gap>.

**Learning rate** (from `part_b_gb_learning_rate.csv`):
- Best: 0.1 (4.91 , 0.93 ).
- 0.01: 10.0018,0.7127. Why it fails: After training 100 trees, since it only corrects 1% of the data. The model is able to explain about 72% of the variation rest everything is unexplained.
- 1.0: 7.3946,0.8429. Why it fails: <overshooting>.

**Cross-validation vs validation** (from `part_b_cv.csv`):
- CV: gradient_boosting 0.9352 ± 0.0044; linear_regression 0.9261 ± 0.0048.
- Do the rankings agree? No.
- Model I would ship and why: I would ship the gradient boosting model, cross validation r2 score is higher than the linear regression measured over 5 folds on 1020 rows.