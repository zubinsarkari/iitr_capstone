# iitr_capstone
Repository for AI/ML course from IIT Roorkee
All numbers below are copied from the output files in `outputs/`.

## PART A

## 1. Most delayed service type
**Express service type** is delayed most often, with a delay rate of **48.97%**
(119 of 243 shipments).
Source: `part_a_delay_by_group.csv`.

## 2. Delay rate from Tier 1 to Tier 3
Delay rate goes from **4%** in Tier 1 to **21.5%** in Tier 2 and
**44%** in Tier 3
Source: `part_a_delay_by_group.csv`, rows `hub_tier / 1, 2, 3`.

## 3. Delay rate across distance bands
Delay rate moves from **2.2%** (0-200) to **13.6%** (201-400),
**43.6%** (401-600) and **85.2%** (601+) .
Source: `part_a_delay_by_group.csv`, rows `distance_band / ...`.

## 4. Columns most correlated with `delivery_hours`
Ranked by absolute correlation in `outputs/part_a_correlations.csv`
(`delivery_hours` itself excluded):

| Rank | Column | Correlation |
|------|--------|-------------|
| 1 | `promised_hours` | 0.8372 |
| 2 | `distance_km` | 0.4013 |
| 3 | `n_stops` | 0.3247 |


## 5. Why `declared_value_inr` was log-transformed
Before the transform the mean was **INR 1,533.05** and the median **INR 1,121.50**;
after it, the mean is **7.0277** and the median **7.0224** (filled column).
Since np.log shrinks the larger value more than the smaller values, the numbers are replaced by their log values. The mean was far above the median (right skew); after the log they are almost equal, so the distribution is more symmetric.

## PART B

**Baseline.**
R² = -0.0212 means it predicts the training mean for every parcel, so it explains none of the variation.

**Validation winner.**
LinearRegression won with RMSE = 4.8912 hours (R² 0.9313).

**Bias vs variance by depth** (from `part_b_tree_depth.csv`):
- High bias: depths 1 to 4. Train and val RMSE are both high and close together (depth 1: 11.6334 vs 12.1324; depth 4: 6.0999 vs 7.3896).
- High variance: depths 8, 12, unlimited. Train RMSE falls (3.3360, 1.3613, 0.0) while val RMSE rises (6.8385, 7.4173, 7.7890), so the gap widens.

**Learning rate** (from `part_b_gb_learning_rate.csv`):
- Best: 0.1 (val_rmse 4.9165, val_r2 0.9306). It has the lowest error and the highest R² of the three.
- 0.01: val_rmse 10.0018, val_r2 0.7127. Why it fails: it is too slow. Each of the 100 trees adds only 1% of its correction, so the fit is still incomplete after 100 trees. The model explains about 71% of the variation in `delivery_hours`, far less than the 93% at 0.1.
- 1.0: val_rmse 7.3946, val_r2 0.8429. Why it fails: it overshoots. Each tree's full correction is added with no damping, so the model overreacts to noise in the training rows and generalises worse, explaining about 84% of the variation.

**Cross-validation vs validation** (from `part_b_cv.csv`):
- CV: gradient_boosting 0.9352 ± 0.0044; linear_regression 0.9261 ± 0.0048.
- Do the rankings agree? No.
- Model I would ship and why: I would ship the gradient boosting model, cross validation r2 score is higher than the linear regression measured over 5 folds on 1,020 rows.


## PART C

**Blind baseline** (`part_c_models.csv`, row `blind_all_on_time`)
- Accuracy = 0.7667 (TN 138, FN 42, TP 0, FP 0).
- Why it is useless: It predicts "not delayed" for all the parcels. It catches none of the 42 delayed parcels and 77% of the parcels are on time delivery so accuracy hides the failure.

**Noise columns** (`part_c_feature_scores.csv`)
- Combined `rf_importance` of the five noise columns: 0.225.
- Top 7 by `f_rank`: distance_km,hub_tier,hub_daily_capacity,n_stops,promised_hours,rain_flag,sale_event_flag.
- Do all five noise columns fall outside the top 7? Yes. Evidence: their `f_rank` values are Rank 8: label_print_batch, Rank 12 : tape_roll_no, Rank 14: sticker_serial_mod, Rank 16: shelf_slot_no, Rank 17:scanner_build_no. 5 columns that carry no signal still receive about a fifth of the forest’s importance, so forest importance alone would be misleading.

**Number of features `k`** (`part_c_selectk.csv`)
- `val_f1` for k = 5, 7, 10, 12, 17: 0.6988, 0.8333, 0.814, 0.8049, 0.8
- What the table shows: k = 7 gives the highest val_f1 (0.8333). Keeping all 17 columns scores lower (0.8), so dropping the noise columns costs nothing and helps slightly. k = 5 drops to 0.6988, so cutting below 7 removes real signal.

**Tuning vs untuned forest** (`part_c_tuning.csv`)
- `val_f1`: grid 0.775, randomized 0.7901, default forest 0.8333.
- Did tuning beat the untuned forest? No.
- What this says about tuning on a 180-row validation set: the untuned forest (0.8333) beat grid (0.775) and randomized (0.7901), so the 135 + 45 extra fits bought no gain. The validation set has only 42 delayed parcels, so a difference of a few parcels moves F1 by several points. That means the ranking of these three models is not reliable, and a tuned model chosen on such a small set can look worse (or better) by chance. Tuning only helps if the gain is larger than this noise.

**Alert threshold** (`part_c_thresholds.csv`)
- Chosen threshold: 0.5, the highest `t` with validation recall >= 0.80 and at 0.6 recall drops to 0.7619, so it fails the rule.
- Trade-off: at 0.5 validation recall is 0.8333 and precision is 0.7955; at the next higher threshold recall drops to 0.7619 (below 0.80) while precision rises to 0.8889.

**Final test result** (`part_c_final_test.csv`)
- Precision 0.6889, recall 0.7561, F1 0.7209 at threshold 0.5.
- Test vs validation recall: 0.7561 vs 0.8333. Is test recall below 0.80? : Yes.
- Honest reading: The test set has only 41 delayed parcels, so one parcel changes recall by about 0.024. The drop from 0.8333 to 0.7561 is about 3 parcels, which is within the range expected from sampling noise, so I would not read it as evidence that the model got worse. The 0.80 target was still not reached on the test rows.