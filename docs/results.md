# ES-Parkour recreation - evaluation results

Level 6, 32 episodes/terrain.

## Success rates (cf. paper Fig. 5)

| Terrain | Teacher ANN | Student SNN |
|---|---|---|
| gap | 34% | 0% |
| step | 91% | 0% |
| hurdle | 97% | 0% |
| parkour | 0% | 0% |

## Joint motor energy per episode (cf. Table IV)

| Terrain | Teacher (J) | Student (J) |
|---|---|---|
| gap | 5739.9 | 1041.3 |
| step | 9134.9 | 3874.7 |
| hurdle | 9038.8 | 5489.7 |
| parkour | 1825.6 | 2378.5 |

## Operations & theoretical energy (cf. Tables II-III)

| Module | SNN FLOPs | SNN SOPs | ANN FLOPs | OPs(SNN):OPs(ANN) | SNN mJ | ANN mJ | Saving |
|---|---|---|---|---|---|---|---|
| encoder (11.19M) | 2.57e+07 | 7.29e+07 | 1.45e+08 | 0.68 : 1 | 0.1838 | 0.6663 | 72.4% |
| actor (0.23M) | 0 | 5.32e+04 | 2.32e+05 | 0.23 : 1 | 4.792e-05 | 0.001067 | 95.5% |
