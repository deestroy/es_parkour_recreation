# ES-Parkour recreation - evaluation results

Level 3, 32 episodes/terrain.

## Success rates (cf. paper Fig. 5)

| Terrain | Teacher ANN | Student SNN |
|---|---|---|
| gap | 100% | 0% |
| step | 100% | 19% |
| hurdle | 100% | 0% |
| parkour | 94% | 0% |

## Joint motor energy per episode (cf. Table IV)

| Terrain | Teacher (J) | Student (J) |
|---|---|---|
| gap | 7805.6 | 1102.5 |
| step | 8127.4 | 5478.3 |
| hurdle | 8296.1 | 5163.2 |
| parkour | 7979.1 | 4156.6 |

## Operations & theoretical energy (cf. Tables II-III)

| Module | SNN FLOPs | SNN SOPs | ANN FLOPs | OPs(SNN):OPs(ANN) | SNN mJ | ANN mJ | Saving |
|---|---|---|---|---|---|---|---|
| encoder (11.19M) | 2.57e+07 | 7.43e+07 | 1.45e+08 | 0.69 : 1 | 0.185 | 0.6663 | 72.2% |
| actor (0.23M) | 0 | 5.28e+04 | 2.32e+05 | 0.23 : 1 | 4.755e-05 | 0.001067 | 95.5% |
