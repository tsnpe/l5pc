First round:

- 30k sims
- 35 features
- ensemble of 10
- path: 2022_04_13__19_32_02_snpe_m/1
- evaluation run: 2022_04_14__09_33_37__multirun/1__2022_04_13__19_32_02_snpe_m/1

R2:
- 30k sims
- 35 features
- ensemble of 10 (with joblib)
- path: 2022_04_18__15_16_19_snpe_m/0
- evaluation run: 2022_04_18__16_22_41__multirun/0__2022_04_18__15_16_19_snpe_m/0

R3:
- 30k sims
- 35 features
- ensemble of 10 (with joblib)
- path: 2022_04_19__10_27_01_snpe_m/0
- evaluation run: 2022_04_19__10_56_46__multirun/0__2022_04_19__10_27_01_snpe_m/0

R4:
- 30k sims
- 35 features
- ensemble of 10 (with joblib)
- path: 2022_04_19__20_55_09_snpe_m/0
- evaluation run: 2022_04_19__21_41_14__multirun/0__2022_04_19__20_55_09_snpe_m/0

R5:
- 30k sims
- 35 features
- ensemble of 10 (with joblib)
- path: 2022_04_20__10_27_58_snpe_m/0
- evaluation run: 2022_04_20__11_10_51__multirun/0__2022_04_20__10_27_58_snpe_m/0

R6:
- path: 2022_04_23__07_38_14_snpe/0

R7:
- 2022_04_24__07_47_31_snpe

R8:
- 2022_04_25__07_21_21_snpe









Run 2 (l20_2):
R1:
- 30k sims
- 35 features
- ensemble of 10
- path: previous_inference=2022_04_27__19_06_37_snpe
- evaluation run: 2022_04_27__23_41_19__2022_04_27__19_06_37_snpe


R2: 
- disregard the samples from the first round and train from scratch
- 30k sims
- 35 features
- ensemble of 10
- path: 2022_04_28__08_15_09_snpe
- evaluation run: 2022_04_28__10_57_24__multirun

R3:
- uses samples from second round (but not first) and does not train from scratch
- 30k sims
- 35 features
- ensemble of 10
- path: 2022_04_28__22_30_29_snpe_m/0
- evaluation run: 2022_04_29__08_16_57__multirun/0__2022_04_28__22_30_29_snpe_m/0

R4:
- started
- 30k sims
- 35 features
- ensemble of 10
- path: 2022_04_29__17_11_50_snpe_m or 2022_04_30__22_00_21_snpe_m/0

R5:
- path: 2022_05_01__12_02_16_snpe_m
- evaluation: 2022_05_01__17_19_46__multirun

R6:
- path: 2022_05_02__19_31_58_snpe_m
- evaluation: 




Run 3 (l20_3): SNPE-C with Flow trained in constrained space
R1:
- 30k sims
- 35 features
- ensemble of 10
- path: previous_inference=2022_04_27__19_06_37_snpe
- evaluation run: 2022_04_27__23_41_19__2022_04_27__19_06_37_snpe

R2:
Failed with only R2 data and train from scratch:
- 2022_04_29__08_52_03_snpe_m

R2:
Succeeded when using data from both rounds and continuing training from R1:
- ensemble size=1
- poor coverage
- 84% good simulations
- path: 2022_04_29__10_02_26_snpe
- evaluation path: 2022_04_29__11_01_35__multirun/0__2022_04_29__10_02_26_snpe










Run 4 (l20_4): SNPE-C with flow in unconstrained space and maf density estimator
R1:
- 30k sims
- 35 features
- ensemble of 1
- path: 2022_04_29__14_05_07_snpe
- evaluation path: 2022_04_29__14_34_57__multirun/0__2022_04_29__14_05_07_snpe
- 5% good simulations
- posterior log-prob 21.008

R2:
- performed on simultions from l20_3
- 30k sims
- 35 features
- ensemble of 1
- path: 2022_04_29__15_04_42_snpe_m/0
- evaluation path: 2022_04_29__16_16_33__multirun/0__2022_04_29__15_04_42_snpe_m/0
- could not sample. Out of 10 million, 0 samples were in the support









Run 5 (l20_5): SNPE-C with flow in unconstrained space and nsf density estimator
R1:
- 30k sims
- 35 features
- ensemble of 1
- path: 2022_04_29__14_15_25_snpe_m/0
- evaluation path: 2022_04_29__14_52_37__multirun/0__2022_04_29__14_15_25_snpe_m
- 9% good simulations
- posterior log-prob 27.699

R2:
- performed on simultions from l20_3
- 30k sims
- 35 features
- ensemble of 1
- path: 2022_04_29__15_04_28_snpe_m/0
- evaluation path: Out of 10 million, 0 samples were in the support










Run 6 (l20_6): SNPE-C with flow in constrained space and MAF
R1:
- 30k sims
- 35 features
- ensemble of 1
- path: 2022_05_01__15_02_01_snpe
- evaluation path: 2022_05_01__15_35_35__multirun


R2:
- path: 2022_05_01__15_17_31_snpe
- evaluation: 2022_05_01__15_55_05__multirun










Run 7 (l20_7):
R1:
path: 2022_04_27__19_06_37_snpe
evaluation: 2022_05_04__17_39_38__multirun

R2:
path: 2022_05_05__07_47_34_snpe_m/0
evaluation: 2022_05_05__11_00_41__2022_05_05__07_47_34_snpe_m

R3:
path: 2022_05_05__21_28_21_snpe
evaluation: 2022_05_05__23_42_37__multirun

R4:
path: 2022_05_06__13_50_43_snpe
evaluation: 2022_05_06__14_50_41__multirun

R5:
path: 
evaluation: 






=================================================================================
Pyloric

P31_1: pyloric net with nsf, forced to constrained space (i.e. ideal setup)

R1:
- 100k sims
- path: 2022_04_30__22_51_26_snpe_m/0
- evaluation: 2022_05_01__15_29_55__2022_04_30__22_51_26_snpe_m
- ensemble of 10

R2: 
- 50k sims
- path: 2022_05_01__15_15_50_snpe_m/0
- ensemble of 1

p31_2: pyloric net with maf, forced to constrained space

R1: 
path: 2022_05_01__17_13_52_snpe_m

p31_2:
R01-10: 2022_05_04__15_52_51__multirun
R11-20: 2022_05_04__23_08_47__multirun
R21-30: 2022_05_05__12_38_36__multirun


2022_05_06__13_08_05__multirun are the three NPE runs

2022_05_06__16_08_24__multirun TSNPE round 1-5