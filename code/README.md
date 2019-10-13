## HOW TO USE iSplitLBI
THE REPRODUCIBILITY IS CHECKED ON A WINDOWS 10 SYSTEM WITH MATLAB R2018.

To run the experiment on your paired comparison dataset:
1. Enter `general` folder, then edit `load_data.m` to load your paired comparison data (e.g. `data/simulation/data.mat` file contains xxx paired comparisons). Each row of the data matrix should be arranged with `[user_id, item_id1, item_id2, comparison_label]`. For the comparison_label, `1` denotes that the user prefers item_id1 to item_id2; `-1` means the user prefers item_id2 to item_id1; and `0` stands for a tie.
2. Change `dataset_name` in `main.m` to your customized name. Set the parameters of iSplit LBI in `get_lbi_param.m` for the 'Micro-F1' or 'Macro-F1' metric.
3. Run `main.m` to get the results.
