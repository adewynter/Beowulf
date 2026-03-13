from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from scipy.stats import binom, chi2


def mcnemar_test(preds1, preds2, gts, continuity_correction=False, test_type="mid-p"):
    # Modified from https://github.com/kakumarabhishek/McNemar-mid-p
    # Statsmodels is hard to install in an arm64 windows machine because windows sucks

    preds1, preds2, gts = np.array(preds1), np.array(preds2), np.array(gts)
    try:
        assert preds1.shape == preds2.shape
        assert preds1.shape == gts.shape
    except AssertionError:
        print("Array shape mismatch.")
        return

    # This function supports the exact statistical test or the mid-p test.
    if test_type not in ["exact", "mid-p"]:
        print("'test_type' must be in ['exact', 'mid-p'].")
        return

    # Generate a binary array denoting which samples are correctly classified by each
    # model. This is useful because even in the N-class classification task, this step
    # ensures that the contingency matrix is 2x2.
    pred1_correct = 1 * (preds1 == gts)
    pred2_correct = 1 * (preds2 == gts)

    # Creating the contingency matrix.
    a = np.sum((pred1_correct == 1) & (pred2_correct == 1))
    b = np.sum((pred1_correct == 1) & (pred2_correct == 0))
    c = np.sum((pred1_correct == 0) & (pred2_correct == 1))
    d = np.sum((pred1_correct == 0) & (pred2_correct == 0))

    ct = np.array([[a, b], [c, d]])

    # Calculate the exact p-value.
    i: int = ct[0, 1]
    n: int = ct[1, 0] + ct[0, 1]
    i_n = np.arange(i + 1, n + 1)

    #p_value_exact = 2 * (1 - np.sum(binom.pmf(i_n, n, 0.5)))
    #pvalue = 2*binom.cdf(b, b + c, 0.5) - binom.pmf(b, b + c,  0.5)
    statistic = (b - c)**2 / (b + c)
    if continuity_correction: statistic = (np.abs(b - c) - 1)**2/(b + c)
    #pvalue = chi2.sf(statistic, 1)
    #if test_type == "exact":
    #    return p_value_exact, statistic, pvalue
    #else:
    #    mid_p_value = p_value_exact - binom.pmf(ct[0, 1], n, 0.5)
    #    return mid_p_value, statistic, pvalue

    # Here we borrow from wikipedia
    n_min, n_max = sorted([b, c])
    corr = int(continuity_correction)
    if (n_min + n_max) < 25:
        pvalue = 2 * binom.cdf(n_min, n_min+n_max, 0.5) - binom.pmf(n_min, n_min+n_max, 0.5)
    else:
        chi2_statistic = (abs(n_min - n_max) - corr) ** 2 / (n_min + n_max)
        pvalue = chi2.sf(chi2_statistic, 1)
    return statistic, pvalue


def compute_pairwise_mcnemar(data, models, sources, metric_label, models2readable, test_type="mid-p"):

    df = pd.DataFrame(data)

    print(f"\\textbf{{Model}} & \\textbf{{Experiment}} & $p$\\textbf{{-value}} & $\\Delta$ \\textbf{{\\% Acc.}} & $\\Delta$ \\textbf{{F$_1$}} \\\\ \\midrule")
    for model in models:
        for i in range(len(sources) - 1):
            source1 = sources[i]
            model_data = df[(df["model"] == model) & (df["source"] == source1)]
            before = model_data[metric_label].values
            ground_truth = model_data["ground_truth"].values
            _before= np.mean([a == p for a, p in zip(before, ground_truth)])*100

            old_min_len = min(len(before), len(ground_truth))
            _f1_before = f1_score(ground_truth[:old_min_len], before[:old_min_len])*100


            for j in range(i, len(sources)):
                if i == j: continue
                source2 = sources[j]
                model_data = df[(df["model"] == model) & (df["source"] == source2)]
                after = model_data[metric_label].values

                min_len = old_min_len
                if len(after) < old_min_len:
                    min_len = len(after)

                statistic, p_value = mcnemar_test(before[:min_len], after[:min_len], ground_truth[:min_len], test_type=test_type)
                _after = np.mean([a == p for a, p in zip(after[:min_len], ground_truth[:min_len])])*100
                relative_percent_increase = round(float((_after - _before)/_before), 2)
                abs_percent_increase = round(float((_after - _before)), 2)

                _f1_after = f1_score(ground_truth[:min_len], after[:min_len])*100
                f1_score_change = round(_f1_after - _f1_before, 2)
                # Print it as a latex table

                p_value_str = round(float(p_value), 2)
                relative_percent_increase_str = relative_percent_increase if relative_percent_increase < 0 else f"\\cellcolor{{blue!4}} {relative_percent_increase}"
                abs_percent_increase_str = abs_percent_increase if abs_percent_increase < 0 else f"\\cellcolor{{blue!4}} {abs_percent_increase}"
                f1_percent_increase_str = f1_score_change if f1_score_change < 0 else f"\\cellcolor{{blue!4}} {f1_score_change}"

                if p_value < 0.05:
                    p_value_str = f"\\cellcolor{{blue!4}} {p_value_str}"
                    #abs_percent_increase_str = f"\\cellcolor{{blue!4}} {abs_percent_increase}"
                    #f1_percent_increase_str = f"\\cellcolor{{blue!4}} {f1_percent_increase}"

                model_str = ""
                exp_str = f"{source1} -> {source2}"
                if i == 0 and j == 1:
                    model_str = f"\\textbf{{{models2readable[model]}}}"
                print(f"{model_str} & {exp_str} & {p_value_str} & {abs_percent_increase_str} & {f1_percent_increase_str} \\\\")


def compute_gwet_ac1(predictions, ground_truth):
    """
    Compute Gwet's AC1 coefficient for binary classification.
    --------
    """

    pred1 = np.asarray(predictions).flatten()
    ground_truth = np.asarray(ground_truth).flatten()
    
    try:
        assert pred1.shape == ground_truth.shape
    except AssertionError:
        print("Array shape mismatch.")
        return

    n = len(pred1)
    agreements = (pred1 == ground_truth).sum()
    po = agreements / n
    
    # Calculate the probability of chance agreement (Pe|AC1)
    p1_pos = np.sum(pred1 == 1) / n  # Proportion of positives in pred1
    p1_neg = np.sum(pred1 == 0) / n  # Proportion of negatives in pred1
    p2_pos = np.sum(ground_truth == 1) / n  # Proportion of positives in ground_truth
    p2_neg = np.sum(ground_truth == 0) / n  # Proportion of negatives in ground_truth
    
    # Calculate marginal probabilities
    pi_1 = (p1_pos + p2_pos) / 2  # Average proportion of positives
    pi_0 = (p1_neg + p2_neg) / 2  # Average proportion of negatives
    
    # Gwet's chance agreement
    pe_ac1 = 2 * pi_1 * pi_0
    
    # Calculate AC1
    if pe_ac1 == 1:
        ac1 = 1 if po == 1 else 0
    else:
        ac1 = (po - pe_ac1) / (1 - pe_ac1)
    
    return ac1


def compute_gwet_with_details(pred1, ground_truth):
    """
    Compute Gwet's AC1 with additional statistics. We don't really use this one
    """

    pred1 = np.asarray(pred1).flatten()
    ground_truth = np.asarray(ground_truth).flatten()
    
    try:
        assert pred1.shape == ground_truth.shape
    except AssertionError:
        print("Array shape mismatch.")
        return
    
    # Create confusion matrix
    tp = np.sum((pred1 == 1) & (ground_truth == 1))
    tn = np.sum((pred1 == 0) & (ground_truth == 0))
    fp = np.sum((pred1 == 1) & (ground_truth == 0))
    fn = np.sum((pred1 == 0) & (ground_truth == 1))
    
    n = len(pred1)
    # Calculate observed agreement
    po = (tp + tn) / n
    
    # Calculate marginal probabilities
    p1_pos = (tp + fp) / n
    p1_neg = (tn + fn) / n
    p2_pos = (tp + fn) / n
    p2_neg = (tn + fp) / n
    
    # Average proportions
    pi_1 = (p1_pos + p2_pos) / 2
    pi_0 = (p1_neg + p2_neg) / 2
    
    # Gwet's chance agreement
    pe_ac1 = 2 * pi_1 * pi_0

    if pe_ac1 == 1:
        ac1 = 1 if po == 1 else 0
    else:
        ac1 = (po - pe_ac1) / (1 - pe_ac1)
    
    if n > 1:
        var_ac1 = (po * (1 - po)) / (n * (1 - pe_ac1) ** 2)
        se_ac1 = np.sqrt(var_ac1)
    else:
        se_ac1 = np.nan
    return ac1
    return {
        'ac1': ac1,
        'observed_agreement': po,
        'chance_agreement': pe_ac1,
        'standard_error': se_ac1,
        'confusion_matrix': {
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        },
        'n_samples': n
    }
