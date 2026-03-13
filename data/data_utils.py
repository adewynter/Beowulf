import json
import pandas as pd
import numpy as np

from evaluation_utils import compute_gwet_ac1

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from collections import defaultdict

from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


def consolidate_files_all_criteria(locale: str, models: list, shots: list=[0, 5, 20, 40], how_many: int=None, 
    root_dir: str='icl_predictions'):
    """
    Load baseline datasets per locale, for all models and shots specified. 
    If you have `how_many` set, it will load baseline starting at that index.
    """
    consolidated = []
    for model in models:
        for shot in shots:
            ofile = f"{root_dir}/{locale}/dev-{model}_{shot}_all_crit.json"
            baseline = [json.loads(l) for l in open(ofile, "r", encoding="utf-8").readlines()]
            if how_many is not None:
                baseline = baseline[how_many:]
            counter = 0
            for l in baseline:
                nbr_fail = l["NoBreakdown"]["failed"]
                br_fail = l["BreakdownReasons"]["failed"]
                bnr_fail = l["BreakdownNoReasons"]["failed"]

                # Phi-4 in particular was sneaky here
                bnr_label = 0
                if "Label" in l["BreakdownNoReasons"]["response"]:
                    bnr_label = l["BreakdownNoReasons"]["response"]["Label"]
                    if bnr_label not in [0, 1]:
                        bnr_label = 0
                        bnr_fail = True
                br_label = 0
                if "Label" in l["BreakdownReasons"]["response"]:
                    br_label = l["BreakdownReasons"]["response"]["Label"]
                    if br_label not in [0, 1]:
                        br_label = 0
                        br_fail = True
                nbr_label = 0
                if "Label" in l["NoBreakdown"]["response"]:
                    nbr_label = l["NoBreakdown"]["response"]["Label"]
                    if nbr_label not in [0, 1]:
                        nbr_label = 0
                        nbr_fail = True

                consolidated.append(
                    {
                    "index": l["Index"] if "Index" in l else (l["Prompt"], l["Output"]),
                    "shots": shot,
                    "model": model,
                    "source": l["Source"],
                    "no_breakdown": nbr_label if not nbr_fail else 0,
                    "breakdown_no_reasons": bnr_label if not bnr_fail else 0,
                    "breakdown_reasons": br_label if not br_fail else 0,
                    "fail_no_breakdown": nbr_fail,
                    "fail_breakdown_no_reasons": bnr_fail,
                    "fail_breakdown_reasons": br_fail,
                    "ground_truth": l["Label"]
                    }
                )
                counter += 1
    return consolidated



def consolidate_files_per_criteria(locale: str, models: list, shots: list, crit: str, how_many: int=None, 
    root_dir: str='icl_predictions', exclude_reasons: bool=False):
    """
    Load all files per crit. Identical to the other function, but with a 'gt_crit' which will comprise the ground truth label.
    This means you will have to reload it on every crit, and if computing aggregate labels (gt_crit == "aggregate"). 
    If you have `how_many` set, it will load baseline starting at that index.
    """
    consolidated = []
    CRIT_NAMES = ["c1", "c2a", "c2b", "c3", "c4", "c5"]

    assert crit in CRIT_NAMES + ["aggregate"], f"gt_crit must be one of: {CRIT_NAMES + ['aggregate']}"

    for model in models:
        for shot in shots:
            ofile = f"{root_dir}/{locale}/dev-{model}_{shot}_per_crit.json"
            baseline = [json.loads(l) for l in open(ofile, "r", encoding="utf-8").readlines()]
            if how_many is not None:
                baseline = baseline[how_many:]

            counter = 0
            for l in baseline:

                if crit != "aggregate":

                    nr_fail = l["NoReasons"][crit]["failure"]
                    r_fail = l["Reasons"][crit]["failure"] if not exclude_reasons else False

                    # Phi-4 in particular was sneaky here
                    nr_label = 0
                    if "response" in l["NoReasons"][crit]:
                        nr_label = l["NoReasons"][crit]["response"]
                        if nr_label not in [0, 1]:
                            nr_label = 0
                            nr_fail = True
                    r_label = 0
                    if not exclude_reasons:
                        if "response" in l["Reasons"][crit]:
                            r_label = l["Reasons"][crit]["response"]
                            if r_label not in [0, 1]:
                                r_label = 0
                                r_fail = True

                    consolidated.append(
                        {
                        "index": l["Index"] if "Index" in l else (l["Prompt"], l["Output"]),
                        "shots": shot,
                        "model": model,
                        "source": l["Source"],
                        "reasons": r_label if not r_fail else 0,
                        "no_reasons": nr_label if not nr_fail else 0,
                        "fail_reasons": r_fail,
                        "fail_no_reasons": nr_fail,
                        "ground_truth": l["Rubric"][crit]
                        }
                    )

                else:
                    r_labels = []
                    nr_labels = []
                    r_failure = False
                    nr_failure = False

                    for crit_name in CRIT_NAMES:

                        # Gather no reasons label
                        nr_fail = l["NoReasons"][crit_name]["failure"]
                        if nr_fail or nr_failure:
                            nr_failure = True
                            nr_labels.append(0)
                        else:
                            nr_label = 0
                            if "response" in l["NoReasons"][crit_name]:
                                nr_label = l["NoReasons"][crit_name]["response"]
                                if nr_label not in [0, 1]:
                                    nr_label = 0
                                    nr_failure = True
                            nr_labels.append(nr_label)

                        # Same for reasons
                        if exclude_reasons: continue
                        r_fail = l["Reasons"][crit_name]["failure"]
                        if r_fail or r_failure:
                            r_failure = True
                            r_labels.append(0)
                        else:
                            r_label = 0
                            if "response" in l["Reasons"][crit_name]:
                                r_label = l["Reasons"][crit_name]["response"]
                                if r_label not in [0, 1]:
                                    r_label = 0
                                    r_failure = True
                            r_labels.append(r_label)

                    r_label = 1 if sum(r_labels) == 6 else 0
                    nr_label = 1 if sum(nr_labels) == 6 else 0

                    consolidated.append(
                        {
                        "index": l["Index"] if "Index" in l else (l["Prompt"], l["Output"]),
                        "shots": shot,
                        "model": model,
                        "source": l["Source"],
                        "reasons": r_label if not r_failure else 0,
                        "no_reasons": nr_label if not nr_failure else 0,
                        "fail_reasons": r_failure,
                        "fail_no_reasons": nr_failure,
                        "ground_truth": l["Label"]
                        }
                    )
                    
                counter += 1
    return consolidated


def plot_accuracy(consolidated_dataset: list, metric_label: str, locale: str, 
                  MODELS: list, COLOURS: list, model2readable: dict, title_suff: str, save_filename: str=None, 
                  do_failures=True, return_full_results: bool=False, no_plot: bool=False):
    """
    Accuracy/Percentage barplots.

    Params:
    - consolidated_dataset: the parsed, consolidated files from `consolidate_files(locale)`
    - metric_label: one of [no_breakdown, breakdown_no_reasons, breakdown_reasons]
    - locale: one of []
    - MODELS: the models plotted
    - COLOURS: vanity
    - models2readable: a map to niceify the model names
    - title_suff: what are you plotting lol
    - save_filename: if not None, will be the prefix for the filename
    """
    df = pd.DataFrame(consolidated_dataset)
    
    # Calculate metrics for each model
    results = []
    for model in MODELS:
        model_data = df[df['model'] == model]
        if len(model_data) > 0:
            accuracy = accuracy_score(model_data['ground_truth'], model_data[metric_label]) * 100
            f1 = f1_score(model_data['ground_truth'], model_data[metric_label]) * 100
            gwet = compute_gwet_ac1(model_data[metric_label], model_data['ground_truth'])
            fail_percentage = model_data[f'fail_{metric_label}'].mean() * 100
            results.append({
                'model': model2readable[model],
                'accuracy': accuracy,
                'f1': f1,
                f'fail_{metric_label}': fail_percentage,
                # We also return some statistics we'll need
                'gwet_ac1': gwet
            })
    
    # Prepare data for plotting
    models = [r['model'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    f1s = [r['f1'] for r in results]
    fail_percentages = [r[f'fail_{metric_label}'] for r in results]

    if not no_plot:    
        # Set up the plot
        fig, ax = plt.subplots()
        # Set the width of bars and positions
        bar_width = 0.25 if do_failures else 0.35
        x_pos = np.arange(len(models))
        
        bars1 = ax.bar(x_pos - bar_width, accuracies, bar_width, label='Accuracy', 
                       color=COLOURS[:len(models)], alpha=0.8, edgecolor='black', linewidth=1.5)
        
        bars2 = ax.bar(x_pos, f1s, bar_width, label='F1 Score',
                       color=COLOURS[:len(models)], alpha=0.8, edgecolor='black', 
                       linewidth=1.5, hatch='///')
        
        if do_failures:
            bars3 = ax.bar(x_pos + bar_width, fail_percentages, bar_width, label='Failure %',
                           color=COLOURS[:len(models)], alpha=0.8, edgecolor='black',
                           linewidth=1.5, hatch='xxx')
            
        accuracy_patch = plt.Rectangle((0, 0), 1, 1, fc='#8c8c8c', edgecolor='black', linewidth=1.5)
        f1_patch = plt.Rectangle((0, 0), 1, 1, fc='#8c8c8c', edgecolor='black',  linewidth=1.5, hatch='///', alpha=0.7)
        failure_patch = plt.Rectangle((0, 0), 1, 1, fc='#8c8c8c', edgecolor='black',  linewidth=1.5, hatch='xxx', alpha=0.7)

        patches = [accuracy_patch, f1_patch]
        patch_labels = ["Accuracy", "F1"]
        if do_failures:
            patches += [failure_patch]
            patch_labels += ["Failures"]
        ax.legend(patches, patch_labels, loc='lower right')
        
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, round(height, 1), ha='center', va='bottom', fontsize=9)
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        add_value_labels(bars3)

        ax.set_xlabel('LLMs', fontsize=12, fontweight='bold')
        ax.set_ylabel('Performance', fontsize=12, fontweight='bold')
        ax.set_title(f'Model Performance: {locale} ({title_suff})',  fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models) #, rotation=45, ha='right')

        ax.grid(True) 

        ax.set_ylim(0, 110)
        plt.tight_layout()

        if save_filename is not None:
            plt.savefig(f"{save_filename}.png")
    else:
        fig, ax = None, None
    
    if not return_full_results:
        results = {"accuracy": {model2readable[m]: percentage for (m, percentage) in zip(MODELS, accuracies)},
                    "f1": {model2readable[m]: percentage for (m, percentage) in zip(MODELS, f1s)}}

    return fig, ax, results


def per_shot_plot(consolidated_dataset: list, metric_label: str, locale: str, MODELS: list, COLOURS: list,
                  models2readable: dict, title_suff: str, save_filename: str=None, do_failures:bool=True,):
    """
    Plot accuracy/F1 over shots. This one is so ugly.

    Params:
    - consolidated_dataset: the parsed, consolidated files from `consolidate_files(locale)`
    - metric_label: one of [no_breakdown, breakdown_no_reasons, breakdown_reasons]
    - locale: one of []
    - MODELS: the models plotted
    - COLOURS: vanity
    - models2readable: a map to niceify the model names
    - title_suff: what are you plotting lol
    - save_filename: if not None, will be the prefix for the filename
    """
    df = pd.DataFrame(consolidated_dataset)
    line_styles = {
        'accuracy': '-',
        'f1': '--',
        'failure': ':'
    }
    
    fig, ax = plt.subplots()
    aggregates = {}
    
    for model_idx, model in enumerate(MODELS):
        model_data = df[df['model'] == model]
        
        colour = COLOURS[model_idx % len(COLOURS)]
        metrics_by_shots = []
        
        all_shots = [s for s in sorted(model_data['shots'].unique())]
        for shots in all_shots:
            shot_data = model_data[model_data['shots'] == shots]
            if not shot_data.empty:

                accuracy = accuracy_score(shot_data['ground_truth'], shot_data[metric_label])*100
                f1 = f1_score(shot_data['ground_truth'], shot_data[metric_label])*100
                failures = shot_data[f'fail_{metric_label}'].mean()*100
                
                metrics_by_shots.append({
                    'shots': shots,
                    'accuracy': accuracy,
                    'f1': f1,
                    'failures': failures
                })
        aggregates[model] = metrics_by_shots
        
        if metrics_by_shots:
            metrics_df = pd.DataFrame(metrics_by_shots)
            shots_values = metrics_df['shots'].values
            
            # Plot accuracy
            line1, = ax.plot(shots_values, metrics_df['accuracy'].values, 
                               color=colour, linestyle=line_styles['accuracy'], 
                               marker='o', markersize=6, label=f'{model}', linewidth=2)
            
            # Plot F1 score
            line2, = ax.plot(shots_values, metrics_df['f1'].values, 
                               color=colour, linestyle=line_styles['f1'], 
                               marker='s', markersize=6, label=f'{model}', linewidth=2)
            
            if do_failures:
                line3, = ax.plot(shots_values, metrics_df['failures'].values, 
                                   color=colour, linestyle=line_styles['failure'], 
                                   marker='^', markersize=6, label=f'{model}', linewidth=2)

    # Customize the plot
    ax.set_xlabel('Shots', fontsize=12)
    ax.set_ylabel('Percentage', fontsize=12)
    ax.set_title(f'Model Performance Over Shots: {locale} ({title_suff})', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)    
    
    ax.set_xticks(all_shots)
    ax.grid(True)

    legend_elements = [Line2D([0], [0], color='#8c8c8c', marker='o', linestyle=line_styles['accuracy'], lw=1.5, label='Line'),
                       Line2D([0], [0], color='#8c8c8c', marker='s', linestyle=line_styles['f1'], lw=1.5, label='Line'),]
    legend_labels = ["Accuracy", "F1"]
    if do_failures: 
        legend_elements += [Line2D([0], [0], color='#8c8c8c', marker='^', linestyle=line_styles['failure'], lw=1.5, label='Line')]
        legend_labels += ["Failures"]

    metric_legend = ax.legend(legend_elements, legend_labels, loc="lower right",
                              bbox_to_anchor=(1, 0.15)) #, handles=[line1])

    models_markers = [Line2D([0], [0], color=c, lw=2, label=models2readable[m]) for m, c in zip(MODELS, COLOURS)]
    model_legend = plt.legend(models_markers, [models2readable[m] for m in MODELS]) #, handles=[line2])

    ax.add_artist(metric_legend)
    plt.tight_layout()

    if save_filename is not None:
        plt.savefig(f"{save_filename}.png")

    return fig, ax, aggregates


def plot_sbs_per_bag(consolidated_dataset: list, metric_label: str, locale: str, MODELS: list, COLOURS: list, 
                     models2readable: dict, title_suff: str, SOURCES: list, plot_size: tuple=None, save_filename: str=None, 
                     suppress_legend:bool=False, do_failures:bool=True, do_accuracy:bool=True, override_bar_width:tuple=None,
                     compute_gwet:bool=False):
    """
    Accuracy/Percentage barplots sbs over a bag (SOURCES).

    Params:
    - consolidated_dataset: the parsed, consolidated files from `consolidate_files(locale)`
    - metric_label: one of [no_breakdown, breakdown_no_reasons, breakdown_reasons]
    - locale: the locale
    - MODELS: the models plotted
    - COLOURS: vanity
    - SOURCES: the aggregate 
    - models2readable: a map to niceify the model names
    - title_suff: what are you plotting lol
    - save_filename: if not None, will be the prefix for the filename
    """

    df = pd.DataFrame(consolidated_dataset)
    df = df[df['model'].isin(MODELS) & df['source'].isin(SOURCES)] # Ideally you should be plotting all but jic

    results = []
    for model in MODELS:
        for source in SOURCES:
            subset = df[(df['model'] == model) & (df['source'] == source)]
            if len(subset) > 0:
                accuracy = accuracy_score(subset['ground_truth'], subset[metric_label])*100
                f1 = f1_score(subset['ground_truth'], subset[metric_label])*100
                gwet = 0 if not compute_gwet else compute_gwet_ac1(subset[metric_label], subset['ground_truth'])*100
                fail_percentage = subset[f'fail_{metric_label}'].mean() * 100
                results.append({
                    'model': models2readable[model],
                    'source': source,
                    'accuracy': round(accuracy),
                    'f1': round(f1),
                    'ac1': round(gwet),
                    'failures': fail_percentage
                })
    _models = [models2readable[m] for m in MODELS]
    
    results_df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(16,4)) if plot_size is None else plt.subplots(figsize=plot_size)

    metrics = ['accuracy', 'f1', 'failures']
    if not do_accuracy:
        metrics.pop(metrics.index("accuracy"))
    if not do_failures:
        metrics.pop(metrics.index("failures"))
    if compute_gwet_ac1:
        metrics.append("ac1")
    n_models, n_metrics = len(_models), len(metrics)
    bar_width = 0.25 if do_failures else 0.35
    group_spacing = 0.5
    if override_bar_width is not None:
        bar_width = override_bar_width[0]
        group_spacing = override_bar_width[-1]

    bar_pos = n_models*n_metrics*bar_width + group_spacing

    positions = []
    for i, source in enumerate(SOURCES):
        base_pos = i*bar_pos
        for j, model in enumerate(_models):
            for k, _ in enumerate(metrics):
                pos = base_pos + j *n_metrics*bar_width + k*bar_width
                positions.append(pos)

    # Plot bars
    for i, source in enumerate(SOURCES):
        for j, model in enumerate(_models):
            model_data = results_df[(results_df['model'] == model) & 
                                   (results_df['source'] == source)]
            if not model_data.empty:
                colour = COLOURS[j % len(COLOURS)]
                # Positions for this model-source combination
                base_pos = i*bar_pos + j*n_metrics*bar_width

                # Accuracy bar
                if do_accuracy:
                    acc_bar = ax.bar(base_pos, model_data['accuracy'].values[0], bar_width,
                                     color=colour, alpha=0.8, edgecolor='black', linewidth=1,
                                     label=f'{model}' if i == 0 else '')
                if compute_gwet:
                    gwet_bar = ax.bar(base_pos, model_data['ac1'].values[0], bar_width,
                                     color=colour, alpha=0.8, edgecolor='black', linewidth=1,
                                     label=f'{model}' if i == 0 else '')
                # F1 bar (with hatching to distinguish)
                f1_bar = ax.bar(base_pos + bar_width if any([do_accuracy, compute_gwet]) else 0, model_data['f1'].values[0], bar_width, 
                                color=colour, alpha=0.8, edgecolor='black', linewidth=1, hatch='//')
                if do_failures:
                    # Failure percentage bar (with different hatching)
                    fail_bar = ax.bar(base_pos + 2*bar_width, 
                                     model_data['failures'].values[0], bar_width, color=colour, 
                                     alpha=0.8, edgecolor='black', linewidth=1, hatch='xx')

    # Customize the plot
    ax.set_xlabel('Source', fontsize=12)
    ax.set_title(f'Model Performance: {locale} ({title_suff})', fontsize=14, fontweight='bold')
    
    source_positions = []
    for i, source in enumerate(SOURCES):
        center_pos = i*bar_pos + (n_models*n_metrics*bar_width)/2 - bar_width/2
        source_positions.append(center_pos)

    ax.set_xticks(source_positions)
    ax.set_xticklabels(SOURCES)
    ax.set_ylim(0, 110)    
    
    # Add grid for better readability
    ax.grid(True) #, axis='y', alpha=0.3, linestyle='--')

    legend_elements = []
    for i, model in enumerate(_models):
        legend_elements.append(Patch(facecolor=COLOURS[i % len(COLOURS)],  alpha=0.8,  
                                     edgecolor='black', label=model))
    
    # Metric patterns
    if do_accuracy:
        legend_elements.append(Patch(facecolor='#8c8c8c', alpha=0.7, label='Accuracy'))
    if compute_gwet:
        legend_elements.append(Patch(facecolor='#8c8c8c', alpha=0.7, label='AC1'))
    legend_elements.append(Patch(facecolor='#8c8c8c', alpha=0.7, hatch='//', label='F1'))
    if do_failures:
        legend_elements.append(Patch(facecolor='#8c8c8c', alpha=0.7, hatch='xx', label='Failures'))

    if not suppress_legend:
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
        ax.set_ylabel('Performance', fontsize=12)
    
    # Add value labels on top of bars
    for i, source in enumerate(SOURCES):
        for j, model in enumerate(_models):
            model_data = results_df[(results_df['model'] == model) & 
                                   (results_df['source'] == source)]
            
            if not model_data.empty:
                base_pos = i * (n_models * n_metrics * bar_width + group_spacing) + \
                          j * n_metrics * bar_width
                
                # Add text labels
                if do_accuracy:
                    ax.text(base_pos, model_data['accuracy'].values[0], # + 0.01, 
                            round(model_data["accuracy"].values[0], 2), ha='center', va='bottom', fontsize=9)
                if compute_gwet:
                    ax.text(base_pos, model_data['ac1'].values[0], # + 0.01, 
                            round(model_data["accuracy"].values[0], 2), ha='center', va='bottom', fontsize=9)
                ax.text(base_pos + bar_width if any([do_accuracy, compute_gwet]) else 0, model_data['f1'].values[0], #+ 0.01, 
                       round(model_data["f1"].values[0], 2),  ha='center', va='bottom', fontsize=9)
                if do_failures:
                    ax.text(base_pos + 2 * bar_width, model_data['failures'].values[0]/100, # + 0.01, 
                           round(model_data["failures"].values[0]), ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()

    if save_filename is not None:
        plt.savefig(f"{save_filename}.png")

    return fig, ax, results_df


def class_prediction_plot(consolidated_dataset: list, metric_label: str, locale: str, MODELS: list, 
                          COLOURS: list, models2readable: dict, title_suff: str, plot_size=None, save_filename: str=None):
    """
    Do a class prediction plot (how many times did what get predicted)

    Params:
    - consolidated_dataset: the parsed, consolidated files from `consolidate_files(locale)`
    - metric_label: one of [no_breakdown, breakdown_no_reasons, breakdown_reasons]
    - locale: the locale
    - MODELS: the models plotted
    - COLOURS: vanity
    - models2readable: a map to niceify the model names
    - title_suff: what are you plotting lol
    - plot_size: to alter plotsizes if needed
    - save_filename: if not None, will be the prefix for the filename
    """
    df = pd.DataFrame(consolidated_dataset)
    
    results = {'ground_truth_0': {}, 'ground_truth_1': {}}

    if "Human" not in models2readable: models2readable["Human"] = "Human"
    
    for gt_value in [0, 1]:
        gt_key = f'ground_truth_{gt_value}'

        gt_data = df[df['ground_truth'] == gt_value]
        results[gt_key]['Human'] = len(gt_data)*100/len(df)
        #100.0 if gt_value == 1 else 0.0

        for model in MODELS:
            model_data = df[(df['model'] == model) & (df[metric_label] == gt_value)]
            model_preds = df[df["model"] == model]
            results[gt_key][model] = len(model_data)*100/len(model_preds)
    

    fig, ax = plt.subplots(figsize=(12, 6)) if plot_size is None else plt.subplots(figsize=plot_size) 
    categories = ['Human'] + MODELS
    x = np.arange(2)
    width = 0.8 / len(categories)
    
    for i, category in enumerate(categories):
        values = []
        for gt_key in ['ground_truth_0', 'ground_truth_1']:
            values.append(results[gt_key].get(category, 0))
        pos = x + (i - len(categories)/2 + 0.5) * width
        
        bars = ax.bar(pos, values, width, label=models2readable[category], color=COLOURS[i % len(COLOURS)], 
                      edgecolor='black', linewidth=0.5)

        for bar, value in zip(bars, values):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, round(value, 1), 
                        ha='center', va='bottom', fontsize=8)
    
    # Customize plot
    ax.set_xlabel('Class Value', fontsize=12)
    ax.set_ylabel('Prediction Frequency (%)', fontsize=12)
    ax.set_title(f'Class Predictions: {locale} ({title_suff})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Class = 0', 'Class = 1'])
    ax.set_ylim(0, 110)  # Set y-axis limit to accommodate percentage labels
    
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Model')
    ax.grid(True) #, axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if save_filename is not None:
        plt.savefig(f"{save_filename}.png")

    return fig, ax, results



def plot_agreements(data_list, CRITERIA, locale, MODELS, COLOURS, models2readable, title_suff, save_filename: str=None):
    """
    Create a spider plot from a list of dictionaries.
    
    Params:
    - data_list: List of dictionaries with 'criterion', 'model', 'agreement' keys
    - CRITERIA: List/set of criterion names
    - locale: the locale to plot
    - MODELS: List/set of model names
    - COLOURS: List of colors for each model
    - models2readable: a map to niceify the model names
    - title_suff: what are you plotting lol
    - save_filename: if not None, will be the prefix for the filename
    """
    num_vars = len(CRITERIA)
    pi = 3.14159265359
    
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    data_dict = {}
    for model in MODELS:
        data_dict[model] = {criterion: 0.0 for criterion in CRITERIA}
    
    for item in data_list:
        if item['model'] in MODELS and item['criterion'] in CRITERIA:
            data_dict[item['model']][item['criterion']] = item['agreement']
    
    for idx, model in enumerate(MODELS):
        values = [data_dict[model][criterion] for criterion in CRITERIA]
        values += values[:1]  # Complete the circle

        colour = COLOURS[idx % len(COLOURS)]
        ax.plot(angles, values, 'o-', linewidth=2, label=models2readable[model], color=colour)
        ax.fill(angles, values, alpha=0.25, color=colour)
    
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.capitalize() for c in CRITERIA], size=18, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=16)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.08), fontsize=16)

    plt.title(f'Agreement: {locale} ({title_suff})', size=24, y=1.08, fontweight='bold')
    plt.tight_layout()

    if save_filename is not None:
        plt.savefig(f"{save_filename}.png")

    return fig, ax, {}


def plot_human_data(consolidated_dataset, locale: str, MODELS: list, COLOURS: list,
                    models2readable: dict, title_suff: str, plot_size=None, supress_legend: bool=False, save_filename: str=None):

    """
    Plot human preference data

    Params:
    - consolidated_dataset: the dataset
    - locale: the locale to plot
    - MODELS: List/set of model names
    - COLOURS: List of colors for each model
    - models2readable: a map to niceify the model names
    - title_suff: what are you plotting lol
    - plot_size: to alter plotsizes if needed
    - save_filename: if not None, will be the prefix for the filename
    """

    df = pd.DataFrame(consolidated_dataset)
    df = df[df['model'].isin(MODELS)]

    # Calculate percentages
    results = []
    for model in MODELS:
        model_data = df[df['model'] == model]
        sources = df['source'].unique()
        for source in sources:
            source_data = model_data[model_data['source'] == source]
            if len(source_data) > 0:
                label_1_percentage = (source_data['label'].sum() / len(source_data)) * 100
            else:
                label_1_percentage = 0
            results.append({
                'model': model,
                'source': source,
                'percentage': label_1_percentage
            })
    
    results_df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(16, 8)) if plot_size is None else plt.subplots(figsize=plot_size)
    
    sources = sorted(df['source'].unique())
    n_sources, n_models = len(sources), len(MODELS)
    group_width = 1 #0.8
    bar_width = group_width / n_models
    x_groups = np.arange(n_sources) * 1.2

    patterns = ['', '///', '\\\\\\', '|||', '---', '+++', 'xxx', 'ooo']
    
    # Plot bars for each model
    for i, model in enumerate(MODELS):
        model_values = []
        for source in sources:
            val = results_df[(results_df['model'] == model) &  (results_df['source'] == source)]['percentage']
            model_values.append(val.values[0] if len(val) > 0 else 0)
        
        x_positions = x_groups + i * bar_width
        
        bars = ax.bar(x_positions, model_values, bar_width,
                      label=models2readable[model], color=COLOURS[i % len(COLOURS)], edgecolor='black',
                      linewidth=1.5, hatch=patterns[i % len(patterns)], alpha=0.8)

        for x_pos, val in zip(x_positions, model_values):
            if val > 0:
                ax.text(x_pos, val + 0.5, round(val, 1), ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Source', fontsize=12, fontweight='bold')
    ax.set_title(f'Human-Determined High-Quality Outputs: {locale}{title_suff}', fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis labels at the center of each group
    ax.set_xticks(x_groups + group_width/2 - bar_width/2)
    ax.set_xticklabels(sources)
    ax.set_ylim(0, 110)  # Set y-axis limit to accommodate percentage labels

    # Add legend and grid
    if not supress_legend: 
        ax.legend(title='LLMs', loc='lower right')
        ax.set_ylabel('Label=1 (%)', fontsize=12, fontweight='bold')

    ax.grid(True)
    
    plt.tight_layout()
    if save_filename is not None:
        plt.savefig(f"{save_filename}.png")
    
    return fig, ax, results_df



def plot_human_data_no_source(consolidated_dataset, locale: str, MODELS: list, COLOURS: list, 
                              models2readable: dict, title_suff: str, plot_size=None, save_filename: str=None):
    
    """
    Plot human preference data without a per-source breakdown

    Params:
    - consolidated_dataset: the dataset
    - locale: the locale to plot
    - MODELS: List/set of model names
    - COLOURS: List of colors for each model
    - models2readable: a map to niceify the model names
    - title_suff: what are you plotting lol
    - plot_size: to alter plotsizes if needed
    - save_filename: if not None, will be the prefix for the filename
    """

    model_counts = defaultdict(lambda: {'total': 0, 'label_true': 0})
    
    # Count total and label=True occurrences for each model
    for item in consolidated_dataset:
        model = item['model']
        if model in MODELS:  # Only process models in MODELS list
            model_counts[model]['total'] += 1
            if item['label']:  # Count True labels
                model_counts[model]['label_true'] += 1
    
    # Calculate percentages
    percentages = []
    models_with_data = []
    
    for model in MODELS:
        if model in model_counts and model_counts[model]['total'] > 0:
            percentage = (model_counts[model]['label_true'] / model_counts[model]['total']) * 100
            percentages.append(percentage)
            models_with_data.append(model)
        else:
            percentages.append(0)
            models_with_data.append(model)
    
    # Create the bar plot
    fig, ax = plt.subplots(figsize=(16, 8)) if plot_size is None else plt.subplots(figsize=plot_size)
    x_pos = np.arange(len(MODELS))
    
    patterns = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
    bars = ax.bar(x_pos, percentages, color=COLOURS[:len(MODELS)], 
                  edgecolor='black', linewidth=1.5)
    
    for bar, pattern in zip(bars, patterns[:len(bars)]): bar.set_hatch(pattern)
    
    # Customize the plot
    ax.set_xlabel('LLM', fontsize=12, fontweight='bold')
    ax.set_ylabel('Label=1 (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Human-Determined High-Quality Outputs: {locale}{title_suff}', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([models2readable[m] for m in MODELS], ha='right') # rotation=45, ha='right')
    
    for i, (bar, percentage) in enumerate(zip(bars, percentages)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)
    
    ax.grid(True) #axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(percentages) * 1.15 if percentages else 100)
    legend_elements = []
    for i, model in enumerate(MODELS[:len(bars)]):
        legend_elements.append(plt.Rectangle((0,0),1,1, 
                                            facecolor=COLOURS[i], 
                                            edgecolor='black',
                                            hatch=patterns[i % len(patterns)],
                                            label=models2readable[model]))
    
    ax.legend(handles=legend_elements, 
              loc='lower left', # if locale in ["West Frisian", "Cornish"] else 'lower right', 
              framealpha=0.9,
              fontsize=9)
    
    plt.tight_layout()
    if save_filename is not None:
        plt.savefig(f"{save_filename}.png")
    
    return fig, ax, {models2readable[m]: percentage for (m, percentage) in zip(MODELS, percentages)}
