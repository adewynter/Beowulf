# Data Analysis

Contains all the code necessary to baseline (ICL) the human annotations and generate synthetic data (`Generator.ipynb`), and perform the data analysis (`Analysis.ipynb`). The notebook `raw_numbers.ipynb` is meant to be for quick, birds-eye view evaluation (`Analisys.ipynb` is gigantic).
It also contains the predictions for our various experiments.

The folder is structured as follows:

```
icl_predictions/
    <dialect or language>/  # The ICL baselines (predictions) per model
data/
    human_annotated/
        <*.json>            # Human-annotated and corrected (prompts only) data.
        distance_analysis/  # The US-locale aligned data for distance analysis only
    synthetic_data/         # Its namesake
finetuned_predictions/
    <dialect or language>/  # The finetuned predictions (DPO, SFT, synthetic SFT)
*.ipynb                     # Notebooks with the analysis/generation
*.py                        # You'll need these if you want to replicate the work lol
```

Please note that due to key considerations we cannot share the client we used for the private models (OpenAI, Anthropic). You need to modify `llmclient.py` to make it work.

# Licencing

Refer to the paper for source data licencing considerations.
