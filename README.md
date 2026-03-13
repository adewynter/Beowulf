# Beowulf

Repository for the paper 'The Hrunting of AI: Where and How to Improve English Dialectal Fairness', by Wei Li and Adrian de Wynter. 

This paper explores the relationship between human-human agreement and LLM-human agreement, and its impact to learnability, in four English dialects and West Frisian. 

We find that human-human agreement harms fine-tuneability (even with human data!); with approaches like DPO and synthetic data + SFT even having a detrimental effect in some locales.

This is not uniform, however: high agreement locales benefit from these algorithms. Non-English, low-resource languages (West Frisian) do benefit as well, though at a minor extent. So this is just for dialects. 

We show that this is stat sig by indicating that there's a correlation between resourceness and downstream performance, and argue that low-resource dialects will be harmed most by lack of annotators. We also note that, lexically, low-resource languages have clear, non-ambiguous grammars--unlike low-resource dialects, where human-human agreement will be low. 

Thus our call to action is to develop linguistics-informed tools to better improve these models. Our second call to action is to involve native speakers to mitigate this low agreement. 


## Locales Covered
- Yorkshire English
- Geordie English
- Cornish (West Country) English
- African-American Vernacular English
- West Frisian


## Repo structure

```
env.yaml    # To repro the things from the finetuning bits
notebooks/  # All the notebooks used for finetuning and evaluating/creating synthetic data
data/       # Most of the stuff you need is here: human data, synthetic data, analysis stuff, etc
```
`data/` has another README since it's a bit of a convoluted folder.

## Notes

- We do not provide an LLMClient object (obviously, these keys are expensive). The skeleton provided works for Qwen under transformers; it is trivial to mod it for other APIs.
- Because of anonymisation, the notebooks might do weird things and some cells might not run at all. Upon release we will provide the original artifacts.

## Citation
```bibtex
Coming soon!
```

## Licence

Every original code in this repository is MIT licence. The original work (datasets) from which this work was sourced belongs to the authors. See the paper for details.

