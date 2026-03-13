from shared_prompt_utils import rubric_good_crit_map


def get_transliteration_prompt(prompt: str, locale: str):
    sys_prompt = """You are a {locale} translation/transliterator bot. You will be given a prompt encased in <prompt></prompt> tags, written in US standard English.
Your job will be to transcreate the prompt into {locale}. 
Transliteration means that you must translate/transliterate the prompt _and_ alter it so that it is culturally relevant to the speakers of {locale}. 
For example, while a prompt in Americans might know about Joe Biden, in the UK they might think more of Rishi Sunak. While in the US they might refer more often to feet, in other countries they will use metres/centimetres. 
So you need to make the necessary changes.

Return your transliterated prompt as a JSON file as follows:
{{
    "transliteration": <your transliteration>
}}
Only use the key "transliteration".
"""
    usr_prompt = "<prompt>\n" + prompt + "\n</prompt>"
    return [{"role": "system", "content": sys_prompt.format(locale=locale)},
            {"role": "user", "content": usr_prompt}]


def get_generation_prompt(prompt, locale):
    sys_prompt = """You are a helpful assistant who only speaks {locale}. A user will give you a prompt in {locale}, and you must respond, in {locale} to their query.
If the prompt does not contain an instruction, continue writing. Ensure your response is culturally relevant. 
For example, while a prompt in Americans might know about Joe Biden, in the UK they might think more of Rishi Sunak. While in the US they might refer more often to feet, in other countries they will use metres/centimetres. 
Return your response to the prompt as a JSON file as follows:
{{
    "response": <your response>
}}
Only use the key "response".
"""
    usr_prompt = prompt
    return [{"role": "system", "content": sys_prompt.format(locale=locale)},
            {"role": "user", "content": usr_prompt + f"\nRespond only in {locale}"}]



def generate_datapoint_from_datapoint_per_criterion(entry: dict, locale: str, criterion=None, request_reasons=False):
    '''
    Prompt to generate similar datapoints given a criterion. It'll be almost the same, except that (a) paraphrased, 
    and (b) with the criterion flipped.
    '''
    system_prompt = """You are a datapoint generator for {locale} data.
You'll be given a user prompt (inside <prompt></prompt> tags) and an output (r. <output></output>).
Your job will be to return a SIMILAR datapoint (prompt, output) {reasons_explanation}such that:

1. The prompt and the output are similar, but not the same. We'd rather have semantically similar entries, instead of a paraphrase.
2. Unless the criterion says otherwise, your answer MUST be in {locale}.
3. The OUTPUT does NOT fufil the criterion below:

<criterion>
{criterion_str}
</criterion>

4. The output DOES fulfil the criteria below:
<criteria>
{good_criteria_str}
</criteria>

Return your output as JSON as follows:
{{
    "Prompt": <the paraphrased prompt>,
    "Output": <the paraphrased output>{criterion_reasons}
}}
Only use the keys "Prompt", "Output"{criterion_reasons_keys}.
"""

    criterion_str = rubric_good_crit_map[criterion]
    if criterion in ["c1", "c3"]:
        criterion_str = criterion_str.format(locale=locale)

    split_crit = []
    for i, (k, v) in enumerate(rubric_good_crit_map.items()):
        c = v
        if k != criterion:
            if k in ["c1", "c3"]:
                c = c.format(locale=locale)
            split_crit.append(f"{i + 1}. {c}")
    good_criteria_str = "\n".join(split_crit)

    criterion_reasons, criterion_reasons_keys = "", ""
    reasons_explanation = ""
    if request_reasons:
        criterion_reasons = """,
    "Reason": <a SHORT sentence for the reason>
"""     
        criterion_reasons_keys = ', and "Reason"'
        reasons_explanation = "along with a reason why you generated that datapoint, "

    user_prompt = """<prompt>
    {prompt}
    </prompt>
    <output>
    {output}
    </output>
"""

    system_prompt = system_prompt.format(criterion_str=criterion_str,
                                         good_criteria_str=good_criteria_str,
                                         locale=locale,
                                         reasons_explanation=reasons_explanation,
                                         criterion_reasons=criterion_reasons,
                                         criterion_reasons_keys=criterion_reasons_keys)
    user_prompt = user_prompt.format(prompt=entry["Prompt"],
                                     output=entry["Response"])

    prompt = [{"role": "system", "content": system_prompt}]
    prompt += [{"role": "user", "content": user_prompt}]
    return prompt
