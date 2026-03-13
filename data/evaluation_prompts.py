from shared_prompt_utils import (base_rubric,
                                 rubric_main,
                                 rubric_paraphrase_main,
                                 aggregator_good,
                                 check1, check2a, check2b, check3, check4, check5,
                                 rubric_good_crit_map,
                                 output_format_specs_label_only,
                                 output_format_specs_all_criteria,
                                 output_format_specs_one_criteria,
                                 output_format_specs_all_criteria_with_reasons,
                                 output_format_specs_one_criteria_with_reasons,
                                )


def get_evaluator_prompt_all_criteria(entry: dict, num_exemplars: int, exemplar_dataset: list, locale: str,
                                      request_breakdown=False, request_reasons=False):
    '''
    Evaluator prompt returning the total label, optionally returning all the criteria _before_ the label, and,
    also optionally, the reasons for every criterion. 
    in one call.
    '''
    get_formatted = lambda p, o: f"<prompt>\n{p}\n</prompt>\n<response>\n{o}\n</response>"

    this_rubric_main = rubric_main.format(aggregator=aggregator_good, locale=locale)
    this_check1 = check1.format(locale=locale)
    this_check3 = check3.format(locale=locale)
    this_system_prompt = base_rubric.format(rubric_main=this_rubric_main,
                                           check1=this_check1, check2a=check2a, check2b=check2b,
                                           check3=this_check3, check4=check4, check5=check5)

    if request_breakdown:
        system_prompt = this_system_prompt + output_format_specs_all_criteria
        if request_reasons:
            system_prompt = this_system_prompt + output_format_specs_all_criteria_with_reasons
    else:
        system_prompt = this_system_prompt + output_format_specs_label_only

    exemplars = []
    do_one_more = False
    for i in range(num_exemplars + 1):
        if not do_one_more and i == num_exemplars: break
        pro, out = exemplar_dataset[i]["Prompt"], exemplar_dataset[i]["Output"]
        if pro == entry["Prompt"] and out == entry["Output"]:
            do_one_more = True 
            continue
        lab = exemplar_dataset[i]["Label"]
        exemplars.append({"role": "user", "content": get_formatted(pro, out)})
        if request_breakdown:
            ass_string = "{" # I do love chatML's nomenclature
            for k, v in exemplar_dataset[i]["Rubric"].items():
                content = exemplar_dataset[i]["Rubric"][k]
                if "reason" in k:
                    if request_reasons:
                        ass_string += f'"{k}": "{content}", '
                else:
                    ass_string += f'"{k}": {content}, '
            ass_string += ' "Label": ' + str(lab) + '}'
            exemplars.append({"role": "assistant", "content": ass_string})
        else:
            exemplars.append({"role": "assistant", "content": '{"Label": ' + str(lab) + '}'})

    prompt = [{"role": "system", "content": system_prompt}]
    prompt += exemplars
    prompt += [{"role": "user", "content": get_formatted(entry["Prompt"], entry["Output"])}]
    return prompt


def get_evaluator_prompt_single_criteria(entry: dict, num_exemplars: int, exemplar_dataset: list, locale: str,
                                         criterion: str, request_reasons=False):
    '''
    Evaluator prompt returning the score for a single criterion. Optionally request the '_reason' field.
    '''
    get_formatted = lambda p, o: f"<prompt>\n{p}\n</prompt>\n<response>\n{o}\n</response>"
    this_system_prompt = rubric_main.format(aggregator=aggregator_good, locale=locale)

    criterion_str = rubric_good_crit_map[criterion]
    if criterion in ["c1", "c3"]:
        criterion_str = criterion_str.format(locale=locale)

    this_system_prompt += f"\n# Criterion:\n{criterion_str}\n\n# Output format:\n"

    system_prompt = this_system_prompt + output_format_specs_one_criteria(cr=criterion)
    if request_reasons:
        system_prompt = this_system_prompt + output_format_specs_one_criteria_with_reasons(cr=criterion)

    exemplars = []
    do_one_more = False
    for i in range(num_exemplars + 1):
        if not do_one_more and i == num_exemplars: break
        pro, out = exemplar_dataset[i]["Prompt"], exemplar_dataset[i]["Output"]
        if pro == entry["Prompt"] and out == entry["Output"]:
            do_one_more = True 
            continue
        label = exemplar_dataset[i]["Rubric"][criterion]
        label_reason = exemplar_dataset[i]["Rubric"][f"{criterion}_reason"]
        exemplars.append({"role": "user", "content": get_formatted(pro, out)})
        ass_string = '{"'
        ass_string += f'{criterion}": {label}'
        if request_reasons:
            ass_string += f', "{criterion}_reason": "{label_reason}"'
        ass_string += '}'
        exemplars.append({"role": "assistant", "content": ass_string})

    prompt = [{"role": "system", "content": system_prompt}]
    prompt += exemplars
    prompt += [{"role": "user", "content": get_formatted(entry["Prompt"], entry["Output"])}]
    return prompt


def get_generator_prompt(x: dict, y: int, locale: str, criteria: dict):
    '''
    Generator prompt, generating a new x-tilde based on the rubric. Here `x` is the original datapoint,
    while `y` is the estimated `y_tilde` (I just misnamed it). `criteria` must be estimated as well.
    '''
    this_rubric_main = rubric_paraphrase_main.format(aggregator=aggregator_good, locale=locale)
    this_check1 = check1.format(locale=locale)
    this_check3 = check3.format(locale=locale)
    rubric = base_rubric.format(rubric_main=this_rubric_main,
                                check1=this_check1, check2a=check2a, check2b=check2b,
                                check3=this_check3, check4=check4, check5=check5)


    system_prompt = """ You are a paraphraser evaluating a prompt and an output for an LLM. 
    You will be given a datapoint (prompt/output), a label, and a list of reasons why that datapoint's output has that label. 
    Your job will be to return a SIMILAR prompt and output, such that the OUTPUT (1) it matches the list of reasons, and (2) matches the label.
    The output must match the values in the list of reasons. 

    Here's the rubric used for these reasons:
    {rubric}

    Your response must be in JSON using the following schema:
    {{
        "Prompt": the new, paraphrased user prompt. 
        "Output": the new, paraphrased output fulfiling the criteria.
    }}
    Only use the keys "Prompt" and "Output"
    """

    user_prompt = """<prompt>
    {user_prompt}
    </prompt>
    <output>
    {user_output}
    </output>
    <reasons>
    {user_crits}
    </reasons>"""

    assistant_response = """{{
    "Prompt": "{user_prompt}",
    "Output": "{user_output}"
    }}"""

    line_separated_crits = ""
    line_separated_crits = "\n".join([f"{k}: {v}" for k, v in criteria.items()])
    line_separated_crits += f"\nLabel: {y}"

    this_system_prompt = system_prompt.format(rubric=rubric)
    this_user_prompt = user_prompt.format(user_prompt=x["Prompt"],
                                          user_output=x["Output"],
                                          user_crits=line_separated_crits)

    prompt = [{"role": "system", "content": this_system_prompt}]
    prompt += [{"role": "user", "content": this_user_prompt}]
    return prompt

