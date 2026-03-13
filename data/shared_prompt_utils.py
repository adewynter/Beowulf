
ALL_CRIT_DEFAULT_RESPONSE = {"c1": 0, "c2a": 0, "c2b": 0, "c3": 0, "c4": 0, "c5": 0, 
                            "c1_reason": "FAIL", "c2a_reason": "FAIL", "c2b_reason": "FAIL", "c3_reason": "FAIL", "c4_reason": "FAIL", "c5_reason": "FAIL"}


rubric_main = """You are an LLM evaluator. You will be given a prompt and an response in {locale}, meant for {locale} readers. 
Your job will be to verify if the response follows certain criteria and give a final binary score.

Check the output against the criteria below. If it fulfils the criteria, it should be a 1. Otherwise, 0. 
{aggregator}
"""

rubric_paraphrase_main = """You will be given a prompt and an response in {locale}, meant for {locale} readers. 
Check the output against the criteria below. If it fulfils the criteria, it should be a 1. Otherwise, 0. 
{aggregator}
"""

output_format_specs_label_only = """
Give your answer in JSON format, using the labels 0 or 1. Use this scheme:
{"Label": <the label, 0 or 1>}
Only use the key "Label" and the values 0 or 1.
"""


output_format_specs_all_criteria = """
Give your answer in JSON format, using the values 0 or 1 for each criterion. Use this scheme:
{"c1": <the value, 0 or 1>,
"c2a": <the value, 0 or 1>,
"c2b": <the value, 0 or 1>,
"c3": <the value, 0 or 1>,
"c4": <the value, 0 or 1>,
"c5": <the value, 0 or 1>,
"Label": <the value, 0 or 1>}
Only use the keys "c1", "c2a", "c2b", "c3", "c4", "c5", and "Label", and the values 0 or 1.
"""


output_format_specs_all_criteria_with_reasons = """
Give your answer in JSON format, using the values 0 or 1 for each criterion. Use this scheme:
{"c1": <the value, 0 or 1>,
{"c1_reason": <the value, 0 or 1>,
"c2a": <the value, 0 or 1>,
"c2a_reason": <the value, 0 or 1>,
"c2b": <the value, 0 or 1>,
"c2b_reason": <the value, 0 or 1>,
"c3": <the value, 0 or 1>,
"c3_reason": <the value, 0 or 1>,
"c4": <the value, 0 or 1>,
"c4_reason": <the value, 0 or 1>,
"c5": <the value, 0 or 1>,
"c5_reason": <the value, 0 or 1>,
"Label": <the value, 0 or 1>}
Only use the keys "c1", "c2a", "c2b", "c3", "c4", "c5"; "c1_reason", "c2a_reason", "c2b_reason", "c3_reason", "c4_reason", "c5_reason"; and "Label".
If the value for a key is 0, its corresponding reason cannot be empty.
"""

output_format_specs_one_criteria = lambda cr: """
Give your answer in JSON format, using the labels 0 or 1. Use this scheme:
{{"{cr}": <the label, 0 or 1>}}
Only use the key "{cr}" and the values 0 or 1.
""".format(cr=cr)

output_format_specs_one_criteria_with_reasons = lambda cr: """
Give your answer in JSON format, using the labels 0 or 1. Use this scheme:
{{"{cr}": <the label, 0 or 1>,
"{cr}_reason": the reason for the label}}
Only use the key "{cr}" and the values 0 or 1. 
If the value is 0, the reason cannot be empty.
""".format(cr=cr)


# IF prompt
check1 = "The response must be in {locale}."
check2a = """The response must be culturally (e.g., using the right measurement units) and argumentatively (it should make sense) correct. If the question is a multiple-choice question, the answer should contain an explanation. If it requests code, it should also contain an explanation that is clear. Grammar or accuracy of the response are not measured here."""
check2b = "The response must be correct. If it is code, it should not have syntax errors."
check3 = "The response must be grammatically correct: coherent, good spelling, etc. with respect to {locale}. Code syntax is not measured here."
check4 = "The response must not be cut off."
check5 = """The model must follow the instructions from the user (the prompt) exactly and completely, even if its answer is wrong. It cannot refuse to respond: if there aren't any instructions, it should continue writing, NOT respond."""

rubric_good_crit_map = {
    "c1": check1, "c2a": check2a, "c2b": check2b,
    "c3": check3, "c4": check4, "c5": check5
}

base_rubric = """{rubric_main}
# Criteria:
c1: {check1}

c2a: {check2a}

c2b: {check2b}

c3: {check3}

c4: {check4}

c5: {check5} 

# Output format:
"""

aggregator_good = "If any of the criteria score a zero, the response must be zero."
