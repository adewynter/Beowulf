import json
import transformers
import torch
from transformers import pipeline
from transformers import logging
from transformers import AutoTokenizer

logging.set_verbosity_error()


def get_llm_response(model, assembled_prompt, debug=False):
    if debug:
        print("Assembled Prompt:", assembled_prompt)

    try:
        resp = model.send_request(assembled_prompt)
    except Exception as e:
        if debug:
            print("LLM error:", e)
        return "FAIL"

    # HF text-generation pipeline returns a string in outputs[0]["generated_text"]
    text = resp[0]["generated_text"]

    if debug:
        print("Raw pipeline output:", resp)

    return text


class LLMClient:
    """
    Client to run SLM code available in Huggingface.
    Designed to support Qwen3 with thinking mode disabled.
    """

    def __init__(self, params, model_id, is_dumb=False):
        self._pipeline = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        self._tokenizer = self._pipeline.tokenizer  # ### NEW
        self._is_dumb = is_dumb
        self._params = params
        self._model = model_id

    def _build_prompt(self, assembled_prompt):
        """
        Convert chat messages to a string prompt with thinking disabled.
        """
        if isinstance(assembled_prompt, list):
            return self._tokenizer.apply_chat_template(
                assembled_prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # ### KEY LINE
            )
        return assembled_prompt

    def send_request(self, assembled_prompt):
        prompt = self._build_prompt(assembled_prompt)

        gen_kwargs = dict(
            max_new_tokens=self._params["max_tokens"],
            pad_token_id=self._tokenizer.eos_token_id,
            eos_token_id=self._tokenizer.eos_token_id,  # ADD THIS
        )

        if self._params["temperature"] <= 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = (
                self._params["temperature"]
                if self._is_dumb
                else self._params["temperature"] / 2
            )
            if "top_p" in self._params:
                gen_kwargs["top_p"] = self._params["top_p"]

        # ADD: Prevent repetition
        gen_kwargs["repetition_penalty"] = 1.1
        gen_kwargs["num_beams"] = 1  # Use greedy decoding

        outputs = self._pipeline(prompt, **gen_kwargs, return_full_text=False)
        return outputs

    def update_params(self, params):
        for k, v in params.items():
            self._params[k] = v

# ...existing code...

def extract_first_json_object(text: str):
    import re
    # strip common wrappers
    text = text.strip()
    text = text.replace("```json", "```").strip()

    # Clean ALL invalid JSON escape sequences BEFORE parsing
    # Valid JSON escapes: \" \\ \/ \b \f \n \r \t \uXXXX
    # Remove backslash before any char that is NOT a valid JSON escape target
    text = re.sub(r'\\(?!["\\/bfnrtu])', '', text)

    # If it's already pure JSON
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text, strict=False)
        except json.JSONDecodeError:
            pass  # Fall through to balanced-brace scanner

    # Scan for first balanced {...}
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found")

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    snippet = text[start:i+1]
                    try:
                        return json.loads(snippet, strict=False)
                    except json.JSONDecodeError:
                        # Try cleaning the snippet too
                        snippet = re.sub(r'\\(?!["\\\//bfnrtu])', '', snippet)
                        return json.loads(snippet, strict=False)

    raise ValueError("Unbalanced JSON braces")

# ...existing code...

def retrieve(prompt: list, llm: LLMClient, DEFAULT_RESPONSE: dict={"Label": 0}, assert_fn=None, return_raw=False, max_tries=5, debug=False) -> tuple:
    """
    Code to retrieve an output and validate it. Returns the response (or `DEFAULT_RESPONSE`) and a boolean (failed true/false).
    ---
    prompt (list): a prompt in ChatML form.
    llm (LLMClient): the LLM
    DEFAULT_RESPONSE (dict): the default response in case this thing fails.
    assert_fn (Callable): if not None, must return True if a test passes; False otherwise.
    return_raw (bool, False): return the raw response instead of parsing it.
    max_tries (int, 5): maximum times to attempt the call (default: 5)
    debug (bool, False): print stuff.
    """             
    import re
    response = None
    tmp_response = None
    tries = 0
    while True:
        if debug: print("tries", tries)
        if tries > max_tries: break
        if response is not None: break
        try:
            response = get_llm_response(llm, prompt, debug=debug)
            if debug: print(response)
            tmp_response = response
            # Use the robust JSON extractor instead of brittle string splitting
            try:
                response = extract_first_json_object(response)
            except (json.JSONDecodeError, ValueError) as e:
                if debug: print(f"First parse failed: {e}")
                # Fallback: aggressively strip all invalid JSON escapes
                cleaned = re.sub(r"\\(?![\"\\\//bfnrtu])", "", response)
                try:
                    response = extract_first_json_object(cleaned)
                except (json.JSONDecodeError, ValueError) as e2:
                    if debug: print(f"Second parse failed after cleaning: {e2}")
                    response = None
                    tries += 1
                    continue
            # Hotfix for LLMAPI's dumbassery
            if "error" in response or response in ["FAIL", "FAIL OOT", "FAIL FLAG", "FAIL"]:
                tries += 1
                response = None
            if response is not None:
                if assert_fn is not None:
                    if not assert_fn(response):
                        if debug: print("assertion error")
                        response = None
                        tries += 1
                else:
                    break
        except Exception as e:
            if debug: print(f"Unexpected error: {e}")
            response = None
            tries += 1
    if return_raw:
        return tmp_response, None
    if response is None:
        return DEFAULT_RESPONSE, True
    return response, False
