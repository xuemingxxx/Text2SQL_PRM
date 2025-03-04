from model_config import MODEL_HOST, BASE_MODEL_NAME
from vllm import LLM, SamplingParams
from typing import List
from cmdline import args as cmd_args
import ipdb
import sys
sys.setrecursionlimit(2000)


token_counter = 0

def increment_token_counter(new_tokens):
    assert new_tokens >= 0
    global token_counter
    token_counter += new_tokens
    # print("Current token count = ", token_counter)


if MODEL_HOST == "openai":
    import openai_generate

    def generate(prompt: str, num: int) -> List[str]:
        if args.token_limit is not None:
            print("WARN: Using OpenAI model, which does not support token limit.")
        return openai_generate.generate(prompt, num)

    def generate_full(prompt: str) -> str:
        if args.token_limit is not None:
            print("WARN: Using OpenAI model, which does not support token limit.")
        return generate(prompt, 1)[0]

elif MODEL_HOST == "huggingface":
    import torch
    import huggingface_generate
    from transformers import TextStreamer

    if cmd_args.infer_methd == 'vllm':
        model = huggingface_generate.load_model_vllm()
    else:
        if  'Qwen' in BASE_MODEL_NAME:
            _, model, tokenizer = huggingface_generate.load_qwen_model()
        else:
            _, model, tokenizer = huggingface_generate.load_model()
        model_generation_token_args = huggingface_generate.get_model_generation_token_args(
            tokenizer
        )
        # model_generation_token_args_with_stopstrings = huggingface_generate.get_model_generation_token_args_with_stopstrings(tokenizer)

        # Needed to find end/beginning of sequence
        eos_token = tokenizer.eos_token
        bos_token = tokenizer.bos_token

    # Needed to keep strings looking normal
    def strip_bos(s):
        if bos_token:
            while s.startswith(bos_token):
                s = s[len(bos_token) :]
        return s.lstrip()

    def gen(
        prompt, model_generation_args, num=1, return_hiddens=False, **kwargs
    ) -> List[str]:
        args = {**model_generation_args, **kwargs}
        if 'Qwen' in cmd_args.base_model_name:
            messages = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}]
            state_with_template = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True)
        else:
            state_with_template = prompt

        model_input = tokenizer(state_with_template, return_tensors="pt").to("cuda")
        begin_input_length = model_input["input_ids"].shape[1]

        def helper(tid):
            return tid not in tokenizer.all_special_ids

        input_ntokens = sum(
            sum(helper(tid) for tid in t) for t in model_input["input_ids"]
        )
        model.eval()
        with torch.no_grad():
            # ipdb.set_trace()
            generate_dict = model.generate(
                **model_input,
                num_return_sequences=num,
                output_hidden_states=return_hiddens,
                return_dict_in_generate=True,
                stopping_criteria=huggingface_generate.get_stopping_criteria_full(
                    tokenizer, model_input["input_ids"].size(1)
                ),
                use_cache=True,
                tokenizer=tokenizer,
                do_sample=False,
                **args
            )
            ts = generate_dict.sequences
            ntokens = sum(sum(helper(tid) for tid in t) for t in ts)
            increment_token_counter(ntokens - input_ntokens)
            rs = [strip_bos(tokenizer.decode(t[begin_input_length:], skip_special_tokens=True)) for t in ts]

        if return_hiddens:
            # Select features for last token by ignoring padding tokens
            eos_idxs = []
            for t in ts:
                suffix = t[len(model_input["input_ids"][0]) :]
                if tokenizer.eos_token_id in suffix:
                    eos_idxs.append(suffix.tolist().index(tokenizer.eos_token_id))
                else:
                    eos_idxs.append(len(suffix))
            layer_idx = -1
            layer = [
                generate_dict.hidden_states[eos_idxs[i] - 1][layer_idx][i, -1, :]
                for i in range(num)
            ]
            layer = torch.stack(layer, dim=0).reshape(num, -1)
            return rs, layer
        return rs

    def generate(prompt: str, num: int, return_hiddens=False, **kwargs) -> List[str]:
        model_generation_search_args = (
            huggingface_generate.get_model_generation_search_args(num)
        )
        model_generation_args = {
            **model_generation_token_args,
            **model_generation_search_args,
        }
        return gen(prompt, model_generation_args, num, return_hiddens, **kwargs)
    
    def generate_by_vllm(prompt: str, num: int, **kwargs) -> List[str]:
        # messages = [{"role": "system", "content": "You are a helpful assistant."},
        #             {"role": "user", "content": prompt}]
        messages = [{"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate"},
                    {"role": "user", "content": prompt}]
        # for qwencoder temperature=1.1, for glm temperature=1.0
        sampling_params = SamplingParams(n=num, max_tokens=max_new_tokens, temperature=1)
        outputs = model.chat(messages,
                   sampling_params=sampling_params,
                   use_tqdm=False)
        outputs = [output.text for output in outputs[0].outputs]
        return outputs
    
    def generate_with_stopstrings(prompt: str, state: str, num: int, return_hiddens=False, **kwargs) -> List[str]:
        model_generation_search_args = (
            huggingface_generate.get_model_generation_search_args(num)
        )
        model_generation_args = {
            **model_generation_token_args_with_stopstrings,
            **model_generation_search_args,
        }
        return gen(prompt, model_generation_args, num, return_hiddens, **kwargs)

    def generate_full(prompt: str, **kwargs) -> str:
        streamer = TextStreamer(tokenizer)
        model_generation_search_args = (
            huggingface_generate.get_model_generation_search_args(1)
        )
        model_input = tokenizer(prompt, return_tensors="pt").to("cuda")
        input_ntokens = model_input["input_ids"].size(1)
        all_args = {
            **(dict(streamer=streamer, max_new_tokens=512)),  # NOTE: used to be 1000
            "return_dict_in_generate": True,
            "stopping_criteria": huggingface_generate.get_stopping_criteria_full(
                tokenizer, model_input["input_ids"].size(1)
            ),
            **model_generation_search_args,
            **kwargs,
        }
        model.eval()
        r = None
        with torch.no_grad():
            model_result = model.generate(**model_input, **all_args)
            ts = model_result.sequences

            def helper(tid):
                return tid not in tokenizer.all_special_ids

            ntokens = sum(sum(helper(tid) for tid in t) for t in ts)
            increment_token_counter(ntokens - input_ntokens)
            r = strip_bos(tokenizer.decode(ts[0], skip_special_tokens=True))
        return r

else:
    assert False
