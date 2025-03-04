import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Qwen2Tokenizer, 
    Qwen2ForCausalLM,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList
)
from trl import AutoModelForCausalLMWithValueHead
from peft import PeftModel
from model_config import (
    BASE_MODEL_NAME,
    PEFT_MODEL_PATH,
    PPO_MODEL_PATH,
    CUSTOM_STOP,
    SAME_FOR_MANY_SAMPLES,
    BEAM_SEARCH,
    MODEL_ARG_TOP_K,
    MODEL_ARG_TOP_P,
    MODEL_ARG_TEMP,
)
from typing import List, Tuple
from vllm import LLM
from cmdline import args

max_tokens=4096
max_tokens=4150

class StopWordCriteria(StoppingCriteria):
    def __init__(self, tokenizer, custom_stop, stop_word, l):
        self.custom_stop = custom_stop
        self.stop_word = stop_word
        self.tokenizer = tokenizer
        self.l = l

    def __call__(self, input_ids, scores, **kwargs):
        if not self.custom_stop:
            return False
        if input_ids.size(1) > self.l + 1:
            for token_id in input_ids[0, self.l + 1 :]:
                token = self.tokenizer.decode(token_id)
                if self.stop_word in token or "```" in token:
                    return True
        return False


def load_qwen_model(
    base_model_name: str = BASE_MODEL_NAME,
    ppo_model_path: str = PPO_MODEL_PATH,
    peft_model_path: str = PEFT_MODEL_PATH,
) -> Tuple[Qwen2ForCausalLM, PeftModel, Qwen2Tokenizer]:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    if ppo_model_path is None:
        base_model = Qwen2ForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_auth_token=True,
        )
        tokenizer = Qwen2Tokenizer.from_pretrained(
            base_model_name, trust_remote_code=True
        )
    else:
        base_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            ppo_model_path, quantization_config=bnb_config
        )
        tokenizer = Qwen2Tokenizer.from_pretrained(ppo_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = (
        PeftModel.from_pretrained(base_model, peft_model_path)
        if peft_model_path
        else base_model
    )
    return (base_model, model, tokenizer)


def load_model_vllm(base_model_name: str = BASE_MODEL_NAME):
    base_model = LLM(model=base_model_name, max_model_len=args.max_tokens, trust_remote_code=True)
    return base_model

def load_model(
    base_model_name: str = BASE_MODEL_NAME,
    ppo_model_path: str = PPO_MODEL_PATH,
    peft_model_path: str = PEFT_MODEL_PATH,
) -> Tuple[AutoModelForCausalLM, PeftModel, AutoTokenizer]:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    if ppo_model_path is None:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_auth_token=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=True
        )
    else:
        base_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            ppo_model_path, quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(ppo_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = (
        PeftModel.from_pretrained(base_model, peft_model_path)
        if peft_model_path
        else base_model
    )
    return (base_model, model, tokenizer)

def get_model_generation_token_args(
    tokenizer: AutoTokenizer, custom_stop: bool = CUSTOM_STOP
):
    return dict(
        min_length=5,
        max_new_tokens=256,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        # repetition_penalty=2.0
    )

def get_stopping_criteria(tokenizer: AutoTokenizer, input_len):
    STOP_WORD = "\n"
    return (
        StoppingCriteriaList(
            [StopWordCriteria(tokenizer, CUSTOM_STOP, STOP_WORD, input_len)]
        ),
    )


def get_stopping_criteria_full(tokenizer: AutoTokenizer, input_len):
    return (
        StoppingCriteriaList(
            [StopWordCriteria(tokenizer, True, tokenizer.eos_token, input_len)]
        ),
    )


def get_model_generation_search_args(num: int, beam_search: bool = BEAM_SEARCH):
    if beam_search:
        return dict(
            num_beams=num,
            num_beam_groups=num,
            diversity_penalty=0.9,
        )
    else:
        if MODEL_ARG_TOP_K is not None:
            if MODEL_ARG_TOP_K == 0:
                top_k = None
            else:
                top_k = MODEL_ARG_TOP_K
        elif num > 1 and not SAME_FOR_MANY_SAMPLES:
            top_k = 50
        else:
            top_k = 7
        return dict(
            top_k=top_k,
            top_p=MODEL_ARG_TOP_P if MODEL_ARG_TOP_P is not None else 0.9,
            do_sample=False,
            temperature=(
                MODEL_ARG_TEMP
                if MODEL_ARG_TEMP is not None
                else 0.9 if num > 1 and not SAME_FOR_MANY_SAMPLES else 0.8
            ),
        )
