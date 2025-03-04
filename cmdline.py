from dataclasses import dataclass, field, asdict
from transformers import HfArgumentParser, set_seed

@dataclass
class CommonArguments:
    dataset_name: str = field(default="spider", metadata={"help": "Choose dataset from 'spider', 'bird'"})
    model_host: str = field(default="huggingface", metadata={"help": "Choose between 'huggingface' or 'openai'"})
    # base_model_name: str = field(default="./models/seeklhy/codes-7b", metadata={"help": "LLM name"})
    # base_model_name: str = field(default="./models/codellama/CodeLlama-7b-Instruct-hf", metadata={"help": "LLM name"})
    # base_model_name: str = field(default="./models/Qwen/Qwen2.5-Coder-7B-Instruct", metadata={"help": "LLM name"})
    base_model_name: str = field(default="./models/THUDM/glm-4-9b-chat", metadata={"help": "LLM name"})
    beam_search: bool = field(default=True, metadata={"help": "Set to True to use beam search instead of sampling"})
    max_tokens: int = field(default=4096, metadata={"help": "max tokens for llm generation"})
    max_new_tokens: int = field(default=256, metadata={"help": "max new tokens for llm generation"})
    beam_search_num: int = field(default=8, metadata={"help": "number for llm generation"})
    model_arg_topk: int = field(default=2, metadata={"help": "Specify top k parameter for llm generation"})
    model_arg_topp: float = field(default=0.9, metadata={"help": "Specify top p parameter for llm generation"})
    model_arg_temp: float = field(default=1.0, metadata={"help": "Specify temperature parameter for llm generation"})
    experiment_name: str = field(default="run.py", metadata={"help": "Pick an experiment to run "})
    seed: int = field(default=None, metadata={"help": "Set the seed for reproducible behavior"})
    expansion_count: int = field(default=300, metadata={"help": "expansion count for tree search"})
    token_limit: int = field(default=None, metadata={"help": "Max amount of tokens before execution is stopped"})
    discovery_factor: float = field(default=1, metadata={"help": "Hyperparameter: discovery factor"})
    widen_policy_value: float = field(default=0.2, metadata={"help": "Hyperparameter: widen policy value"})
    infer_methd: str = field(default='normal', metadata={"help": "'vllm' or 'normal'"})

    def dict(self):
        return {k: str(v) for k,v in asdict(self).items()}

def get_args():
    parser = HfArgumentParser(CommonArguments)
    args = parser.parse_args_into_dataclasses()[0]
    return args

args = get_args()
if args.seed is not None:
    set_seed(args.seed)