from cmdline import args

MODEL_HOST = args.model_host
# MODEL_HOST = "huggingface"
# MODEL_HOST = "openai"

# Pick an OpenAI model:
OPENAI_MODEL = "gpt-4"

# Pick a base model hosted on HuggingFace:
BASE_MODEL_NAME = args.base_model_name

# Provide an optional checkpoint on top of the base model:
# PEFT_MODEL_PATH = args.peft_model_path
PEFT_MODEL_PATH = None

# Or pick an entire PPO model -- overrides all the above:
# PPO_MODEL_PATH = args.ppo_model_path
# PPO_MODEL_PATH = "./my_ppo_model"
PPO_MODEL_PATH = None

# Set to None for run.py to instead use base model (and optional peft):
# PPO_MODEL_PATH = None

# Set to True to use custom stop words:
# CUSTOM_STOP = args.custom_stop
CUSTOM_STOP = True

# Sample the same whether generating one or many samples
# SAME_FOR_MANY_SAMPLES = args.same_for_many_samples
SAME_FOR_MANY_SAMPLES = False

# Set to True to use beam search instead of sampling
# BEAM_SEARCH = args.beam_search
BEAM_SEARCH = True

# Can set to None
MODEL_ARG_TOP_K = args.model_arg_topk
MODEL_ARG_TOP_P = args.model_arg_topp
MODEL_ARG_TEMP = args.model_arg_temp
# MODEL_ARG_TOP_K = 0.0
# MODEL_ARG_TOP_P = 1.0
# MODEL_ARG_TEMP = 1.0
# MODEL_ARG_TOP_K = None
# MODEL_ARG_TOP_P = None
# MODEL_ARG_TEMP = None
