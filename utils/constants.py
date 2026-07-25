MODEL = "Qwen/Qwen2.5-3B-Instruct"
DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 500

ATTN_IMPL = "eager"

# Speculative decoding
GAMMA = 4  # number of draft tokens proposed per verification round
TEMPERATURE = 0.7
TOP_P = 0.9
