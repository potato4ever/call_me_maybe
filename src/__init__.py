"""
call_me_maybe: translate natural language prompts
into structured function calls.

This package uses constrained decoding
(token-by-token logit masking) on top of the
Small_LLM_Model wrapper from llm_sdk to guarantee
syntactically and semantically valid
JSON output, even though the
underlying model is a small (0.6B parameter) LLM.
"""
