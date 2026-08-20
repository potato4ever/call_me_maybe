from llm_sdk.llm_sdk import Small_LLM_Model
import json
with open("tmp.json", mode="r+") as file:
    json.dump('{"name": .}', file, indent=2)
exit()
s = Small_LLM_Model()
tests = [
    "235",
    " hello",
    "function",
    "fn_add_numbers",
    "123",
    "3.14",
    "true",
    "false",
    '"hello"',
    "foobar",
    "HelloWorld",
]
for text in tests:
    ids = s.encode(text)[0].tolist()

    print("\nINPUT:", repr(text))
    print("IDS:", ids)

    for token_id in ids:
        print(
            token_id,
            s.decode([token_id])
        )
