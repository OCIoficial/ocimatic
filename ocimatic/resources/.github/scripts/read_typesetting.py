import tomllib
import os

with open("contest.toml", "rb") as f:
    data = tomllib.load(f)
typesetting = data.get("contest", {}).get("typesetting", "")
print(f"typesetting: {data['contest'].get('typesetting', '')}")
if "GITHUB_OUTPUT" in os.environ:
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"typesetting={typesetting}\n")
