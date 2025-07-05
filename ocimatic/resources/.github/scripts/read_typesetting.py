# Read typeseting fom contest.toml, the toml library should be installed before executing this action
import toml
import os

data = toml.load("contest.toml")
typesetting = data.get("contest", {}).get("typesetting", "")
print(f"typesetting: {data['contest'].get('typesetting', '')}")
# Use environment file for GitHub Actions output
if "GITHUB_OUTPUT" in os.environ:
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"typesetting={typesetting}\n")
