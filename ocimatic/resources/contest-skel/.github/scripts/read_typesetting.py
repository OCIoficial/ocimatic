# Read typeseting fom contest.toml, the toml library should be installed before executing this action
import toml

data = toml.load("contest.toml")
typesetting = data.get("contest", {}).get("typesetting", "")
print(f"typesetting: {data['contest'].get('typesetting', '')}")
print(f"::set-output name=typesetting::{typesetting}")
