#!/usr/bin/env python3

import subprocess


completed = subprocess.run(
    [
        "git",
        "ls-remote",
        "https://github.com/ocioficial/ocimatic.git",
        "HEAD",
    ],
    stdout=subprocess.PIPE,
)

commit = completed.stdout.decode("utf-8").split()[0]

msg = f"""
Competencia generada con `ocimatic` version `{commit}`. Para regenerar casos de prueba y enunciados, instala esta version usando el siguiente comando:

```bash
pip install git+https://github.com/OCIoficial/ocimatic@{commit}
```
"""


print(msg)
