# This file shows how to generate multiple test cases in a single invocation of a script. It
# generates the following 3 cases.

# ```
# 0 1
# ```

# ```
# 1 2
# ```

# ```
# 2 3
# ```

import sys

# We use the file separator control code to signal the end of a test case.
# https://en.wikipedia.org/wiki/C0_and_C1_control_codes#Field_separators
FS = chr(28)

for i in range(3):
    print(i, i + 1)
    sys.stdout.write(FS)
