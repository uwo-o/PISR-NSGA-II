import re

with open("src/pde_problems.cpp", "r") as f:
    content = f.read()

# Replace all PI or PI_VAL with 3.14159265358979323846 to be safe
content = re.sub(r'\bPI\b', '3.14159265358979323846', content)
content = re.sub(r'\bPI_VAL\b', '3.14159265358979323846', content)

# I_COMPLEX is defined in common.hpp as const Complex I_COMPLEX(0.0, 1.0);
# But we need to make sure we don't have multiple definitions.

with open("src/pde_problems.cpp", "w") as f:
    f.write(content)
