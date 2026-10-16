"""How far one request's tool loop may run."""

# The most model-to-tool rounds a request may ask for, and the ceiling a
# workspace's code execution policy may only lower.
MAX_TOOL_ITERATIONS_CAP = 25
