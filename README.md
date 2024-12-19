# LLM Deep Dive

Exploring LLMs by implementing and understanding all the key components, perform experiments and optimize it.


## Installation

`uv pip install -e .`

## Training Tokenizers

Run the following command to train the tokenizers:

`python src/llm/wmt_data_utils.py`

To configure the tokenizers, you can pass the following arguments check out the available arguments in the script.

`python src/llm/wmt_data_utils.py --help`


## Contribution

Lint/formatting:

Simply do `make` to run all the formatting and linting checks.
