# Meta-CoT

**Meta-CoT** is a generalizable CoT prompting method in mixed-task scenarios where the type of input questions is unknown. It consists of three phases: (i) *scenario identification*: categorizes the scenario of the input question; (ii) *demonstration selection*: fetches the ICL demonstrations for the categorized scenario; (iii) *answer derivation*: performs the answer inference by feeding the LLM with the prompt comprising the fetched ICL demonstrations and the input question.

![](pics/overview.png)


## Requirements

Install all required python dependencies:

```
pip install -r requirements.txt
```

## Datasets

Download the datasets from the following repository and put them under `dataset` and `log`:

```
https://github.com/kojima-takeshi188/zero_shot_cot/tree/main/dataset
https://github.com/kojima-takeshi188/zero_shot_cot/tree/main/log
```

## Implementations
Input your own openai api key in `llm_utils.py`.

### Mixed Data Preprocessing

```
python mixed_preprocessing.py \
--input_style que \
--output_style cat-form
#if you want to run preliminary experiments for scenario identification, you can set run_test to True.
```

### Demos Construction

```
mkdir -p demos_inference/diversity-based

python demos_inference.py \
--demo_sampling_method diversity \
--output_style cat-form
```

### Run Inference

```
python run.py
```
