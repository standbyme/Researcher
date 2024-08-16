# WhizReviewer

This repository contains the code for inference, detection, and evaluation of the WhizReviewer model, a large language model designed for academic paper review in the field of machine learning.

### Table of Contents

1. [Introduction](#introduction)
2. [Model Specifications](#model-specifications)
3. [Installation](#installation)
5. [Inference](#inference)
6. [Detection](#detection)
7. [Evaluation](#evaluation)
8. [Ethical Considerations](#ethical-considerations)
9. [Limitations](#limitations)
10. [Intended Uses](#intended-uses)
11. [License](#license)

### Introduction

WhizReviewer is a generative large language model that has undergone additional supervised training to provide expert-level review comments for academic papers.

### Model Specifications

- **Release Date**: August 16, 2024

- **Knowledge Cutoff Date**: January 2024

  |            Model Name             |                 Pre-training Language Model                  |                           HF Link                            |  MS Link   |
  | :-------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :--------: |
  |    WhizReviewer-ML-Llama3.1-8B    | [Llama3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) | [🤗 link](https://huggingface.co/WestlakeNLP/WhizReviewer-ML-Llama3.1-8B) | [🤖 TODO]() |
  |   WhizReviewer-ML-Llama3.1-70B    | [Llama3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct) | [🤗 link](https://huggingface.co/WestlakeNLP/WhizReviewer-ML-Llama3.1-70B) | [🤖 TODO]() |
  |     WhizReviewer-ML-Pro-123B      | [Mistral-Large-2](https://huggingface.co/mistralai/Mistral-Large-Instruct-2407) | [🤗 link](https://huggingface.co/WestlakeNLP/WhizReviewer-ML-Pro-123B) | [🤖 TODO]() |
  | WhizReviewer-Science-Llama3.1-8B  | [Llama3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) |                          [🤗 TODO]()                          | [🤖 TODO]() |
  | WhizReviewer-Science-Llama3.1-70B | [Llama3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct) |                          [🤗 TODO]()                          | [🤖 TODO]() |
  |   WhizReviewer-Science-Pro-123B   | [Mistral-Large-2](https://huggingface.co/mistralai/Mistral-Large-Instruct-2407) |                          [🤗 TODO]()                          | [🤖 TODO]() |

  #### 

### Training

We use the [snowflake-arctic](https://github.com/Snowflake-Labs/snowflake-arctic) code for training.



### Inference

The models included in this repository can be used with the `transformers` or `vllm` code libraries.

To generate Review comments, we need a long context (**14000 tokens for Input and 5000 tokens for Output**), please ensure you have enough GPU memory. Here are our recommended configurations:

|          Model Name          | Recommended Config (bs>=5) |          Minimum Config (bs=1)          |
| :--------------------------: | :------------------------: | :-------------------------------------: |
| WhizReviewer-ML-Llama3.1-8B  |    2 x A100/H100 (bf16)    | 1 x A100/H100 (int8) / 1 x A6000 (int4) |
| WhizReviewer-ML-Llama3.1-70B |    8 x A100/H100 (bf16)    |          4 x A100/H100 (bf16)           |
|   WhizReviewer-ML-Pro-123B   |    8 x A100/H100 (bf16)    |          4 x A100/H100 (bf16)           |

##### Getting Your Paper Text

If you can provide the original Latex version or Markdown version of your paper, that would be ideal, and you can skip this step.

If you only have the PDF version of the paper, you need to convert it to Markdown or Latex format first. We recommend using one of the following two methods for conversion:

**Online** You don't need to download any models, just register and get free tokens from [doc2x](https://doc2x.noedgeai.com/?inviteCode=WE5L94), then make sure your `pdfdeal` is the latest version: `pip install --upgrade pdfdeal`

```python
from pdfdeal import Doc2X
from pdfdeal import get_files
client = Doc2X(apikey='xxx') # apikey from doc2x
file_list, rename = get_files(path=r"path/PDF", mode="pdf", out="md")
success, failed, flag = client.pdfdeal(
    pdf_file=file_list,
    output_path=r"OutputPath/PDF",
    output_format='md',
    output_names=rename,
)
print(success)
print(failed)
print(flag)
```

At this point, you will be able to view the markdown format of the paper.

**Offline** If you need to run locally, we recommend using [MagicPDF](https://github.com/magicpdf/Magic-Doc). First, please follow the relevant guide to install it, then you will be able to use the code below to convert PDF paper files to markdown format:

```python
from magic_doc.docconv import DocConverter, S3Config
converter = DocConverter(s3_config=None)
markdown_cotent, time_cost = converter.convert("path/PDF", conv_timeout=300)
```

##### Using with transformers

Starting from `transformers >= 4.44.0`, first make sure your `transformers` is updated: `pip install -U transformers`

```python
import transformers
import torch
import re

def process_text(text, skip_appendix=True):
    pattern = re.compile(r"Under review as a conference paper at ICLR 2024", re.IGNORECASE)
    text = pattern.sub("", text)

    pattern = re.compile(r"Published as a conference paper at ICLR 2024", re.IGNORECASE)
    text = pattern.sub("", text)
    
    if skip_appendix:
        match = re.search(r"REFERENCES", text, re.IGNORECASE)

        if match:
            # Truncate the text at "REFERENCES"
            text = text[:match.start()]

    match = re.search(r"ABSTRACT", text, re.IGNORECASE)

    if match:
        text = text[match.start():]

    return text.strip()
    
model_id = "WestlakeNLP/WhizReviewer-ML-Pro-123B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

system_prompt = \
"""You are an expert academic reviewer tasked with providing a thorough and balanced evaluation of research papers. For each paper submitted, conduct a comprehensive review addressing the following aspects:

1. Summary: Briefly outline main points and objectives.
2. Soundness: Assess methodology and logical consistency.
3. Presentation: Evaluate clarity, organization, and visual aids.
4. Contribution: Analyze significance and novelty in the field.
5. Strengths: Identify the paper's strongest aspects.
6. Weaknesses: Point out areas for improvement.
7. Questions: Pose questions for the authors.
8. Rating: Score 1-10, justify your rating.
9. Meta Review: Provide overall assessment and recommendation (Accept/Reject).

Maintain objectivity and provide specific examples from the paper to support your evaluation.

You need to fill out **4** review opinions."""


markdown_context = "xxxxxxx" # Your paper's context
markdown_context = process_text(markdown_context, skip_appendix=True) # We suggest to skip appendix.

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": markdown_context},
]

outputs = pipeline(
    messages,
    max_new_tokens=4096,
)
print(outputs[0]["generated_text"][-1])
```

##### Using with vllm

Compared to `transformers`, we more strongly recommend using `vllm` for fast text generation. Usually, it can complete generation within 2 minutes: `pip install -U vllm`.

```python
from vllm import LLM, SamplingParams
import torch
import re

def process_text(text, skip_appendix=True):
    pattern = re.compile(r"Under review as a conference paper at ICLR 2024", re.IGNORECASE)
    text = pattern.sub("", text)

    pattern = re.compile(r"Published as a conference paper at ICLR 2024", re.IGNORECASE)
    text = pattern.sub("", text)
    
    if skip_appendix:
        match = re.search(r"REFERENCES", text, re.IGNORECASE)
    
        if match:
            # Truncate the text at "REFERENCES"
            text = text[:match.start()]
    
    match = re.search(r"ABSTRACT", text, re.IGNORECASE)
    
    if match:
        text = text[match.start():]
    
    return text.strip()

model_id = "WestlakeNLP/WhizReviewer-ML-Pro-123B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(
        model=model_name,
        tensor_parallel_size=8,
        max_model_len=16000,
        gpu_memory_utilization=0.95,
    )

system_prompt = \
"""You are an expert academic reviewer tasked with providing a thorough and balanced evaluation of research papers. For each paper submitted, conduct a comprehensive review addressing the following aspects:

1. Summary: Briefly outline main points and objectives.
2. Soundness: Assess methodology and logical consistency.
3. Presentation: Evaluate clarity, organization, and visual aids.
4. Contribution: Analyze significance and novelty in the field.
5. Strengths: Identify the paper's strongest aspects.
6. Weaknesses: Point out areas for improvement.
7. Questions: Pose questions for the authors.
8. Rating: Score 1-10, justify your rating.
9. Meta Review: Provide overall assessment and recommendation (Accept/Reject).

Maintain objectivity and provide specific examples from the paper to support your evaluation.

You need to fill out **4** review opinions."""


markdown_context = "xxxxxxx" # Your paper's context
markdown_context = process_text(markdown_context, skip_appendix=True) # We suggest to skip appendix.

sampling_params = SamplingParams(temperature=0.4, top_p=0.95, max_tokens=4000)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": markdown_context},
]

input_ids = tokenizer.apply_chat_template(messages+[{'role':'assistant','content':'\n\n## Reviewer\n'}], tokenize=False,add_generation_prompt=True)[:-4]
outputs = llm.generate([input_ids], sampling_params)
```

For more usage methods, please refer to the [vLLM](https://docs.vllm.ai/en/latest/) documentation.



### Detection

To detect potential misuse of the WhizReviewer model:

```python
from whizreviewer.fast_detect_gpt import WhizReviewerDetector

detector = WhizReviewerDetector()

text = "Suspected review text..."
is_whizreviewer, confidence = detector.detect(text)
print(f"Is WhizReviewer generated: {is_whizreviewer}")
print(f"Confidence: {confidence}")
```

or CLI format:

```
python whizreviewer/fast_detect_gpt.py --sentence "Hello words" --model_name "gpt2" --device "cpu"
```



#### Acc

We mixed 300 review comment samples from ICLR2024 and generated samples from WhizReviewer-ML as the evaluated dataset, with Llama-3.1-8B as the reference model. Detect Acc indicates the accuracy of being correctly detected by Fast-Detect-GPT.

| Model                        | Detect Acc |
| ---------------------------- | ---------- |
| WhizReviewer-ML-Llama3.1-8B  | 98.43      |
| WhizReviewer-ML-Llama3.1-70B | 99.47      |
| WhizReviewer-ML-Pro-123B     | 95.14      |



### Evaluation

To evaluate the performance of the WhizReviewer-ML-Pro-123B model:

```python
from whizreviewer.evaluation import WhizReviewerEvaluator

evaluator = WhizReviewerEvaluator(
    test_data_path="path/to/test/data"
)

results = evaluator.evaluate()
print(results)
```

The evaluation script will output metrics such as:
- Decisions (Accept/Reject) Accuracy
- Score Average Absolute Difference
- Score Perfect Match Rate
- Score Average Accuracy



| Metric                        | [WhizReviewer-ML-Llama3.1-8B](https://huggingface.co/WestlakeNLP/WhizReviewer-ML-Llama3.1-8B) | [WhizReviewer-ML-Llama3.1-70B](https://huggingface.co/WestlakeNLP/WhizReviewer-ML-Llama3.1-70B) | [WhizReviewer-ML-Pro-123B](https://huggingface.co/WestlakeNLP/WhizReviewer-ML-Pro-123B) |
| ----------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Decisions (Accept/Reject) Acc | 59.41%                                                       | 61.58%                                                       | **74.55%**                                                   |
| Score Avg Abs                 | 1.24                                                         | 1.28                                                         | **1.05**                                                     |
| Score Min Abs                 | 1.31                                                         | **1.18**                                                     | 1.45                                                         |
| Score Max Abs                 | 1.73                                                         | 1.71                                                         | **1.01**                                                     |
| Score Perfect Match           | 3.23%                                                        | 1.47%                                                        | **3.65%**                                                    |
| Score Avg Acc                 | 7.93%                                                        | 6.83%                                                        | **10.94%**                                                   |
| Score Min Acc                 | 36.96%                                                       | **42.70%**                                                   | 31.77%                                                       |
| Score Max Acc                 | 24.73%                                                       | 23.69%                                                       | **49.09%**                                                   |

We instruct the WhizReviewer-ML model to simulate reviewers from low-scoring to high-scoring, generating review comments and final scores in sequence. After collecting all review comments, a Meta-Reviewer is generated, which can predict the final acceptance result. In the evaluation results, Decisions Acc represents the accuracy of predicting the correct outcome given a paper, while Score Avg Abs represents the absolute difference between the average predicted score and the original score.

### Ethical Considerations

1. Academic Integrity: This model should not replace the real peer review process. Use it only as an auxiliary means for self-improvement and learning.
2. Fairness: Be aware of potential biases, especially for interdisciplinary or emerging field research.
3. Responsible Use: Do not use this model to produce false review opinions or manipulate the academic evaluation process.
4. Transparency: When using content generated by this model in any public setting, clearly state the WhizReviewer source.

### Limitations

1. Knowledge Cutoff: The model's knowledge is cut off in January 2024.
2. Pure Text Limitations: Cannot directly parse or evaluate images, charts, or complex formulas.
3. Depth in Specialized Fields: May not be as accurate as human experts in very specialized sub-fields.
4. Lack of Real-time Information: Cannot access real-time academic databases or the latest published papers.
5. Disciplinary Bias: May have preferences for certain disciplines or research methods.
6. Language and Cultural Limitations: May perform poorly with non-English papers or cross-cultural research.
7. Scoring Consistency: May have inconsistencies, especially with borderline cases or interdisciplinary research.

### Intended Uses

Expected Use Cases:
1. Paper Improvement
2. Writing Practice
3. Self-assessment Tool
4. Learning Aid
5. Feedback Simulation
6. Revision Guide
7. Concept Validator
8. Reward Model
9. Educational Resource
10. Research Assistant
11. Supplementary Tool

Out of Scope:
1. Official Reviews
2. Legal or Ethical Decisions
3. Factual Verification
4. Plagiarism Detection
5. Publication Decisions
6. Expert Consultation

### License

This project is licensed under the WhizReviewer License. Key points:

1. The model cannot be used for any formal review work.
2. Users must agree not to use the model for official reviews and publication decisions.

If you are unsure whether you meet our License requirements, please send your relevant application to zhuminjun@westlake.edu.cn for further inquiry.
