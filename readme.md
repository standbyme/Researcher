# CycleResearcher: AI-powered Research and Review Ecosystem [ICLR 2025]

<div align="center">
  
[![GitHub stars](https://img.shields.io/github/stars/zhu-minjun/Researcher)](https://github.com/zhu-minjun/Researcher/stargazers) 
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/) 
[![arXiv](https://img.shields.io/badge/arXiv-2411.00816-b31b1b.svg)](https://arxiv.org/abs/2411.00816)
[![OpenReview](https://img.shields.io/badge/OpenReview-ICLR2025-8b1a1a.svg)](https://openreview.net/forum?id=bjcsVLoHYs)
[![Homepage](https://img.shields.io/badge/Homepage-ai--researcher.cn-green.svg)](http://ai-researcher.cn)


<div align="center">
  <img src="img/ai-research.png" alt="AI Research Ecosystem" width="90%">
</div>
</div>

## 🔍 Overview

CycleResearcher is a comprehensive open-source ecosystem for AI-powered academic research and review. Our system features three integrated components:

- **CycleResearcher**: Generates high-quality research papers
- **CycleReviewer**: Provides detailed academic reviews
- **DeepReviewer**: Delivers multi-perspective review simulations with self-verification

By creating a complete feedback loop between research generation and evaluation, we aim to:

- 🤖 Automate academic research processes
- 📝 Provide rigorous, multi-perspective research reviews
- 🔄 Establish research-review feedback loops
- 🚀 Accelerate scientific discovery

<img src="img/method.png" alt="CycleResearcher Architecture" width="80%">

## 🚀 Getting Started

### Installation

```bash
pip install -e .
```

### Using CycleResearcher

```python
# Import necessary libraries
from ai_researcher import CycleResearcher
from ai_researcher.utils import print_paper_summary

# Initialize CycleResearcher with default 12B model
researcher = CycleResearcher(model_size="12B")

# Load references from BibTeX file
with open('cycleresearcher_references.bib', 'r') as f:
    references_content = f.read()

# Generate paper with specific references
generated_papers = researcher.generate_paper(
    topic="AI Researcher",
    references=references_content,
    n=1  # Generate a single paper
)

# Print summary of generated paper
print_paper_summary(generated_papers[0])
```

### Using CycleReviewer

```python
# Import necessary libraries
from ai_researcher import CycleReviewer

# Initialize CycleReviewer with default 8B model
reviewer = CycleReviewer(model_size="8B")

# Review a paper (assuming paper_text contains the paper content)
review_results = reviewer.evaluate(paper_text)

# Print review results
print(f"Average score: {review_results[0]['avg_rating']}")
print(f"Decision: {review_results[0]['paper_decision']}")
```

### Using DeepReviewer

```python
# Import necessary libraries
from ai_researcher import DeepReviewer

# Initialize DeepReviewer with 14B model
deep_reviewer = DeepReviewer(model_size="14B")

# Review a paper with multiple simulated reviewers in Standard Mode
review_results = deep_reviewer.evaluate(
    paper_text,
    mode="Standard Mode",  # Options: "Fast Mode", "Standard Mode", "Best Mode"
    reviewer_num=4         # Simulate 4 different reviewers
)

# Print review results
for i, review in enumerate(review_results[0]['reviews']):
    print(f"Reviewer {i+1} Rating: {review.get('rating', 'N/A')}")
    print(f"Reviewer {i+1} Summary: {review.get('summary', 'N/A')[:100]}...")
```
<img src="img/deepreviewer.png" alt="CycleResearcher Architecture" width="80%">

## 📊 Model Evaluation

<div class="evaluation-grid">
  <div class="evaluation-card">
    <h3>CycleResearcher</h3>
    <img src="img/cycleresearcher.png" alt="CycleResearcher Evaluation" width="100%">
    <p>CycleResearcher-12B achieves an average score of 5.36, approaching the 5.69 average for conference-accepted papers and surpassing AI Scientist's score of 4.31.</p>
  </div>
  
  <div class="evaluation-card">
    <h3>CycleReviewer</h3>
    <img src="img/cyclereviewer.png" alt="CycleReviewer Evaluation" width="100%">
    <p>CycleReviewer outperforms both proprietary systems and human experts with a 48.77% reduction in Proxy MSE and a 26.89% reduction in Proxy MAE compared to human reviewers. With a decision accuracy of 74.24%, our model demonstrates a significant lead over other closed-source systems.</p>
  </div>
  
  <div class="evaluation-card">
    <h3>DeepReviewer</h3>
    <img src="img/deepreviewer.png" alt="DeepReviewer Evaluation" width="100%">
    <p>DeepReviewer provides multi-perspective simulation with self-verification, enabling more comprehensive and balanced feedback. It offers three distinct review modes: Fast Mode, Standard Mode, and Best Mode to accommodate different use cases.</p>
  </div>
</div>

## 🧠 Models & Datasets

### Models Overview

<details open>
<summary><b>CycleResearcher Models</b></summary>
<table>
  <tr>
    <th>Model Name</th>
    <th>Pre-training Language Model</th>
    <th>HF Link</th>
  </tr>
  <tr>
    <td>CycleResearcher-ML-12B</td>
    <td><a href="https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407">Mistral-Nemo-Instruct-2407</a></td>
    <td><a href="https://huggingface.co/WestlakeNLP/CycleResearcher-ML-12B">🤗 link</a></td>
  </tr>
  <tr>
    <td>CycleResearcher-ML-72B</td>
    <td><a href="https://huggingface.co/Qwen/Qwen2.5-72B-Instruct">Qwen2.5-72B-Instruct</a></td>
    <td><a href="https://huggingface.co/WestlakeNLP/CycleResearcher-ML-72B">🤗 link</a></td>
  </tr>
  <tr>
    <td>CycleResearcher-ML-123B</td>
    <td><a href="https://huggingface.co/mistralai/Mistral-Large-Instruct-2407">Mistral-Large-2</a></td>
    <td><a href="https://huggingface.co/WestlakeNLP/CycleResearcher-ML-123B">🤗 link</a></td>
  </tr>
</table>
</details>

<details open>
<summary><b>CycleReviewer Models</b></summary>
<table>
  <tr>
    <th>Model Name</th>
    <th>Pre-training Language Model</th>
    <th>HF Link</th>
  </tr>
  <tr>
    <td>CycleReviewer-ML-Llama3.1-8B</td>
    <td><a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct">Llama3.1-8B-Instruct</a></td>
    <td><a href="https://huggingface.co/WestlakeNLP/CycleReviewer-ML-Llama3.1-8B">🤗 link</a></td>
  </tr>
  <tr>
    <td>CycleReviewer-ML-Llama3.1-70B</td>
    <td><a href="https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct">Llama3.1-70B-Instruct</a></td>
    <td><a href="https://huggingface.co/WestlakeNLP/CycleReviewer-ML-Llama3.1-70B">🤗 link</a></td>
  </tr>
  <tr>
    <td>CycleReviewer-ML-Pro-123B</td>
    <td><a href="https://huggingface.co/mistralai/Mistral-Large-Instruct-2407">Mistral-Large-2</a></td>
    <td><a href="https://huggingface.co/WestlakeNLP/CycleReviewer-ML-Pro-123B">🤗 link</a></td>
  </tr>
</table>
</details>

<details open>
<summary><b>DeepReviewer Models</b></summary>
<table>
  <tr>
    <th>Model Name</th>
    <th>Parameters</th>
    <th>HF Link</th>
  </tr>
  <tr>
    <td>DeepReviewer-7B</td>
    <td>7B</td>
    <td><a href="https://huggingface.co/WestlakeNLP/DeepReviewer-7B">🤗 link</a></td>
  </tr>
  <tr>
    <td>DeepReviewer-14B</td>
    <td>14B</td>
    <td><a href="https://huggingface.co/WestlakeNLP/DeepReviewer-14B">🤗 link</a></td>
  </tr>
</table>
</details>

### Datasets

<div align="center">
  <img src="img/dataset.png" alt="Datasets Overview" width="80%">
</div>

<table>
  <tr>
    <th>Dataset Name</th>
    <th>Train Data</th>
    <th>Test Data</th>
    <th>Description</th>
    <th>HF Link</th>
  </tr>
  <tr>
    <td>Review-5K</td>
    <td>4,189</td>
    <td>781</td>
    <td>Peer review dataset for CycleReviewer training</td>
    <td><a href="https://huggingface.co/WestlakeNLP/WhizReviewer-ML-Llama3.1-8B">🤗 link</a></td>
  </tr>
  <tr>
    <td>Research-14K</td>
    <td>12,696</td>
    <td>802</td>
    <td>Research paper dataset for CycleResearcher training</td>
    <td><a href="https://huggingface.co/WestlakeNLP/WhizReviewer-ML-Llama3.1-70B">🤗 link</a></td>
  </tr>
  <tr>
    <td>DeepReview-13K</td>
    <td>11,500+</td>
    <td>1,500+</td>
    <td>Multi-perspective review dataset for DeepReviewer training</td>
    <td><a href="https://huggingface.co/WestlakeNLP/DeepReview-13K">🤗 link</a></td>
  </tr>
</table>

## 💡 Features

### DeepReviewer Review Modes

DeepReviewer offers three distinct review modes to accommodate different use cases:

<div class="feature-cards">
  <div class="feature-card">
    <h4>🏃‍♂️ Fast Mode</h4>
    <p>Quick review generation for rapid feedback. Provides essential evaluation without multi-reviewer simulation.</p>
  </div>
  
  <div class="feature-card">
    <h4>🔄 Standard Mode</h4>
    <p>Default mode that simulates multiple reviewers and includes self-verification to ensure reliable assessments.</p>
  </div>
  
  <div class="feature-card">
    <h4>⭐ Best Mode</h4>
    <p>Most comprehensive mode with background knowledge search, multi-reviewer simulation, and self-verification for in-depth analysis.</p>
  </div>
</div>

### AI Detection

Detect if content was generated by AI models:

```python
from ai_researcher import AIDetector

# Initialize AI detector
detector = AIDetector(device='cpu')

# Analyze the generated paper
detection_result = detector.analyze_paper(paper)

print("Detection Results:")
print(f"Probability of AI generation: {detection_result['probability'] * 100:.2f}%")
print(f"Confidence Level: {detection_result['confidence_level']}")
```

## 📚 Tutorials and Demos

We have prepared comprehensive tutorials to help users understand and utilize our models:

- [Tutorial 1:](https://github.com/zhu-minjun/Researcher/blob/main/Tutorial/tutorial_1.ipynb) Getting Started with CycleResearcher 🚀
- [Tutorial 2:](https://github.com/zhu-minjun/Researcher/blob/main/Tutorial/tutorial_2.ipynb) Understanding CycleReviewer 📝
- [Tutorial 3:](https://github.com/zhu-minjun/Researcher/blob/main/Tutorial/tutorial_3.ipynb) Mastering DeepReviewer 🔍
- [Tutorial 4:](https://github.com/zhu-minjun/Researcher/blob/main/Tutorial/tutorial_4.ipynb) Creating an End-to-End Research Workflow 🔄

## 📄 License

This code and the models' weights are provided under the *CycleResearcher-License*. See the [LICENSE.md](LICENSE.md) file for details.

## 📚 Citation

If CycleResearcher is helpful to your work, please cite our paper:

```bibtex
@inproceedings{
weng2025cycleresearcher,
title={CycleResearcher: Improving Automated Research via Automated Review},
author={Yixuan Weng and Minjun Zhu and Guangsheng Bao and Hongbo Zhang and Jindong Wang and Yue Zhang and Linyi Yang},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=bjcsVLoHYs}
}
```

## 📮 Contact

- [Submit an Issue](https://github.com/zhu-minjun/Researcher/issues)
- Email: zhuminjun@westlake.edu.cn


