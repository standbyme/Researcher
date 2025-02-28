import re

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class DeepReviewer:
    """
    A class for generating automated academic peer reviews using DeepReviewer models.
    """

    def __init__(self,
                 model_size="14B",
                 custom_model_name=None,
                 device="cuda",
                 tensor_parallel_size=1,
                 gpu_memory_utilization=0.95):
        """
        Initialize the DeepReviewer.

        Args:
            model_size (str): Size of the default model to use. Options: "14B", "70B", "123B"
            custom_model_name (str, optional): Custom model name to override default mapping
            device (str): Device to run the model on. Default is "cuda"
            tensor_parallel_size (int): Number of GPUs to use for tensor parallelism
            gpu_memory_utilization (float): Fraction of GPU memory to use
        """
        model_mapping = {
            "14B": "WestlakeNLP/DeepReviewer-14B",
            "7B": "WestlakeNLP/DeepReviewer-7B",
        }

        # Determine model name
        if custom_model_name:
            model_name = custom_model_name
        else:
            if model_size not in model_mapping:
                raise ValueError(f"Invalid model size. Choose from {list(model_mapping.keys())}")
            model_name = model_mapping[model_size]

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model using vLLM
        self.model = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=70000,
            gpu_memory_utilization=gpu_memory_utilization
        )

        # Store model configuration for reference
        self.model_name = model_name
        self.model_config = {
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization
        }

    def _generate_system_prompt(self, mode="Standard Mode", reviewer_num=4):
        """
        Generate the system prompt based on the review mode and number of reviewers.

        Args:
            mode (str): Review mode. Options: "Fast Mode", "Standard Mode", "Best Mode"
            reviewer_num (int): Number of reviewers to simulate

        Returns:
            str: System prompt for the specified mode
        """
        simreviewer_prompt = "When you simulate different reviewers, write the sections in this order: Summary, Soundness, Presentation, Contribution, Strengths, Weaknesses, Suggestions, Questions, Rating and Confidence."

        if mode == "Best Mode":
            prompt = f"""You are an expert academic reviewer tasked with providing a thorough and balanced evaluation of research papers. Your thinking mode is Best Mode. In this mode, you should aim to provide the most reliable review results by conducting a thorough analysis of the paper. I allow you to use search tools to obtain background knowledge about the paper - please provide three different questions. I will help you with the search. After you complete your thinking, you should review by simulating {reviewer_num} different reviewers, and use self-verification to double-check any paper deficiencies identified. Finally, provide complete review results."""
            return prompt + simreviewer_prompt
        elif mode == "Standard Mode":
            prompt = f"""You are an expert academic reviewer tasked with providing a thorough and balanced evaluation of research papers. Your thinking mode is Standard Mode. In this mode, you should review by simulating {reviewer_num} different reviewers, and use self-verification to double-check any paper deficiencies identified. Finally, provide complete review results."""
            return prompt + simreviewer_prompt
        elif mode == "Fast Mode":
            return "You are an expert academic reviewer tasked with providing a thorough and balanced evaluation of research papers. Your thinking mode is Fast Mode. In this mode, you should quickly provide the review results."
        else:
            return "You are an expert academic reviewer tasked with providing a thorough and balanced evaluation of research papers."

    def evaluate(self, paper_context, mode="Standard Mode", reviewer_num=4, max_tokens=35000):
        """
        Generate a peer review for the given academic paper.

        Args:
            paper_context (str): The paper content to review
            mode (str): Review mode. Options: "Fast Mode", "Standard Mode", "Best Mode"
            reviewer_num (int): Number of reviewers to simulate
            max_tokens (int): Maximum number of tokens to generate

        Returns:
            dict: Generated review with scores and feedback
        """
        # Prepare system prompt
        system_prompt = self._generate_system_prompt(mode, reviewer_num)

        if type(paper_context) == str:
            paper_context = [paper_context]



        generated_reviews = []
        batch_size = 10
        for n in range(0,len(paper_context),batch_size):
            # Apply chat template
            prompts = []
            for r in range(min(batch_size, len(paper_context) - n)):
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": paper_context[n+r]}
                ]
                input_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompts.append(input_text)
            # Prepare sampling parameters
            sampling_params = SamplingParams(
                temperature=0.4,
                top_p=0.95,
                max_tokens=45000
            )

            # Generate review
            outputs = self.model.generate(
                prompts,
                sampling_params
            )

            # Process generated review text
            for output_num in range(len(outputs)):
                # Process generated text
                generated_text = outputs[output_num].outputs[0].text
                # Use existing CycleResearcher utility to parse generated text
                print(generated_text)
                review = self._parse_review(generated_text)
                generated_reviews.append(review)

        return generated_reviews

    def _parse_review(self, generated_text):
        """
        Parse the generated review text into structured format.

        Args:
            generated_text (str): Raw generated review text

        Returns:
            dict: Structured review with metadata and reviews
        """
        result = {
            "raw_text": generated_text,
            "reviews": [],
            "meta_review": {},
            "decision": ""
        }

        # Extract meta review if present
        meta_review_match = re.search(r'\\boxed_review\{(.*?)\n}', generated_text, re.DOTALL)
        if meta_review_match:
            result["meta_review"]['content'] = meta_review_match.group(1).strip()
            section = meta_review_match.group(1).strip()
            # Extract summary
            summary_match = re.search(r'## Summary:\s+(.*?)(?=##|\Z)', section, re.DOTALL)
            if summary_match:
                result["meta_review"]["summary"] = summary_match.group(1).strip()

            # Extract rating
            rating_match = re.search(r'## Rating:\s+(.*?)(?=##|\Z)', section, re.DOTALL)
            if rating_match:
                rating_text = rating_match.group(1).strip()
                # Try to extract a numerical rating (1-10)
                number_match = re.search(r'(\d+(?:\.\d+)?)', rating_text)
                if number_match:
                    result["meta_review"]["rating"] = float(number_match.group(1))
                else:
                    result["meta_review"]["rating"] = rating_text

            # Extract other sections as needed
            for section_name in ["Soundness", "Presentation", "Contribution",
                                 "Strengths", "Weaknesses", "Suggestions", "Questions"]:
                section_match = re.search(f'## {section_name}:\s+(.*?)(?=##|\Z)', section, re.DOTALL)
                if section_match:
                    result["meta_review"][section_name.lower()] = section_match.group(1).strip()

        # Extract simulated reviewers' feedback
        simreviewer_match = re.search(r'\\boxed_simreviewers\{(.*?)\n}', generated_text, re.DOTALL)
        if simreviewer_match:
            simreviewer_text = simreviewer_match.group(1).strip()
            # Split into individual reviewer sections
            reviewer_sections = re.split(r'## Reviewer \d+', simreviewer_text)
            # Skip the first empty section if it exists
            if reviewer_sections and not reviewer_sections[0].strip():
                reviewer_sections = reviewer_sections[1:]

            for i, section in enumerate(reviewer_sections):
                review = {
                    "reviewer_id": i + 1,
                    "text": section.strip()
                }

                # Extract summary
                summary_match = re.search(r'## Summary:\s+(.*?)(?=##|\Z)', section, re.DOTALL)
                if summary_match:
                    review["summary"] = summary_match.group(1).strip()

                # Extract rating
                rating_match = re.search(r'## Rating:\s+(.*?)(?=##|\Z)', section, re.DOTALL)
                if rating_match:
                    rating_text = rating_match.group(1).strip()
                    # Try to extract a numerical rating (1-10)
                    number_match = re.search(r'(\d+(?:\.\d+)?)', rating_text)
                    if number_match:
                        review["rating"] = float(number_match.group(1))
                    else:
                        review["rating"] = rating_text

                # Extract other sections as needed
                for section_name in ["Soundness", "Presentation", "Contribution",
                                     "Strengths", "Weaknesses", "Suggestions", "Questions"]:
                    section_match = re.search(f'## {section_name}:\s+(.*?)(?=##|\Z)', section, re.DOTALL)
                    if section_match:
                        review[section_name.lower()] = section_match.group(1).strip()

                result["reviews"].append(review)

        # Extract decision if present
        decision_match = re.search(r'## Decision:\s*\n\s*(\w+)', generated_text)
        if decision_match:
            result["decision"] = decision_match.group(1).strip()

        return result
