import json


def get_paper_from_generated_text(generated_text):
    """
    Parse and extract different sections from a generated academic paper text.
    The function expects the text to contain specific sections marked with '##'
    and LaTeX content.

    Args:
        generated_text (str): The full text of the generated paper containing
            marked sections and LaTeX content

    Returns:
        dict: A dictionary containing parsed paper sections and metadata
        None: If parsing fails or the content is not properly formatted
    """

    item = {}
    try:
        # Store the original generated text
        item['generated_text'] = generated_text

        # Extract main sections marked with '##'
        Motivation = generated_text.split('## Motivation')[1].split('## Main Idea')[0]
        Idea = generated_text.split('## Main Idea')[1].split('## Interestingness')[0]
        Interestingness = generated_text.split('## Interestingness')[1].split('## Feasibility')[0]
        Feasibility = generated_text.split('## Feasibility')[1].split('## Novelty')[0]
        Novelty = generated_text.split('## Novelty')[1].split('```latex')[0]

        # Extract and process LaTeX content
        latex = ''
        latex += generated_text.split('```latex')[1].split('```')[0]

        # Parse title and abstract from LaTeX
        title = latex.split(r'\title{')[1].split(r'\begin{abstract}')[0]
        abstract = latex.split(r'\begin{abstract}')[1].split('\end{abstract}')[0]

        # Extract and parse experimental setup
        Experimental_Setup = generated_text.split('## Experimental Setup')[1].split('```json')[1].split('```')[0]
        try:
            # Attempt to parse experimental setup as JSON
            Experimental_Setup = json.loads(Experimental_Setup)
        except:
            pass

        # Extract experimental results, handling two possible formats
        if '## Experimental_results' in generated_text:
            Experimental_results = generated_text.split('## Experimental_results')[1].split('```json')[1].split('```')[
                0]
        else:
            Experimental_results = generated_text.split('## Experimental Setup')[1].split('```json')[2].split('```')[0]

        # Add remaining LaTeX content after experimental results
        latex += generated_text.split(Experimental_results)[1]

        try:
            # Attempt to parse experimental results as JSON
            Experimental_results = json.loads(Experimental_results)
        except:
            pass

        # Process LaTeX content to extract main body
        # Stops at acknowledgment, conclusion, disclosure sections or clearpage
        latex_context = ''
        is_latex = False
        for l in latex.split('\n'):
            if 'section' in l.lower() and 'acknowledgment' in l.lower():
                is_latex = True
                break
            latex_context += l + '\n'
            if 'section' in l.lower() and 'conclusion' in l.lower():
                if is_latex:
                    break
                is_latex = True
            if 'section' in l.lower() and 'disclosure' in l.lower():
                if is_latex:
                    break
                is_latex = True
            if r'\clearpage' in l.lower():
                if is_latex:
                    break
                is_latex = True
                break

        # Populate the return dictionary with all extracted contents
        item['motivation'] = Motivation
        item['idea'] = Idea
        item['interestingness'] = Interestingness
        item['feasibility'] = Feasibility
        item['novelty'] = Novelty
        item['title'] = title
        item['abstract'] = abstract
        item['Experimental_Setup'] = Experimental_Setup
        item['Experimental_results'] = Experimental_results

        # Clean up LaTeX context by removing markdown code markers
        latex_context = latex_context.replace('```latex', '').replace('```json', '').replace('```', '')

        # Return the item only if valid LaTeX content was found
        if is_latex:
            item['latex'] = latex_context
            return item
        else:
            item['latex'] = ''
            return None

    except:
        # Return None if any parsing error occurs
        item['latex'] = ''
        return None