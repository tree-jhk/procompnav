"""
Prompts for Property-based Binary Partitioning (PBP) module.
"""

import math
from typing import List, Dict, Tuple


def build_property_prompt(descriptions: List[str], used_properties: List[str], category: str) -> str:
    """
    Build prompt for LLM to propose a discriminative property.
    
    Args:
        descriptions: List of object descriptions
        used_properties: List of already used properties
    
    Returns:
        Formatted prompt string
    """
    n = len(descriptions)
    k = math.ceil(n / 2)
    enumerated = [f"{i + 1}. {d}" for i, d in enumerate(descriptions)]

    used_block = "None" if not used_properties else ", ".join(used_properties)

    template = (
        f"You are given a list of [{category}] descriptions.\n"
        f"Task: Identify one clear property that is shared by about half of the [{category}s] in the list "
        f"(roughly {k} out of {n} [{category}s]). The other [{category}s] should not have this property.\n"
        "- The property should be based on visible appearance or nearby surrounding objects, not fine details.\n"
        "- If an object is key to the split, explicitly name it.\n"
        "List of descriptions:\n{descriptions}\n"
        "DO NOT USE: [{used_block}].\nExtract another property.\n"
        "Answer only in the following format only:\n"
        "PROPERTY: <the property>"
    )

    return template.format(
        descriptions="\n".join(enumerated),
        used_block=used_block,
    )

def build_common_property_prompt(
    descriptions: List[str],
    used_properties: List[str],
    category: str
) -> str:
    """
    Build prompt for LLM to extract shared properties across all objects.
    If only one description is given, extract key properties of that object.
    """
    enumerated = [f"{i + 1}. {d}" for i, d in enumerate(descriptions)]
    used_block = "None" if not used_properties else ", ".join(used_properties)

    if len(descriptions) > 1:
        template = f"""You are given a list of [{category}] descriptions.
Task: Identify properties that are shared by ALL [{category}s] in the list.
- Properties should be based on visible appearance or nearby surrounding objects.
- If a surrounding object is distinctive, explicitly name it.
- Each property should be specific but concise.
- Each property must be no more than 10 words.
- Output only properties, one per line.

List of descriptions:
{{descriptions}}

DO NOT USE the following properties: [{used_block}].
Each property must apply to every object in the list.
Extract at most 10 valid shared properties.

Format:
property1
property2
property3
"""
        return template.format(
            descriptions="\n".join(enumerated),
        )

    template = f"""You are given a single [{category}] description.
Task: Identify distinctive properties of this [{category}].
- Properties should be based on visible appearance or nearby surrounding objects.
- Each property should be specific but concise.
- Each property must be no more than 10 words.
- Output only properties, one per line.

Description:
{{descriptions}}

DO NOT USE the following properties: [{used_block}].
Extract at most 10 valid properties.

Format:
property1
property2
property3
"""
    return template.format(
        descriptions=enumerated[0],
    )


def build_text_goal_property_prompt(text_goal: str, category: str) -> str:
    """
    Build prompt for LLM to extract candidate attributes from a text_goal (description).
    """
    template = f"""You are given a single [{category}] description.
Task: Extract distinctive properties of this [{category}].
- Properties must be specific to this [{category}], not generic.
- Include only directly observable physical attributes of the [{category}] itself or objects adjacent to it.
- Exclude atmosphere, emotional tone, interpretive statements, and descriptive effects.
- Each property should be specific but concise.
- Each property must be no more than 10 words.
- Output only properties, one per line.

Description:
{text_goal}

Extract at most 10 valid properties.

Format:
property1
property2
property3
"""
    return template


def build_text_goal_candidate_yesno_prompt(
    text_goal: str,
    category: str,
    candidate_caption: str,
) -> str:
    return (
        f"You are given two captions describing a [{category}].\n"
        f"Caption A (reference): {text_goal}\n"
        f"Caption B (candidate): {candidate_caption}\n"
        "Notes:\n"
        "- Each caption can include both the object description and nearby surrounding context.\n"
        "- Focus on whether they refer to the same object; surrounding context may differ.\n"
        f"Question: Do Caption A and Caption B refer to the same [{category}]?\n"
        "Answer strictly with 'yes' or 'no'. Do not say anything else."
    )


def build_mllm_property_prompt(category: str, num_images: int, used_properties: List[str]) -> str:
    """
    Build prompt for MLLM to propose a discriminative property from images.
    Used as fallback when LLM gets stuck.
    
    Args:
        category: Object category name
        num_images: Number of candidate images
        used_properties: List of already used properties
    
    Returns:
        Formatted prompt string
    """
    k = math.ceil(num_images / 2)
    used_block = "None" if not used_properties else ", ".join(used_properties)
    image_placeholders = "image: <image>\n" * num_images

    # Images will be prepended by the caller as "image: <image>\n" * num_images
    template = (
        f"{image_placeholders}"
        f"You are given {num_images} images of [{category}].\n"
        f"Task: Identify one clear property that is shared by about half of the [{category}s] in the list "
        f"(roughly {k} out of {num_images} [{category}s]). The other [{category}s] should not have this property.\n"
        "- The property should be based on visible appearance or nearby surrounding objects, not fine details.\n"
        f"- Do not describe the images; focus on identifying one single splitting property.\n\n"
        f"- If an object is key to the split, explicitly name it.\n\n"
        "DO NOT USE: [{used_block}].\nChoose another property.\n"
        "Answer only in the following format only:\n"
        "PROPERTY: <the property>"
    )
    
    return template.format(used_block=used_block)


def build_yesno_property_prompt(category: str, property_text: str) -> str:
    """
    Build prompt for MLLM to check if a candidate has a specific property.
    
    Args:
        category: Object category name
        property_text: Property to check
    
    Returns:
        Formatted prompt string
    """
    template = (
        f"image: <image>\n"
        f"Does this [{category}] have the following property?\n"
        f"PROPERTY: [{property_text}]\n"
        f"Answer strictly with 'yes' or 'no'. Do not say anything else."
    )
    
    return template


def build_occluded_view_filter_prompt(category: str, n_views: int) -> str:
    return f"""You are given one collage image containing {n_views} views of [{category}], labeled as View 0 to View {n_views - 1}.
Task: Select only the views that should be removed because the [{category}] is not clearly visible or is heavily occluded.
- If a view is acceptable, do not list it.
- If all views are acceptable, output nothing.
- Output only indices (one per line), no extra text.

Format:
index
index
"""


def build_final_verification_prompt(
    category: str,
    used_properties: List[str],
    mode: str = "candidate_text_mode",
    candidate_description: str = "",
) -> str:
    """
    Build prompt for MLLM final verification of target vs candidate.
    
    Args:
        category: Object category name
        used_properties: List of properties already used
    
    Returns:
        Formatted prompt string
    """
    used_properties_block = "None" if not used_properties else ", ".join(f"'{p}'" for p in used_properties)
    
    if mode == "candidate_text_mode":
        if not candidate_description:
            raise ValueError("candidate_description must be provided in candidate_text_mode")
        template = (
            f"target image: <image>\n"
            f"candidate image description: {candidate_description}\n"
            f"USED PROPERTIES: {used_properties_block}\n\n"
            f"Question: Is the candidate object the same as the target object?\n\n"
            f"Strictly follow this output format:\n"
            f"DECISION: <'yes' or 'no'>\n"
            f"PROPERTY: <If DECISION is 'no', provide a new visual property that distinguishes the two. This property must not be in USED PROPERTIES. If DECISION is 'yes', write 'N/A'.>"
        )
    else:
        template = (
            f"target image: <image>\n"
            f"candidate image: <image>\n"
            f"USED PROPERTIES: {used_properties_block}\n\n"
            f"Question: Is the candidate object the same as the target object?\n\n"
            f"Strictly follow this output format:\n"
            f"DECISION: <'yes' or 'no'>\n"
            f"PROPERTY: <If DECISION is 'no', provide a new visual property that distinguishes the two. This property must not be in USED PROPERTIES. If DECISION is 'yes', write 'N/A'.>"
        )
    return template


def build_fallback_selection_prompt(
    candidates_descriptions: List[str],
    dialogue_history: List[Tuple[str, str, int]]
) -> str:
    """
    Build prompt for LLM to select the best candidate when PBP cannot narrow down to one.
    
    Args:
        candidates_descriptions: List of candidate descriptions (in order)
        dialogue_history: List of (property, user_response, depth) tuples
    
    Returns:
        Formatted prompt string
    """
    n = len(candidates_descriptions)
    enumerated = [f"{i+1}. {desc}" for i, desc in enumerate(candidates_descriptions)]
    
    dialogue_str = "\n".join([
        f"- Property: {prop} → Answer: {ans}"
        for prop, ans, depth in dialogue_history
    ])
    
    if not dialogue_str:
        dialogue_str = "No dialogue history available."
    
    template = (
        "You are given candidate objects and their descriptions.\n"
        "Task: Based on the dialogue history showing which properties the TARGET object has, "
        "select the ONE candidate most similar to the target.\n\n"
        "Candidates:\n{candidates}\n\n"
        "Dialogue History (target properties):\n{dialogue}\n\n"
        "IMPORTANT: You MUST select exactly one candidate, even if none seem perfect.\n"
        "Choose the most similar one based on the dialogue history.\n\n"
        "Answer only in the following format:\n"
        "SELECTED_INDEX: <number>"
    )
    
    return template.format(
        candidates="\n".join(enumerated),
        dialogue=dialogue_str
    )
