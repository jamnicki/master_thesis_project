from typing import List


def construct_ARC_prompt(question: str, options: List[str], enum_chars: List[str]):
    instructions = f"Pick one from the given options, {', '.join(enum_chars[:-1])} or {enum_chars[-1]}?"
    prompt = (
        instructions
        + "\n\n"
        + "Question: {question}\nOptions:\n\t{options}\n\nAnswer: "
    )
    options_str = "\n\t".join(
        f"{letter}. {choice}" for letter, choice in zip(enum_chars, options)
    )
    return prompt.format(question=question, options=options_str)
