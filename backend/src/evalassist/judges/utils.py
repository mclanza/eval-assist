import json
import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, create_model, field_validator

from .types import Criteria, Instance


def generate_dynamic_pydantic_model(
    model_name: str,
    field_definitions: list[tuple[str, type, Any, list[Callable[..., Any]]]],
) -> type[BaseModel]:
    validators: dict[str, Callable[..., Any]] = {
        validator.__name__: field_validator(field_definition[0], mode="after")(
            validator
        )
        for field_definition in field_definitions
        for validator in field_definition[3]
    }
    field_defs: dict[str, tuple[type, Any]] = {
        field_definition[0]: (field_definition[1], field_definition[2])
        for field_definition in field_definitions
    }
    return create_model(
        model_name,
        __config__=ConfigDict(extra="forbid"),
        __doc__=None,
        __base__=BaseModel,
        __module__=__name__,
        __validators__=validators,
        __cls_kwargs__=None,
        **field_defs,
    )


def get_context_dict(instance: Instance, criteria: Criteria) -> dict[str, str]:
    """
    Return a context dict using the instance context and the criteria declared context_fields.
    The criteria context_fields takes precedense. This is useful for multi criteria evaluations
    where different criteria require different context.
    """
    if criteria.context_fields is not None:
        # criteria implicitly expects no context
        if len(criteria.context_fields) == 0:
            return {}
        # criteria expects some context, get it from instance.context if available
        if all(field in instance.context for field in criteria.context_fields):
            return {
                context_field: instance.context[context_field]
                for context_field in criteria.context_fields
            }
    # criteria does not specify whether it expects context or not, return the instance context
    return instance.context


def is_float(element: Any) -> bool:
    # If you expect None to be passed:
    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


def build_format_instructions(model: type[BaseModel]) -> str:
    """Generate text format instructions based on JSON schema."""
    return (
        "The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\n"
        "As an example, for the schema "
        '{"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}\n'
        'the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. '
        'The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\n'
        "Here is the output schema:\n```json\n"
        f"{model.model_json_schema()}\n```\n"
    )


def sanitize_and_parse_json(raw_json: str) -> str:
    """
    Sanitize a JSON string.

    This function:
    - Escapes unescaped newlines (\n), carriage returns (\r), tabs (\t), and quotes (")
    - Fixes partial or malformed JSON (e.g., missing closing braces)
    - Handles JSON wrapped in Markdown triple-backticks

    Args:
        raw_json: The raw JSON string (possibly with unescaped characters).

    Returns:
        Parsed Python object (dict, list, etc.)

    Raises:
        json.JSONDecodeError if the string cannot be fixed.
    """

    # -------------------------------
    # 1. Extract JSON from Markdown
    # -------------------------------
    markdown_match = re.search(r"```(?:json)?(.*?)```", raw_json, re.DOTALL)
    json_str = markdown_match.group(1) if markdown_match else raw_json

    # -------------------------------
    # 2. Escape problematic characters
    # -------------------------------
    def _replace_chars(match: re.Match[str]) -> str:
        value = match.group(2)
        value = re.sub(r"\n", r"\\n", value)
        value = re.sub(r"\r", r"\\r", value)
        value = re.sub(r"\t", r"\\t", value)
        value = re.sub(r'(?<!\\)"', r"\"", value)
        return match.group(1) + value + match.group(3)

    # Apply only to JSON string values
    json_str = re.sub(
        r'(".*?"\s*:\s*")(.*?)(")', _replace_chars, json_str, flags=re.DOTALL
    )

    # -------------------------------
    # 3. Attempt to parse, fixing partial JSON
    # -------------------------------
    try:
        return json_str
    except json.JSONDecodeError:
        # Fix incomplete structures
        new_chars = []
        stack = []
        is_inside_string = False
        escaped = False

        for char in json_str:
            new_char = char
            if is_inside_string:
                if char == '"' and not escaped:
                    is_inside_string = False
                elif char == "\n" and not escaped:
                    new_char = "\\n"
                elif char == "\\":
                    escaped = not escaped
                else:
                    escaped = False
            elif char == '"':
                is_inside_string = True
                escaped = False
            elif char == "{":
                stack.append("}")
            elif char == "[":
                stack.append("]")
            elif char in {"}", "]"}:
                if stack and stack[-1] == char:
                    stack.pop()
                else:
                    return ""
            new_chars.append(new_char)

        if is_inside_string:
            if escaped:
                new_chars.pop()
            new_chars.append('"')

        stack.reverse()

        # Try progressively closing JSON until it parses
        while new_chars:
            try:
                return "".join(new_chars + stack)
            except json.JSONDecodeError:
                new_chars.pop()

        # If everything fails, raise
        return json_str
