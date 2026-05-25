#!/usr/bin/env python3
"""Validate benchmark case YAML files and result JSON/JSONL files.

This tool intentionally uses only the Python standard library for Milestone 0.
It supports the JSON Schema keywords used by this repository's schemas, plus a
small YAML subset for the hand-authored case files.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_SCHEMA = REPO_ROOT / "schemas" / "case.schema.json"
RESULT_SCHEMA = REPO_ROOT / "schemas" / "result.schema.json"


class ValidationError(Exception):
    """Raised when a file does not satisfy its schema or semantic checks."""


def _strip_comment(line: str) -> str:
    in_quote: str | None = None
    for index, char in enumerate(line):
        if char in ("'", '"'):
            if in_quote == char:
                in_quote = None
            elif in_quote is None:
                in_quote = char
        elif char == "#" and in_quote is None:
            return line[:index]
    return line


def _yaml_lines(text: str) -> List[Tuple[int, str]]:
    lines: List[Tuple[int, str]] = []
    for raw in text.splitlines():
        line = _strip_comment(raw).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2:
            raise ValidationError("YAML indentation must use multiples of two spaces")
        lines.append((indent, line.strip()))
    return lines


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    if value == "null":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        body = value[1:-1].strip()
        if not body:
            return []
        return [_parse_scalar(item.strip()) for item in body.split(",")]
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?(\d+\.\d*|\d*\.\d+|\d+)([eE][-+]?\d+)?", value):
        return float(value)
    return value


def _parse_yaml_block(lines: Sequence[Tuple[int, str]], index: int, indent: int) -> Tuple[Any, int]:
    if index >= len(lines):
        return {}, index

    current_indent, current_text = lines[index]
    if current_indent < indent:
        return {}, index
    if current_indent > indent:
        raise ValidationError(f"unexpected YAML indentation before: {current_text}")

    if current_text.startswith("- "):
        result: List[Any] = []
        while index < len(lines):
            current_indent, current_text = lines[index]
            if current_indent != indent or not current_text.startswith("- "):
                break
            item = current_text[2:].strip()
            index += 1
            if item:
                result.append(_parse_scalar(item))
            else:
                nested, index = _parse_yaml_block(lines, index, indent + 2)
                result.append(nested)
        return result, index

    result: Dict[str, Any] = {}
    while index < len(lines):
        current_indent, current_text = lines[index]
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ValidationError(f"unexpected YAML indentation before: {current_text}")
        if current_text.startswith("- "):
            break
        if ":" not in current_text:
            raise ValidationError(f"expected YAML mapping entry: {current_text}")
        key, raw_value = current_text.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        index += 1
        if raw_value:
            result[key] = _parse_scalar(raw_value)
        else:
            nested, index = _parse_yaml_block(lines, index, indent + 2)
            result[key] = nested
    return result, index


def load_yaml(path: Path) -> Any:
    lines = _yaml_lines(path.read_text(encoding="utf-8"))
    data, index = _parse_yaml_block(lines, 0, 0)
    if index != len(lines):
        raise ValidationError(f"could not parse all YAML content in {path}")
    return data


def load_document(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        return load_yaml(path)
    raise ValidationError(f"unsupported file extension: {path}")


def load_jsonl(path: Path) -> Iterable[Any]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if stripped:
                try:
                    yield json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValidationError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def _schema_ref(root_schema: Dict[str, Any], ref: str) -> Dict[str, Any]:
    if not ref.startswith("#/"):
        raise ValidationError(f"unsupported schema reference: {ref}")
    current: Any = root_schema
    for part in ref[2:].split("/"):
        current = current[part]
    return current


def _is_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    raise ValidationError(f"unsupported schema type: {expected}")


def _validate_json_schema(value: Any, schema: Dict[str, Any], root_schema: Dict[str, Any], path: str) -> None:
    if "$ref" in schema:
        _validate_json_schema(value, _schema_ref(root_schema, schema["$ref"]), root_schema, path)
        return

    if "oneOf" in schema:
        matches = 0
        errors: List[str] = []
        for option in schema["oneOf"]:
            try:
                _validate_json_schema(value, option, root_schema, path)
                matches += 1
            except ValidationError as exc:
                errors.append(str(exc))
        if matches != 1:
            detail = "; ".join(errors[:2])
            raise ValidationError(f"{path}: expected exactly one oneOf match, got {matches}. {detail}")
        return

    expected_type = schema.get("type")
    if expected_type is not None:
        expected_types = expected_type if isinstance(expected_type, list) else [expected_type]
        if not any(_is_type(value, item) for item in expected_types):
            raise ValidationError(f"{path}: expected type {expected_type}, got {type(value).__name__}")

    if "enum" in schema and value not in schema["enum"]:
        raise ValidationError(f"{path}: expected one of {schema['enum']!r}, got {value!r}")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise ValidationError(f"{path}: string is shorter than minLength")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            raise ValidationError(f"{path}: value {value!r} does not match pattern {schema['pattern']!r}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValidationError(f"{path}: value {value!r} is less than minimum {schema['minimum']!r}")

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise ValidationError(f"{path}: array has fewer than minItems")
        if schema.get("uniqueItems") and len({json.dumps(item, sort_keys=True) for item in value}) != len(value):
            raise ValidationError(f"{path}: array items must be unique")
        if "items" in schema:
            for index, item in enumerate(value):
                _validate_json_schema(item, schema["items"], root_schema, f"{path}[{index}]")

    if isinstance(value, dict):
        if "minProperties" in schema and len(value) < schema["minProperties"]:
            raise ValidationError(f"{path}: object has fewer than minProperties")
        for required in schema.get("required", []):
            if required not in value:
                raise ValidationError(f"{path}: missing required property {required!r}")

        property_schemas = schema.get("properties", {})
        allowed_properties = set(property_schemas)
        additional = schema.get("additionalProperties", True)

        if "propertyNames" in schema:
            for key in value:
                _validate_json_schema(key, schema["propertyNames"], root_schema, f"{path}.{key}<name>")

        for key, item in value.items():
            child_path = f"{path}.{key}"
            if key in property_schemas:
                _validate_json_schema(item, property_schemas[key], root_schema, child_path)
            elif additional is False:
                raise ValidationError(f"{child_path}: additional property is not allowed")
            elif isinstance(additional, dict):
                _validate_json_schema(item, additional, root_schema, child_path)
            elif additional is not True:
                raise ValidationError(f"{path}: unsupported additionalProperties value")


def _parse_einsum(einsum: str) -> Tuple[List[List[str]], List[str]]:
    lhs, rhs = einsum.split("->", 1)
    return [list(term) for term in lhs.split(",")], list(rhs)


def _same_items(left: Sequence[str], right: Sequence[str]) -> bool:
    return sorted(left) == sorted(right)


def validate_case_semantics(case: Dict[str, Any], path: Path) -> None:
    input_terms, output_term = _parse_einsum(case["einsum"])
    input_names = list(case["inputs"])
    if len(input_terms) != len(input_names):
        raise ValidationError(f"{path}: einsum input count does not match inputs")

    parameters = set(case["parameters"])
    all_input_labels: List[str] = []
    for input_name, term in zip(input_names, input_terms):
        tensor = case["inputs"][input_name]
        labels = tensor["labels"]
        dims = tensor["dims"]
        layout = tensor["layout"]
        all_input_labels.extend(labels)
        if labels != term:
            raise ValidationError(f"{path}: inputs.{input_name}.labels does not match einsum term")
        if set(dims) != set(labels):
            raise ValidationError(f"{path}: inputs.{input_name}.dims keys must match labels")
        if not _same_items(layout["physical_order"], labels):
            raise ValidationError(f"{path}: inputs.{input_name}.layout.physical_order must permute labels")
        if layout["kind"] == "strided" and len(layout.get("strides", [])) != len(labels):
            raise ValidationError(f"{path}: strided input layout must include one stride per label")
        if layout["kind"] != "strided" and "strides" in layout:
            raise ValidationError(f"{path}: only strided layouts may specify strides")
        for label, dimension in dims.items():
            if isinstance(dimension, str) and dimension not in parameters:
                raise ValidationError(f"{path}: dimension {label!r} references unknown parameter {dimension!r}")

    output = case["output"]
    if output["labels"] != output_term:
        raise ValidationError(f"{path}: output.labels does not match einsum output")
    if not _same_items(output["layout"]["physical_order"], output["labels"]):
        raise ValidationError(f"{path}: output.layout.physical_order must permute labels")
    if output["layout"]["kind"] == "strided" and len(output["layout"].get("strides", [])) != len(output["labels"]):
        raise ValidationError(f"{path}: strided output layout must include one stride per label")
    if output["layout"]["kind"] != "strided" and "strides" in output["layout"]:
        raise ValidationError(f"{path}: only strided layouts may specify strides")

    output_labels = set(output["labels"])
    input_label_set = set(all_input_labels)
    if not output_labels <= input_label_set:
        raise ValidationError(f"{path}: output labels must appear in inputs")

    contraction_variants = case["variants"]["contraction_permutations"]
    if isinstance(contraction_variants, list):
        for variant in contraction_variants:
            variant_inputs, variant_output = _parse_einsum(variant)
            if len(variant_inputs) != len(input_names):
                raise ValidationError(f"{path}: contraction variant input count does not match inputs")
            if not _same_items(variant_output, output["labels"]):
                raise ValidationError(f"{path}: contraction variant output must permute output labels")
            for input_name, variant_term in zip(input_names, variant_inputs):
                if not _same_items(variant_term, case["inputs"][input_name]["labels"]):
                    raise ValidationError(
                        f"{path}: contraction variant term for {input_name} must permute input labels"
                    )


def validate_result_semantics(result: Dict[str, Any], path: Path) -> None:
    timing = result["timing"]
    times = timing["times_sec"]
    if timing["repeat"] != len(times):
        raise ValidationError(f"{path}: timing.repeat must equal len(timing.times_sec)")
    if abs(timing["min_sec"] - min(times)) > 1.0e-15:
        raise ValidationError(f"{path}: timing.min_sec must equal min(timing.times_sec)")

    for name, layout in result["input_layouts"].items():
        if not _same_items(layout["labels"], layout["physical_order"]):
            raise ValidationError(f"{path}: input_layouts.{name}.physical_order must permute labels")
        if len(layout["strides"]) != len(layout["labels"]):
            raise ValidationError(f"{path}: input_layouts.{name}.strides must match labels length")
    output_layout = result["output_layout"]
    if not _same_items(output_layout["labels"], output_layout["physical_order"]):
        raise ValidationError(f"{path}: output_layout.physical_order must permute labels")
    if len(output_layout["strides"]) != len(output_layout["labels"]):
        raise ValidationError(f"{path}: output_layout.strides must match labels length")


def validate_case(path: Path) -> None:
    schema = json.loads(CASE_SCHEMA.read_text(encoding="utf-8"))
    case = load_document(path)
    _validate_json_schema(case, schema, schema, "$")
    validate_case_semantics(case, path)


def validate_result(path: Path) -> None:
    schema = json.loads(RESULT_SCHEMA.read_text(encoding="utf-8"))
    if path.suffix.lower() == ".jsonl":
        for result in load_jsonl(path):
            _validate_json_schema(result, schema, schema, "$")
            validate_result_semantics(result, path)
    else:
        result = load_document(path)
        _validate_json_schema(result, schema, schema, "$")
        validate_result_semantics(result, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind",
        choices=("case", "result"),
        required=True,
        help="schema kind to validate against",
    )
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)

    validator = validate_case if args.kind == "case" else validate_result
    ok = True
    for path in args.paths:
        try:
            validator(path)
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            print(f"{path}: {exc}", file=sys.stderr)
            ok = False
        else:
            print(f"{path}: ok")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
