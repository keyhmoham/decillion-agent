from app.llm.toolcall import (
    coerce_arguments, parse_tool_call_body, resolve_tool_name, validate_arguments,
)

SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "count": {"type": "integer"},
        "ratio": {"type": "number"},
        "flag": {"type": "boolean"},
        "items": {"type": "array", "items": {"type": "string"}},
        "todos": {"type": "array", "items": {"type": "object", "properties": {"content": {"type": "string"}}}},
        "opt": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "obj": {"type": "object"},
    },
    "required": ["path"],
}


def test_native_format():
    c = parse_tool_call_body("\n<function=bash>\n<parameter=command>\necho hi\n</parameter>\n</function>\n")
    assert c.name == "bash" and c.arguments == {"command": "echo hi"} and not c.repaired


def test_multiline_values_keep_inner_newlines():
    c = parse_tool_call_body("<function=w><parameter=content>\nline1\n\nline3\n</parameter></function>")
    assert c.arguments["content"] == "line1\n\nline3"


def test_value_containing_angle_brackets():
    c = parse_tool_call_body("<function=w><parameter=content>\nif a < b and b > c: <tag>\n</parameter></function>")
    assert c.arguments["content"] == "if a < b and b > c: <tag>"


def test_hermes_json_fallback():
    c = parse_tool_call_body('{"name": "bash", "arguments": {"command": "ls"}}')
    assert c.name == "bash" and c.arguments == {"command": "ls"} and c.repaired


def test_json_args_inside_function_tag():
    c = parse_tool_call_body('<function=bash>{"command": "ls"}</function>')
    assert c.arguments == {"command": "ls"}


def test_missing_closing_tags_are_repaired():
    c = parse_tool_call_body("<function=bash>\n<parameter=command>\nls\n<parameter=timeout>\n5")
    assert c.arguments == {"command": "ls", "timeout": "5"} and c.repaired


def test_quoted_function_name():
    assert parse_tool_call_body('<function="bash"><parameter=a>\n1\n</parameter></function>').name == "bash"


def test_garbage_is_an_error_not_a_crash():
    c = parse_tool_call_body("lol no")
    assert c.error and not c.name
    assert parse_tool_call_body("").error


def test_coercion():
    args = {"path": "007", "count": "5", "ratio": "0.5", "flag": "True", "items": '["a", "b"]',
            "todos": '[{"content": "x"}]', "opt": "null", "obj": '{"a": 1}'}
    out = coerce_arguments(args, SCHEMA)
    assert out == {"path": "007", "count": 5, "ratio": 0.5, "flag": True, "items": ["a", "b"],
                   "todos": [{"content": "x"}], "opt": None, "obj": {"a": 1}}


def test_coercion_python_literal_fallback_and_bad_values_survive():
    out = coerce_arguments({"items": "['a', 'b']", "count": "many"}, SCHEMA)
    assert out["items"] == ["a", "b"] and out["count"] == "many"


def test_validation_messages():
    assert validate_arguments({"path": "x"}, SCHEMA) == []
    problems = validate_arguments({"count": "many"}, SCHEMA)
    assert any("path" in p for p in problems) and any("count" in p for p in problems)


def test_name_resolution():
    names = ["bash", "read_file", "write_file"]
    assert resolve_tool_name("Bash", names) == "bash"
    assert resolve_tool_name("readfile", names) == "read_file"
    assert resolve_tool_name("write_files", names) == "write_file"
    assert resolve_tool_name("rm_rf", names) is None


def test_compact_schema_keeps_property_named_title():
    from app.llm.toolcall import compact_tool_schema

    tool = {"type": "function", "function": {"name": "x", "description": "d", "parameters": {
        "title": "XArgs", "type": "object", "additionalProperties": False,
        "properties": {"title": {"title": "Title", "type": "string", "default": None},
                       "rows": {"type": "array", "items": {"title": "Row", "type": "object", "properties": {"title": {"type": "string"}}}}},
        "required": ["title"]}}}
    out = compact_tool_schema(tool)["function"]["parameters"]
    assert "title" in out["properties"] and "title" not in out["properties"]["title"]
    assert "title" in out["properties"]["rows"]["items"]["properties"]
    assert "additionalProperties" not in out and "title" not in out
