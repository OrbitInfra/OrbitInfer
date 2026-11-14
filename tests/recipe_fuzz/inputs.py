# Adapted from Resin a1989594 for OrbitInfer; changes: docs/source-import.md.
"""Translate the reference libraries' JSON fixtures into OrbitInfer's typed inputs."""

import orbitinfer


def native_inputs(request, *, tools_before_text=False):
    """Keep oracle dictionaries outside the native recipe API under test."""
    messages = []
    for message in request["messages"]:
        content = []
        if (thinking := message.get("reasoning_content")) is not None:
            content.append(orbitinfer.recipe.Thinking(thinking))
        value = message.get("content")
        if message["role"] == "tool":
            content.append(
                orbitinfer.recipe.ToolResult(
                    value or "",
                    call_id=message.get("tool_call_id"),
                    name=message.get("name"),
                )
            )
        elif isinstance(value, str):
            content.append(orbitinfer.recipe.Text(value))
        elif value:
            for block in value:
                kind = block["type"]
                if kind == "text":
                    content.append(orbitinfer.recipe.Text(block["text"]))
                elif kind == "image":
                    content.append(orbitinfer.recipe.Image(source=block.get("image")))
                elif kind == "video":
                    content.append(orbitinfer.recipe.Video(source=block.get("video")))
                else:
                    raise AssertionError(f"unhandled oracle content: {kind}")
        for call in message.get("tool_calls", ()):
            function = call.get("function", call)
            content.append(
                orbitinfer.recipe.ToolCall(
                    function["name"], function["arguments"], id=call.get("id")
                )
            )
        if tools_before_text:
            content.sort(
                key=lambda block: 1 if isinstance(block, orbitinfer.recipe.Text) else 0
            )
        messages.append(
            orbitinfer.recipe.Message(
                getattr(orbitinfer.recipe.Role, message["role"].upper()),
                content,
                tools=[native_tool(tool) for tool in message.get("tools", ())],
            )
        )
    controls = dict(request.get("extra") or {})
    controls.update(
        {
            key: value
            for key, value in request.items()
            if key not in {"messages", "tools", "extra"}
        }
    )
    return dict(
        messages=messages,
        tools=[native_tool(tool) for tool in request.get("tools") or ()],
        options=orbitinfer.recipe.RenderOptions(**controls),
    )


def native_tool(tool):
    tool = tool.get("function", tool)
    return orbitinfer.recipe.Tool(
        tool["name"], **{key: value for key, value in tool.items() if key != "name"}
    )
