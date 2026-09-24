"""E2 语料：按 **目标 token 数** 生成文本 / 构造 chat 请求。

严禁把「字符数」当「token 数」用：所有 ISL 分组都先用真实 tokenizer
（renderer.get_tokenizer()）把文本编码后校准到目标 token 数。
"""

from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------- 基础文本块

EN_UNIT = (
    "The quick brown fox jumps over the lazy dog, and the pipeline moves "
    "tokens through the frontend renderer before the engine core ever sees "
    "them. "
)

ZH_UNIT = (
    "分词器把用户消息渲染成聊天模板，再编码成 token 序列送进推理引擎；"
    "在线服务的首字延迟里包含了这段纯 CPU 的前端开销。"
)

CODE_UNIT = (
    "def tokenize(text: str, *, add_special_tokens: bool = True) -> list[int]:\n"
    "    ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)\n"
    "    return [int(i) for i in ids]\n\n"
)

MIX_UNIT = EN_UNIT + ZH_UNIT + CODE_UNIT

UNITS = {
    "en": EN_UNIT,
    "zh": ZH_UNIT,
    "code": CODE_UNIT,
    "mixed": MIX_UNIT,
}


def make_text(tokenizer, target_tokens: int, corpus: str = "mixed") -> tuple[str, int]:
    """生成一段 **恰好** 约 target_tokens 个 token 的文本。

    返回 (text, actual_tokens)。做法是按 unit 的实际 token 密度线性外推，
    再逐次微调；不做字符数假设。

    **短文本（< 一个 unit 的 token 数）会走单独的分支**：按 token 粒度从
    unit 的编码结果反推需要的字符数，再二分收敛。
    早期版本直接 `unit * max(1, round(...))`，导致"目标 8 token"实际给了 110 token
    （一个 unit），把短 prompt 的成本测成了长 prompt 的成本——这个坑记在这里。
    """
    unit = UNITS[corpus]
    unit_ids = tokenizer.encode(unit, add_special_tokens=False)
    if isinstance(unit_ids, dict):  # transformers v5 BatchEncoding
        unit_ids = unit_ids["input_ids"]
    if len(unit_ids) == 0:
        raise ValueError("unit tokenizes to 0 ids")

    if target_tokens < len(unit_ids):
        # 短文本分支：对「字符长度」二分，找第一个 ≥ target 的前缀
        def count(s: str) -> int:
            ids = tokenizer.encode(s, add_special_tokens=False)
            return len(ids if isinstance(ids, list) else ids["input_ids"])

        lo, hi = 1, len(unit)
        while lo < hi:
            mid = (lo + hi) // 2
            if count(unit[:mid]) >= target_tokens:
                hi = mid
            else:
                lo = mid + 1
        text = unit[:lo]
        # 允许轻微超出（BPE 边界切不碎），把实际 token 数如实返回
        while count(text) > target_tokens and len(text) > 1:
            text = text[:-1]
            lo -= 1
        return text, count(text)

    reps = max(1, round(target_tokens / len(unit_ids)))
    text = unit * reps
    ids = tokenizer.encode(text, add_special_tokens=False)
    if isinstance(ids, dict):
        ids = ids["input_ids"]

    # 用 unit 粒度逼近目标（保证可预测、不切碎 UTF-8）
    guard = 0
    while len(ids) < target_tokens and guard < 4 * reps + 16:
        text += unit
        ids = tokenizer.encode(text, add_special_tokens=False)
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        guard += 1
    while len(ids) > target_tokens and reps > 1 and guard < 8 * reps + 32:
        text = text[: -len(unit)]
        ids = tokenizer.encode(text, add_special_tokens=False)
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        guard += 1

    return text, len(ids)


# ---------------------------------------------------------------- chat 请求

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state, e.g. 'Shanghai, CN'.",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search available flights between two cities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["origin", "destination", "date"],
            },
        },
    },
]


def make_chat_messages(
    filler: str,
    *,
    with_tools: bool = True,
    with_system: bool = True,
    n_turns: int = 1,
    tool_call_turn: bool = False,
) -> list[dict[str, Any]]:
    """构造 OpenAI 格式的 messages。

    ``tool_call_turn=True`` 时在模型输出侧插入一轮 assistant tool_calls +
    tool 结果，覆盖多轮 function calling 模板分支（真实 agent 负载的形态）。
    """
    msgs: list[dict[str, Any]] = []
    if with_system:
        msgs.append(
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant with access to tools. "
                    "Always answer in the user's language and cite the tool result."
                ),
            }
        )
    msgs.append({"role": "user", "content": filler})
    if n_turns > 1:
        msgs.append(
            {"role": "assistant", "content": "Let me check that for you."}
        )
        msgs.append({"role": "user", "content": filler})
    if tool_call_turn:
        msgs.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": json.dumps(
                                {"location": "Shanghai, CN", "unit": "celsius"}
                            ),
                        },
                    }
                ],
            }
        )
        msgs.append(
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "26 C, partly cloudy, humidity 61%.",
            }
        )
    return msgs
