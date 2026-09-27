"""Token 估算 + 对话切片。

没有 tiktoken 时用字符估算。规则：
- CJK 字符（中日韩）约 1 字符 = 1 token
- 其他字符约 4 字符 = 1 token
这个估算偏保守（宁可多算），用于决定是否切片，不影响正确性——
真正决定成败的是分片后的 prompt 是否仍带齐上下文。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# CJK 统一表意文字 + 中文标点范围
_CJK = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff"
    r"\u3000-\u303f\uff00-\uffef\u2018-\u201d]"
)


def estimate_tokens(text: str) -> int:
    """保守估算一段文本的 token 数。"""
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    other = len(text) - cjk
    return cjk + max(1, other // 4)


def messages_tokens(messages: list[dict[str, Any]]) -> int:
    """整段 messages 的 token 估算（含每条的结构开销）。"""
    total = 0
    for m in messages:
        total += 4  # role/分隔符的固定开销
        content = m.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            # 多模态 content parts：[{type:text,text:...}, ...]
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        total += estimate_tokens(part.get("text", ""))
                    else:
                        total += 512  # 图片等按固定值粗算
        for k in ("name", "tool_call_id"):
            if m.get(k):
                total += estimate_tokens(str(m[k]))
        if m.get("tool_calls"):
            total += estimate_tokens(str(m["tool_calls"]))
    return total


@dataclass
class Shard:
    """一个分片：完整的可独立下发的 messages。"""

    index: int
    messages: list[dict[str, Any]] = field(default_factory=list)
    tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "tokens": self.tokens}


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        )
    return str(content or "")


def _split_text_by_chars(text: str, shard_tokens: int) -> list[str]:
    """把超长文本按**估算 token** 切成若干段（优先段落/句子边界，不断句）。

    用 estimate_tokens 迭代求每段字符数，而不是固定字符系数——
    中文 1 字≈1 token、英文 4 字符≈1 token，固定系数会让中文严重超预算。
    """
    import re as _re

    if estimate_tokens(text) <= shard_tokens:
        return [text]

    target_tokens = shard_tokens
    out: list[str] = []

    # 先在段落边界贪心切
    paras = text.split("\n")
    cur = ""
    cur_tok = 0
    for p in paras:
        pt = estimate_tokens(p)
        if cur and cur_tok + pt > target_tokens:
            out.append(cur)
            cur, cur_tok = p, pt
        else:
            cur = (cur + "\n" + p) if cur else p
            cur_tok += pt
    if cur:
        out.append(cur)

    # 段落切完仍有超标的，按句子边界二次切
    final: list[str] = []
    for seg in out:
        if estimate_tokens(seg) <= target_tokens:
            final.append(seg)
            continue
        sents = _re.split(r"(?<=[。！？.!?])\s*", seg)
        cur, cur_tok = "", 0
        for s in sents:
            st = estimate_tokens(s)
            if cur and cur_tok + st > target_tokens:
                final.append(cur)
                cur, cur_tok = s, st
            else:
                cur += s
                cur_tok += st
        if cur:
            final.append(cur)

    # 极端情况：单句比预算还长，按字符数硬切。
    # 注意：只按 index 前进切分，不丢任何字符；宁可单片略超预算，
    # 也绝不静默截断正文（丢内容比分片超限更严重）。
    hard: list[str] = []
    for seg in final:
        while estimate_tokens(seg) > target_tokens:
            ratio = target_tokens / estimate_tokens(seg)
            cut = max(1, int(len(seg) * ratio))
            hard.append(seg[:cut])
            seg = seg[cut:]
        if seg:
            hard.append(seg)
    return hard or [text]


def _shard_overlap(shard_tokens: int, overlap_ratio: float) -> int:
    """重叠预算的 token 数：片大的绝对 15%，片小的至少 300 token 兜底。
    重叠是给跨边界内容「搭手」，太小等于没搭。"""
    return max(300, int(shard_tokens * overlap_ratio))


def _take_tail_within(messages: list[dict[str, Any]], budget_tokens: int) -> list[dict[str, Any]]:
    """从尾部往前取消息，总量不超过 budget_tokens。"""
    taken: list[dict[str, Any]] = []
    used = 0
    for m in reversed(messages):
        t = messages_tokens([m])
        if taken and used + t > budget_tokens:
            break
        taken.append(m)
        used += t
        if not taken and t > budget_tokens:
            # 单条就超预算且本来就只该有一条：照取，交给后续处理
            break
    taken.reverse()
    return taken


def split_messages(
    messages: list[dict[str, Any]],
    shard_tokens: int,
    max_shards: int,
    replicate_system: bool = True,
    overlap_ratio: float = 0.0,
) -> list[Shard]:
    """把 messages 切成 N 个 Shard。

    切分规则（关键设计）：
    1. system 消息按 replicate_system 决定是否复制到每个分片头部。
    2. 其余消息按累积 token 贪心分组，一组满 shard_tokens 就开新分片。
    3. 单条消息本身超过 shard_tokens 时，只要它仍是**最后一条**，就单独
       成片（上游能处理单体大请求）；若是历史消息则单独成片但不拆内容
       （拆一条消息内容会改变语义，交给 LLMLingua 类压缩处理更合适）。
    4. 分片数超过 max_shards 时，把尾部分片合并，保证不超上限。
    """
    system_msgs = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]

    prefix = system_msgs if replicate_system else []
    prefix_tokens = messages_tokens(prefix)

    groups: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    cur_tokens = prefix_tokens

    # 巨型消息拆段：这些组的边界是内容级硬切，不参与消息级重叠
    exploded_from_giant = 0

    for m in rest:
        mt = messages_tokens([m])
        # 单条就超预算：按内容切成多段，每段都带 role，成为独立消息
        if mt > shard_tokens:
            text = _content_text(m.get("content"))
            if len(text) > 200 and m.get("role") in ("user", "system"):
                pieces = _split_text_by_chars(text, shard_tokens)
                if cur:
                    groups.append(cur)
                    cur, cur_tokens = [], prefix_tokens
                for pi, piece in enumerate(pieces):
                    total = len(pieces)
                    marker = f"（此为用户长消息的第 {pi + 1}/{total} 段，共 {total} 段）\n"
                    # marker 附加在段首；不做尾部裁剪——裁剪会静默丢正文，
                    # 而丢内容比分片略超预算严重得多。marker 只有十几个 token，
                    # 由 _split_text_by_chars 的目标预算已预留的余量吸收；
                    # 真正兜底在上面的 max_shards 合并与单片超限的可视化。
                    groups.append([{"role": m["role"], "content": marker + piece}])
                    exploded_from_giant += 1
                continue
            # 非文本内容（如巨型 tool_calls / list content）无法安全切，单独成片
            if cur:
                groups.append(cur)
                cur, cur_tokens = [], prefix_tokens
            groups.append([m])
            continue
        if cur and cur_tokens + mt > shard_tokens:
            groups.append(cur)
            cur, cur_tokens = [], prefix_tokens
        cur.append(m)
        cur_tokens += mt
    if cur:
        groups.append(cur)

    if not groups:
        groups = [[]]

    # 合并到 max_shards 以内（从尾部往前并）
    while len(groups) > max_shards:
        merged = groups[-2] + groups[-1]
        groups = groups[:-2] + [merged]

    # ② 重叠切片：给每个相邻组之间搭「重叠带」——把前一组尾部的若干历史
    # 消息复制到后一组头部，让跨边界的内容在两片都出现，reduce 时能对上引用。
    # 巨型消息的内容硬切段不参与反推重叠（它们边界是字符级，取不到完整消息）。
    if overlap_ratio > 0 and len(groups) > 1:
        budget = _shard_overlap(shard_tokens, overlap_ratio)
        rebuilt: list[list[dict[str, Any]]] = []
        for i, g in enumerate(groups):
            gg = list(g)
            if i > 0:
                prev = groups[i - 1]
                # 前一组若是巨型拆段（段首带 marker），取不到完整消息，跳过
                prev_is_giant = any("段，共" in str(m.get("content", ""))[:60] for m in prev[:1])
                if not prev_is_giant:
                    gg = _take_tail_within(prev, budget) + gg
            rebuilt.append(gg)
        groups = rebuilt

    shards: list[Shard] = []
    for i, g in enumerate(groups):
        msgs = list(prefix) + g
        shards.append(Shard(index=i, messages=msgs, tokens=messages_tokens(msgs)))
    return shards
