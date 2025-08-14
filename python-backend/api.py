from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from uuid import uuid4
import time
import logging
from fastapi.responses import StreamingResponse
import asyncio
import os
import threading
import queue
import json
from openai import OpenAI

from main import (
    triage_agent,
    faq_agent,
    plan_change_agent,
    billing_agent,
    tech_support_agent,
    data_usage_agent,
    roaming_agent,
    human_support_agent,
    AGENT_MODEL,
    create_initial_context,
)

from agents import (
    Runner,
    ItemHelpers,
    MessageOutputItem,
    HandoffOutputItem,
    ToolCallItem,
    ToolCallOutputItem,
    InputGuardrailTripwireTriggered,
    Handoff,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS configuration (adjust as needed for deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Models
# =========================

class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str

class MessageResponse(BaseModel):
    content: str
    agent: str

class AgentEvent(BaseModel):
    id: str
    type: str
    agent: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None

class GuardrailCheck(BaseModel):
    id: str
    name: str
    input: str
    reasoning: str
    passed: bool
    timestamp: float

class ChatResponse(BaseModel):
    conversation_id: str
    current_agent: str
    messages: List[MessageResponse]
    events: List[AgentEvent]
    context: Dict[str, Any]
    agents: List[Dict[str, Any]]
    guardrails: List[GuardrailCheck] = []


class HumanReplyRequest(BaseModel):
    conversation_id: str
    message: str


class HumanBackRequest(BaseModel):
    conversation_id: str

# =========================
# In-memory store for conversation state
# =========================

class ConversationStore:
    def get(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        pass

    def save(self, conversation_id: str, state: Dict[str, Any]):
        pass

class InMemoryConversationStore(ConversationStore):
    _conversations: Dict[str, Dict[str, Any]] = {}

    def get(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        return self._conversations.get(conversation_id)

    def save(self, conversation_id: str, state: Dict[str, Any]):
        self._conversations[conversation_id] = state

# TODO: when deploying this app in scale, switch to your own production-ready implementation
conversation_store = InMemoryConversationStore()

# =========================
# Helpers
# =========================

def _get_agent_by_name(name: str):
    """Return the agent object by name."""
    agents = {
        triage_agent.name: triage_agent,
        faq_agent.name: faq_agent,
        plan_change_agent.name: plan_change_agent,
        billing_agent.name: billing_agent,
        tech_support_agent.name: tech_support_agent,
        data_usage_agent.name: data_usage_agent,
        roaming_agent.name: roaming_agent,
        human_support_agent.name: human_support_agent,
    }
    return agents.get(name, triage_agent)

def _get_guardrail_name(g) -> str:
    """Extract a friendly guardrail name."""
    name_attr = getattr(g, "name", None)
    if isinstance(name_attr, str) and name_attr:
        return name_attr
    guard_fn = getattr(g, "guardrail_function", None)
    if guard_fn is not None and hasattr(guard_fn, "__name__"):
        return guard_fn.__name__.replace("_", " ").title()
    fn_name = getattr(g, "__name__", None)
    if isinstance(fn_name, str) and fn_name:
        return fn_name.replace("_", " ").title()
    return str(g)

def _build_agents_list() -> List[Dict[str, Any]]:
    """Build a list of all available agents and their metadata."""
    def make_agent_dict(agent):
        return {
            "name": agent.name,
            "description": getattr(agent, "handoff_description", ""),
            "handoffs": [getattr(h, "agent_name", getattr(h, "name", "")) for h in getattr(agent, "handoffs", [])],
            "tools": [getattr(t, "name", getattr(t, "__name__", "")) for t in getattr(agent, "tools", [])],
            "input_guardrails": [_get_guardrail_name(g) for g in getattr(agent, "input_guardrails", [])],
        }
    return [
        make_agent_dict(triage_agent),
        make_agent_dict(faq_agent),
        make_agent_dict(plan_change_agent),
        make_agent_dict(billing_agent),
        make_agent_dict(tech_support_agent),
        make_agent_dict(data_usage_agent),
        make_agent_dict(roaming_agent),
        make_agent_dict(human_support_agent),
    ]

# =========================
# Main Chat Endpoint
# =========================

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Main chat endpoint for agent orchestration.
    Handles conversation state, agent routing, and guardrail checks.
    """
    # Log inbound request
    try:
        logger.info("/chat request: %s", {"conversation_id": req.conversation_id, "message": req.message})
    except Exception:
        pass
    # Initialize or retrieve conversation state
    is_new = not req.conversation_id or conversation_store.get(req.conversation_id) is None
    if is_new:
        # If the client supplied an id but we don't have state (e.g., server restart), reuse it
        conversation_id: str = req.conversation_id or uuid4().hex
        ctx = create_initial_context()
        current_agent_name = triage_agent.name
        state: Dict[str, Any] = {
            "input_items": [],
            "context": ctx,
            "current_agent": current_agent_name,
        }
        if req.message.strip() == "":
            conversation_store.save(conversation_id, state)
            return ChatResponse(
                conversation_id=conversation_id,
                current_agent=current_agent_name,
                messages=[],
                events=[],
                context=ctx.model_dump(),
                agents=_build_agents_list(),
                guardrails=[],
            )
    else:
        conversation_id = req.conversation_id  # type: ignore
        state = conversation_store.get(conversation_id)

    current_agent = _get_agent_by_name(state["current_agent"])
    state["input_items"].append({"content": req.message, "role": "user"})
    old_context = state["context"].model_dump().copy()
    guardrail_checks: List[GuardrailCheck] = []

    # If the current agent is Human Support, do not auto-run the model.
    if current_agent.name == human_support_agent.name:
        conversation_store.save(conversation_id, state)
        return ChatResponse(
            conversation_id=conversation_id,
            current_agent=current_agent.name,
            messages=[],
            events=[],
            context=state["context"].model_dump(),
            agents=_build_agents_list(),
            guardrails=[],
        )

    try:
        result = await Runner.run(current_agent, state["input_items"], context=state["context"])
    except InputGuardrailTripwireTriggered as e:
        failed = e.guardrail_result.guardrail
        gr_output = e.guardrail_result.output.output_info
        gr_reasoning = getattr(gr_output, "reasoning", "")
        gr_input = req.message
        gr_timestamp = time.time() * 1000
        for g in current_agent.input_guardrails:
            guardrail_checks.append(GuardrailCheck(
                id=uuid4().hex,
                name=_get_guardrail_name(g),
                input=gr_input,
                reasoning=(gr_reasoning if g == failed else ""),
                passed=(g != failed),
                timestamp=gr_timestamp,
            ))
        refusal = "Sorry, I can only answer questions related to telco support (plans, billing, usage, roaming, outages)."
        state["input_items"].append({"role": "assistant", "content": refusal})
        resp = ChatResponse(
            conversation_id=conversation_id,
            current_agent=current_agent.name,
            messages=[MessageResponse(content=refusal, agent=current_agent.name)],
            events=[],
            context=state["context"].model_dump(),
            agents=_build_agents_list(),
            guardrails=guardrail_checks,
        )
        try:
            logger.info("/chat response (guardrail refusal): %s", resp.model_dump())
        except Exception:
            pass
        return resp

    messages: List[MessageResponse] = []
    events: List[AgentEvent] = []
    # Capture preamble (most recent assistant message) per agent to attach to next tool_call for GPT-5 models
    attach_preambles = isinstance(AGENT_MODEL, str) and AGENT_MODEL.startswith("gpt-5")
    last_message_by_agent: Dict[str, str] = {}

    handoff_target_agent = None
    for item in result.new_items:
        if isinstance(item, MessageOutputItem):
            text = ItemHelpers.text_message_output(item)
            messages.append(MessageResponse(content=text, agent=item.agent.name))
            events.append(AgentEvent(id=uuid4().hex, type="message", agent=item.agent.name, content=text))
            if attach_preambles:
                last_message_by_agent[item.agent.name] = text
        # Handle handoff output and agent switching
        elif isinstance(item, HandoffOutputItem):
            from_agent = item.source_agent
            proposed_target = item.target_agent
            final_target = proposed_target
            # Find the Handoff object on the source agent matching the target
            ho = next(
                (h for h in getattr(from_agent, "handoffs", [])
                 if isinstance(h, Handoff) and getattr(h, "agent_name", None) == proposed_target.name),
                None,
            )
            cb_name = None
            if ho:
                fn = ho.on_invoke_handoff
                fv = fn.__code__.co_freevars
                cl = fn.__closure__ or []
                if "on_handoff" in fv:
                    idx = fv.index("on_handoff")
                    if idx < len(cl) and cl[idx].cell_contents:
                        cb = cl[idx].cell_contents
                        cb_name = getattr(cb, "__name__", repr(cb))
                        # Previously routed all on_*_handoff callbacks to Human Support.
                        # Adjusted to only route Technical Support handoffs to Human, so Billing/Plan go to their agents first.
                        if cb_name in {"on_support_handoff"}:
                            final_target = human_support_agent
            # If routing to Human Support, remember where to resume later
            try:
                # Proposed target is the specialist the model wanted to hand off to
                proposed_name = getattr(proposed_target, "name", None)
                final_name = getattr(final_target, "name", None)
                if final_name == human_support_agent.name:
                    if proposed_name and proposed_name != human_support_agent.name:
                        # Resume with the proposed specialist after human finishes
                        state["resume_agent_name"] = proposed_name
                    else:
                        # If the model directly handed to Human from triage, resume with triage
                        state["resume_agent_name"] = from_agent.name
            except Exception:
                pass

            # Record the handoff event with final target
            events.append(
                AgentEvent(
                    id=uuid4().hex,
                    type="handoff",
                    agent=from_agent.name,
                    content=f"{from_agent.name} -> {final_target.name}",
                    metadata={"source_agent": from_agent.name, "target_agent": final_target.name},
                )
            )
            # Record the callback as a tool call (attribute to final target for clarity)
            if cb_name:
                events.append(
                    AgentEvent(
                        id=uuid4().hex,
                        type="tool_call",
                        agent=final_target.name,
                        content=cb_name,
                    )
                )
            current_agent = final_target
            handoff_target_agent = final_target
        elif isinstance(item, ToolCallItem):
            tool_name = getattr(item.raw_item, "name", None)
            raw_args = getattr(item.raw_item, "arguments", None)
            tool_args: Any = raw_args
            if isinstance(raw_args, str):
                try:
                    import json
                    tool_args = json.loads(raw_args)
                except Exception:
                    pass
            meta: Dict[str, Any] = {"tool_args": tool_args}
            if attach_preambles:
                pre = last_message_by_agent.get(item.agent.name)
                if pre:
                    meta["preamble"] = pre
                    # Clear after attaching once
                    last_message_by_agent[item.agent.name] = ""
            events.append(
                AgentEvent(
                    id=uuid4().hex,
                    type="tool_call",
                    agent=item.agent.name,
                    content=tool_name or "",
                    metadata=meta,
                )
            )
            # No special UI triggers for telco tools
        elif isinstance(item, ToolCallOutputItem):
            events.append(
                AgentEvent(
                    id=uuid4().hex,
                    type="tool_output",
                    agent=item.agent.name,
                    content=str(item.output),
                    metadata={"tool_result": item.output},
                )
            )

    # If we just handed off and the target agent didn't produce a message yet, do one follow-up run
    if handoff_target_agent is not None and handoff_target_agent.name != human_support_agent.name:
        produced_message_from_target = any(m.agent == handoff_target_agent.name for m in messages)
        if not produced_message_from_target:
            follow_input = result.to_input_list()
            follow_result = await Runner.run(handoff_target_agent, follow_input, context=state["context"])
            for item in follow_result.new_items:
                if isinstance(item, MessageOutputItem):
                    text = ItemHelpers.text_message_output(item)
                    messages.append(MessageResponse(content=text, agent=item.agent.name))
                    events.append(AgentEvent(id=uuid4().hex, type="message", agent=item.agent.name, content=text))
                    if attach_preambles:
                        last_message_by_agent[item.agent.name] = text
                elif isinstance(item, ToolCallItem):
                    tool_name = getattr(item.raw_item, "name", None)
                    raw_args = getattr(item.raw_item, "arguments", None)
                    tool_args: Any = raw_args
                    if isinstance(raw_args, str):
                        try:
                            import json
                            tool_args = json.loads(raw_args)
                        except Exception:
                            pass
                    meta: Dict[str, Any] = {"tool_args": tool_args}
                    if attach_preambles:
                        pre = last_message_by_agent.get(item.agent.name)
                        if pre:
                            meta["preamble"] = pre
                            last_message_by_agent[item.agent.name] = ""
                    events.append(
                        AgentEvent(
                            id=uuid4().hex,
                            type="tool_call",
                            agent=item.agent.name,
                            content=tool_name or "",
                            metadata=meta,
                        )
                    )
                elif isinstance(item, ToolCallOutputItem):
                    events.append(
                        AgentEvent(
                            id=uuid4().hex,
                            type="tool_output",
                            agent=item.agent.name,
                            content=str(item.output),
                            metadata={"tool_result": item.output},
                        )
                    )
            # Update the base result to reflect follow-up for state persistence below
            result = follow_result

    new_context = state["context"].dict()
    changes = {k: new_context[k] for k in new_context if old_context.get(k) != new_context[k]}
    if changes:
        events.append(
            AgentEvent(
                id=uuid4().hex,
                type="context_update",
                agent=current_agent.name,
                content="",
                metadata={"changes": changes},
            )
        )

    state["input_items"] = result.to_input_list()
    state["current_agent"] = current_agent.name
    conversation_store.save(conversation_id, state)

    # Build guardrail results: mark failures (if any), and any others as passed
    final_guardrails: List[GuardrailCheck] = []
    for g in getattr(current_agent, "input_guardrails", []):
        name = _get_guardrail_name(g)
        failed = next((gc for gc in guardrail_checks if gc.name == name), None)
        if failed:
            final_guardrails.append(failed)
        else:
            final_guardrails.append(GuardrailCheck(
                id=uuid4().hex,
                name=name,
                input=req.message,
                reasoning="",
                passed=True,
                timestamp=time.time() * 1000,
            ))

    resp = ChatResponse(
        conversation_id=conversation_id,
        current_agent=current_agent.name,
        messages=messages,
        events=events,
        context=state["context"].dict(),
        agents=_build_agents_list(),
        guardrails=final_guardrails,
    )
    try:
        logger.info("/chat response: %s", resp.model_dump())
    except Exception:
        pass
    return resp


async def _yield_keepalive(interval_seconds: float = 10.0):
    last = time.time()
    while True:
        now = time.time()
        if now - last >= interval_seconds:
            yield b":\n\n"  # comment/keepalive
            last = now
        await asyncio.sleep(1.0)


def _format_sse(event: str, data: Dict[str, Any]) -> bytes:
    import json
    return (f"event: {event}\n" + f"data: {json.dumps(data, ensure_ascii=False)}\n\n").encode("utf-8")


def _base_response_envelope(status: str, model: str) -> Dict[str, Any]:
    return {
        "type": f"response.{status}",
        "response": {
            "id": f"resp_{uuid4().hex}",
            "object": "response",
            "created_at": int(time.time()),
            "status": status if status != "created" else "in_progress",
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "max_output_tokens": None,
            "model": model,
            "output": [],
            "parallel_tool_calls": True,
            "previous_response_id": None,
            "reasoning": {"effort": None, "summary": None},
            "store": True,
            "temperature": 1.0,
            "text": {"format": {"type": "text"}},
            "tool_choice": "auto",
            "tools": [],
            "top_p": 1.0,
            "truncation": "disabled",
            "usage": None,
            "user": None,
            "metadata": {},
        },
    }


@app.post("/chat_stream")
async def chat_stream_endpoint(req: ChatRequest):
    """
    Streaming variant of /chat emitting Server-Sent Events compatible with
    OpenAI Responses streaming event names. Token deltas are sent as a single
    completed chunk for now; incremental token streaming can be added later.
    """
    import asyncio

    async def event_generator():
        # Log request
        try:
            logger.info("/chat_stream request: %s", {"conversation_id": req.conversation_id, "message": req.message})
        except Exception:
            pass

        # Initialize or load state (same as /chat)
        is_new = not req.conversation_id or conversation_store.get(req.conversation_id) is None
        if is_new:
            conversation_id: str = req.conversation_id or uuid4().hex
            ctx = create_initial_context()
            current_agent_name = triage_agent.name
            state: Dict[str, Any] = {
                "input_items": [],
                "context": ctx,
                "current_agent": current_agent_name,
            }
            conversation_store.save(conversation_id, state)
            # Immediately emit created + in_progress
            base_created = _base_response_envelope("created", AGENT_MODEL)
            yield _format_sse("response.created", base_created)
            base_inprog = _base_response_envelope("in_progress", AGENT_MODEL)
            yield _format_sse("response.in_progress", base_inprog)
            # If empty message (boot), complete quickly with envelope
            if req.message.strip() == "":
                resp = ChatResponse(
                    conversation_id=conversation_id,
                    current_agent=current_agent_name,
                    messages=[],
                    events=[],
                    context=ctx.model_dump(),
                    agents=_build_agents_list(),
                    guardrails=[],
                )
                yield _format_sse("response.completed", {"type": "response.completed", "response": resp.model_dump()})
                return
        else:
            conversation_id = req.conversation_id  # type: ignore
            state = conversation_store.get(conversation_id)

        # Normal flow
        current_agent = _get_agent_by_name(state["current_agent"])  # type: ignore[index]
        state["input_items"].append({"content": req.message, "role": "user"})
        old_context = state["context"].model_dump().copy()
        guardrail_checks: List[GuardrailCheck] = []

        # Emit created/in_progress upfront
        yield _format_sse("response.created", _base_response_envelope("created", AGENT_MODEL))
        yield _format_sse("response.in_progress", _base_response_envelope("in_progress", AGENT_MODEL))

        # Handle Human Support short-circuit
        if current_agent.name == human_support_agent.name:
            conversation_store.save(conversation_id, state)
            resp = ChatResponse(
                conversation_id=conversation_id,
                current_agent=current_agent.name,
                messages=[],
                events=[],
                context=state["context"].model_dump(),
                agents=_build_agents_list(),
                guardrails=[],
            )
            yield _format_sse("response.completed", {"type": "response.completed", "response": resp.model_dump()})
            return

        try:
            # Use streaming API per OpenAI Agents best practices
            result_stream = Runner.run_streamed(current_agent, state["input_items"], context=state["context"])  # type: ignore[index]
        except InputGuardrailTripwireTriggered as e:
            failed = e.guardrail_result.guardrail
            gr_output = e.guardrail_result.output.output_info
            gr_reasoning = getattr(gr_output, "reasoning", "")
            gr_input = req.message
            gr_timestamp = time.time() * 1000
            for g in current_agent.input_guardrails:
                guardrail_checks.append(GuardrailCheck(
                    id=uuid4().hex,
                    name=_get_guardrail_name(g),
                    input=gr_input,
                    reasoning=(gr_reasoning if g == failed else ""),
                    passed=(g != failed),
                    timestamp=gr_timestamp,
                ))
            refusal = "Sorry, I can only answer questions related to telco support (plans, billing, usage, roaming, outages)."
            state["input_items"].append({"role": "assistant", "content": refusal})
            resp = ChatResponse(
                conversation_id=conversation_id,
                current_agent=current_agent.name,
                messages=[MessageResponse(content=refusal, agent=current_agent.name)],
                events=[],
                context=state["context"].model_dump(),
                agents=_build_agents_list(),
                guardrails=guardrail_checks,
            )
            # Minimal message streaming for refusal
            msg_id = f"msg_{uuid4().hex}"
            yield _format_sse(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "id": msg_id,
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                        "agent": current_agent.name,
                    },
                },
            )
            yield _format_sse(
                "response.content_part.added",
                {
                    "type": "response.content_part.added",
                    "item_id": msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                },
            )
            yield _format_sse(
                "response.output_text.done",
                {
                    "type": "response.output_text.done",
                    "item_id": msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": refusal,
                },
            )
            yield _format_sse(
                "response.content_part.done",
                {
                    "type": "response.content_part.done",
                    "item_id": msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": refusal, "annotations": []},
                },
            )
            yield _format_sse(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "id": msg_id,
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": refusal, "annotations": []}],
                    },
                },
            )
            yield _format_sse("response.completed", {"type": "response.completed", "response": resp.model_dump()})
            return

        messages: List[MessageResponse] = []
        events: List[AgentEvent] = []
        last_agent_name: str = current_agent.name
        final_text: str = ""

        try:
            async for ev in result_stream.stream_events():
                et = getattr(ev, "type", "")
                # Forward raw Responses events token-by-token
                if et == "raw_response_event":
                    try:
                        raw = getattr(ev, "data", None)
                        if raw is None:
                            continue
                        to_dict = getattr(raw, "model_dump", None)
                        data_dict = to_dict() if callable(to_dict) else json.loads(getattr(raw, "json", lambda: "{}")())
                        ev_type = data_dict.get("type") or "message"
                        # Annotate item with current agent so UI can color/label bubbles
                        try:
                            item = data_dict.get("item")
                            if isinstance(item, dict):
                                item.setdefault("agent", last_agent_name)
                            data_dict.setdefault("agent", last_agent_name)
                        except Exception:
                            pass
                        if ev_type == "response.output_text.delta":
                            final_text += data_dict.get("delta", "")
                        # Forward the raw Responses event
                        yield _format_sse(ev_type, data_dict)
                    except Exception:
                        # Always keep stream alive
                        yield _format_sse("message", {"type": "raw_response_event"})
                elif et == "agent_updated_stream_event":
                    try:
                        prev = last_agent_name
                        new_agent = getattr(ev, "new_agent", None)
                        if new_agent is not None:
                            last_agent_name = getattr(new_agent, "name", last_agent_name)
                        events.append(AgentEvent(id=uuid4().hex, type="handoff", agent=prev, content=f"{prev} -> {last_agent_name}", metadata={"source_agent": prev, "target_agent": last_agent_name}))
                        yield _format_sse("response.event", {"type": "handoff", "source_agent": prev, "target_agent": last_agent_name})
                    except Exception:
                        pass
                elif et == "run_item_stream_event":
                    try:
                        item = getattr(ev, "item", None)
                        if isinstance(item, ToolCallItem):
                            tool_name = getattr(item.raw_item, "name", None)
                            raw_args = getattr(item.raw_item, "arguments", None)
                            tool_args: Any = raw_args
                            if isinstance(raw_args, str):
                                try:
                                    tool_args = json.loads(raw_args)
                                except Exception:
                                    pass
                            meta = {"tool_args": tool_args}
                            events.append(AgentEvent(id=uuid4().hex, type="tool_call", agent=item.agent.name, content=tool_name or "", metadata=meta))
                            yield _format_sse("response.event", {"type": "tool_call", "agent": item.agent.name, "tool": tool_name or "", "metadata": meta})
                        elif isinstance(item, ToolCallOutputItem):
                            events.append(AgentEvent(id=uuid4().hex, type="tool_output", agent=item.agent.name, content=str(item.output), metadata={"tool_result": item.output}))
                            yield _format_sse("response.event", {"type": "tool_output", "agent": item.agent.name})
                        elif isinstance(item, MessageOutputItem):
                            # Record message for completion envelope; live streaming already forwarded above
                            text = ItemHelpers.text_message_output(item)
                            messages.append(MessageResponse(content=text, agent=item.agent.name))
                    except Exception:
                        pass
        except InputGuardrailTripwireTriggered as e:
            # If a guardrail triggers during streaming, emit a refusal and complete gracefully
            failed = e.guardrail_result.guardrail
            gr_output = e.guardrail_result.output.output_info
            gr_reasoning = getattr(gr_output, "reasoning", "")
            gr_input = req.message
            gr_timestamp = time.time() * 1000
            for g in current_agent.input_guardrails:
                guardrail_checks.append(GuardrailCheck(
                    id=uuid4().hex,
                    name=_get_guardrail_name(g),
                    input=gr_input,
                    reasoning=(gr_reasoning if g == failed else ""),
                    passed=(g != failed),
                    timestamp=gr_timestamp,
                ))
            refusal = "Sorry, I can only answer questions related to telco support (plans, billing, usage, roaming, outages)."
            state["input_items"].append({"role": "assistant", "content": refusal})
            resp = ChatResponse(
                conversation_id=conversation_id,
                current_agent=current_agent.name,
                messages=[MessageResponse(content=refusal, agent=current_agent.name)],
                events=[],
                context=state["context"].model_dump(),
                agents=_build_agents_list(),
                guardrails=guardrail_checks,
            )
            msg_id = f"msg_{uuid4().hex}"
            yield _format_sse(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "id": msg_id,
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                        "agent": current_agent.name,
                    },
                },
            )
            yield _format_sse(
                "response.content_part.added",
                {
                    "type": "response.content_part.added",
                    "item_id": msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                },
            )
            yield _format_sse(
                "response.output_text.done",
                {
                    "type": "response.output_text.done",
                    "item_id": msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": refusal,
                },
            )
            yield _format_sse(
                "response.content_part.done",
                {
                    "type": "response.content_part.done",
                    "item_id": msg_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": refusal, "annotations": []},
                },
            )
            yield _format_sse(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "id": msg_id,
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": refusal, "annotations": []}],
                    },
                },
            )
            yield _format_sse("response.completed", {"type": "response.completed", "response": resp.model_dump()})
            return

        # No manual follow-up run here; rely on agent updates via streaming

        # Context changes
        new_context = state["context"].dict()
        changes = {k: new_context[k] for k in new_context if old_context.get(k) != new_context[k]}
        if changes:
            events.append(AgentEvent(id=uuid4().hex, type="context_update", agent=current_agent.name, content="", metadata={"changes": changes}))
            yield _format_sse("response.event", {"type": "context_update", "agent": current_agent.name, "changes": changes})

        # Persist
        try:
            # Derive final inputs from messages list + last user
            if final_text:
                state["input_items"].append({"role": "assistant", "content": final_text})
            state["current_agent"] = last_agent_name
        except Exception:
            pass
        conversation_store.save(conversation_id, state)

        # Build guardrail results
        final_guardrails: List[GuardrailCheck] = []
        for g in getattr(current_agent, "input_guardrails", []):
            name = _get_guardrail_name(g)
            failed = next((gc for gc in guardrail_checks if gc.name == name), None)
            if failed:
                final_guardrails.append(failed)
            else:
                final_guardrails.append(GuardrailCheck(
                    id=uuid4().hex,
                    name=name,
                    input=req.message,
                    reasoning="",
                    passed=True,
                    timestamp=time.time() * 1000,
                ))

        complete = ChatResponse(
            conversation_id=conversation_id,
            current_agent=last_agent_name,
            messages=messages if messages else ([MessageResponse(content=final_text, agent=last_agent_name)] if final_text else []),
            events=events,
            context=state["context"].dict(),
            agents=_build_agents_list(),
            guardrails=final_guardrails,
        )
        yield _format_sse("response.completed", {"type": "response.completed", "response": complete.model_dump()})

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_generator(), headers=headers)

@app.post("/chat_stream_direct")
async def chat_stream_direct_endpoint(req: ChatRequest):
    """
    Proof-of-concept: bypass Agent Runner and stream directly from OpenAI Responses
    with stream=True. Forwards OpenAI event names and deltas as they arrive.
    On completion, emits a synthetic response.completed with our ChatResponse envelope
    so the existing UI updates state and unlocks the input.
    """
    model = os.getenv("DIRECT_STREAM_MODEL", os.getenv("AGENT_MODEL", "gpt-4.1"))
    instructions = os.getenv(
        "DIRECT_STREAM_INSTRUCTIONS",
        "You are a helpful telco customer service assistant. Be concise and friendly.",
    )

    # Prepare conversation state (lightweight)
    is_new = not req.conversation_id or conversation_store.get(req.conversation_id) is None
    if is_new:
        conversation_id: str = req.conversation_id or uuid4().hex
        ctx = create_initial_context()
        state: Dict[str, Any] = {
            "input_items": [],
            "context": ctx,
            "current_agent": triage_agent.name,
        }
        conversation_store.save(conversation_id, state)
    else:
        conversation_id = req.conversation_id  # type: ignore
        state = conversation_store.get(conversation_id)
        if state is None:
            # Shouldn't happen; make a brand new one
            ctx = create_initial_context()
            state = {"input_items": [], "context": ctx, "current_agent": triage_agent.name}
    state["input_items"].append({"role": "user", "content": req.message})

    client = OpenAI()

    def _summarize_context(ctx: Any) -> str:
        try:
            data = ctx.model_dump() if hasattr(ctx, "model_dump") else dict(ctx)
        except Exception:
            return ""
        keys = [
            "customer_name",
            "account_number",
            "mobile_number",
            "plan_name",
            "data_remaining_gb",
            "billing_balance",
            "address_postal_code",
            "ticket_id",
            "roaming_active",
        ]
        parts = []
        for k in keys:
            v = data.get(k)
            if v is not None and v != "":
                parts.append(f"{k}={v}")
        return "; ".join(parts)

    def _build_transcript(items: List[Dict[str, Any]], latest_user: str, agent_name: str, ctx: Any) -> str:
        ctx_line = _summarize_context(ctx)
        lines: List[str] = []
        if ctx_line:
            lines.append(f"[Context] {ctx_line}")
        # Replay brief transcript
        for it in items[-8:]:  # cap history to last 8 turns for brevity
            role = it.get("role")
            content = it.get("content")
            if not content:
                continue
            if role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant":
                lines.append(f"{agent_name}: {content}")
        lines.append(f"User: {latest_user}")
        lines.append(f"{agent_name}:")
        return "\n".join(lines)

    def _tools_schema() -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "handoff_to",
                "description": "Switch to a specialist agent when appropriate.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_name": {
                            "type": "string",
                            "enum": [
                                triage_agent.name,
                                faq_agent.name,
                                plan_change_agent.name,
                                billing_agent.name,
                                tech_support_agent.name,
                                data_usage_agent.name,
                                roaming_agent.name,
                                human_support_agent.name,
                            ],
                        }
                    },
                    "required": ["agent_name"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "telco_faq_lookup",
                "description": "Lookup common telco FAQs (local KB).",
                "parameters": {
                    "type": "object",
                    "properties": {"question": {"type": "string"}},
                    "required": ["question"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "upgrade_plan",
                "description": "Upgrade the customer's plan for a given mobile number.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mobile_number": {"type": "string"},
                        "new_plan": {"type": "string"},
                    },
                    "required": ["mobile_number", "new_plan"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "check_data_usage",
                "description": "Check/update data usage for a mobile number.",
                "parameters": {
                    "type": "object",
                    "properties": {"mobile_number": {"type": "string"}},
                    "required": ["mobile_number"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "check_outage",
                "description": "Check if any known outages at a postal code.",
                "parameters": {
                    "type": "object",
                    "properties": {"postal_code": {"type": "string"}},
                    "required": ["postal_code"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "pay_bill",
                "description": "Apply a bill payment to the customer's account.",
                "parameters": {
                    "type": "object",
                    "properties": {"amount": {"type": "number"}},
                    "required": ["amount"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "activate_roaming",
                "description": "Activate international roaming.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mobile_number": {"type": "string"},
                        "destination_country": {"type": "string"},
                    },
                    "required": ["mobile_number"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "deactivate_roaming",
                "description": "Deactivate international roaming.",
                "parameters": {
                    "type": "object",
                    "properties": {"mobile_number": {"type": "string"}},
                    "required": ["mobile_number"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "book_technician",
                "description": "Book a technician appointment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "postal_code": {"type": "string"},
                        "preferred_timeslot": {"type": "string"},
                    },
                    "required": ["postal_code", "preferred_timeslot"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "get_current_plan",
                "description": "Return the current plan from context.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "type": "function",
                "name": "list_available_plans",
                "description": "List available mobile plans.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "type": "function",
                "name": "plan_details",
                "description": "Show detailed specifications for a named plan.",
                "parameters": {
                    "type": "object",
                    "properties": {"plan_name": {"type": "string"}},
                    "required": ["plan_name"],
                    "additionalProperties": False,
                },
            },
        ]

    def _execute_tool(name: str, args: Dict[str, Any]) -> str:
        try:
            ctx = state["context"]
            if name == "handoff_to":
                target = str(args.get("agent_name", "")).strip() or triage_agent.name
                prev = state.get("current_agent", triage_agent.name)
                state["current_agent"] = target
                return f"__HANDOFF__::{prev}::{target}"
            if name == "telco_faq_lookup":
                try:
                    from main import telco_faq_lookup as _faq
                    import anyio
                    return anyio.from_thread.run(_faq, args.get("question", ""))  # type: ignore
                except Exception:
                    return "Sorry, FAQ lookup is temporarily unavailable."
            if name == "upgrade_plan":
                ctx.plan_name = str(args.get("new_plan", ctx.plan_name))
                ctx.mobile_number = str(args.get("mobile_number", ctx.mobile_number))
                return f"Upgraded {ctx.mobile_number} to plan '{ctx.plan_name}'."
            if name == "check_data_usage":
                import random as _r
                ctx.mobile_number = str(args.get("mobile_number", ctx.mobile_number))
                ctx.data_remaining_gb = round(_r.uniform(0, 100), 1)
                return f"{ctx.mobile_number} has {ctx.data_remaining_gb} GB remaining this cycle."
            if name == "check_outage":
                pc = str(args.get("postal_code", ""))
                affected = {"03895", "52953", "31049"}
                if pc in affected:
                    return f"Known broadband outage affecting area {pc}. Technicians are on-site, ETA 2 hours."
                return f"No known outages at {pc}."
            if name == "pay_bill":
                amt = float(args.get("amount", 0))
                cur = getattr(ctx, "billing_balance", 0.0) or 0.0
                new_bal = round(max(0.0, float(cur) - amt), 2)
                ctx.billing_balance = new_bal
                return f"Payment of ${amt:.2f} received. Outstanding balance is now ${new_bal:.2f}."
            if name == "activate_roaming":
                ctx.mobile_number = str(args.get("mobile_number", ctx.mobile_number))
                ctx.roaming_active = True
                dc = args.get("destination_country")
                return f"Roaming activated for {ctx.mobile_number}{(' for travel to ' + dc) if dc else ''}."
            if name == "deactivate_roaming":
                ctx.mobile_number = str(args.get("mobile_number", ctx.mobile_number))
                ctx.roaming_active = False
                return f"Roaming deactivated for {ctx.mobile_number}."
            if name == "book_technician":
                ctx.address_postal_code = str(args.get("postal_code", ctx.address_postal_code))
                import random as _r
                t = f"{_r.randint(0, 9999999):07d}"
                ctx.ticket_id = t
                ts = str(args.get("preferred_timeslot", ""))
                return f"Technician booked for {ctx.address_postal_code} at {ts}. Ticket {t}."
            if name == "get_current_plan":
                plan = getattr(ctx, "plan_name", None)
                mobile = getattr(ctx, "mobile_number", None) or "[unknown number]"
                if not plan:
                    return "No plan is on file yet for this session."
                return f"Current plan on file is '{plan}' for {mobile}."
            if name == "list_available_plans":
                from main import PLAN_CATALOG
                lines = ["Here are our current mobile plans:"]
                for p in PLAN_CATALOG:
                    lines.append(
                        f"- {p.name} — ${p.monthly_price_sgd}/mo, {p.data_gb}GB data, "
                        f"{'5G SA' if p.includes_5g_sa else '4G'}, {p.talktime_sms}"
                        + (f", Contract: {p.contract_months} months" if p.contract_months else ", No contract")
                        + (f", {p.roaming_note}" if p.roaming_note else "")
                    )
                lines.append("Reply with a plan name if you'd like more details or to switch.")
                return "\n".join(lines)
            if name == "plan_details":
                from main import PLAN_CATALOG
                target = str(args.get("plan_name", "")).lower()
                for p in PLAN_CATALOG:
                    if p.name.lower() == target:
                        return (
                            f"Plan: {p.name}\n"
                            f"Price: ${p.monthly_price_sgd}/mo\n"
                            f"Data: {p.data_gb}GB\n"
                            f"Network: {'5G Standalone' if p.includes_5g_sa else '4G/LTE'}\n"
                            f"Talktime/SMS: {p.talktime_sms}\n"
                            f"Contract: {p.contract_months or 'None'} months\n"
                            f"Roaming: {p.roaming_note or 'See app for options'}"
                        )
                return "Sorry, I couldn't find that plan. Please provide an exact plan name."
        except Exception as ex:
            return f"Tool '{name}' failed: {ex}"
        return f"Tool '{name}' not implemented."

    def openai_stream_worker(q: "queue.Queue[dict | None]"):
        try:
            current_agent_name = state.get("current_agent", triage_agent.name)
            transcript = _build_transcript(state["input_items"], req.message, current_agent_name, state["context"])  # type: ignore[index]
            dir_instructions = (
                f"{instructions}\n\n"
                f"You are currently acting as '{current_agent_name}'. "
                f"If another specialist agent is more appropriate, call the 'handoff_to' function with agent_name set to one of: "
                f"['{triage_agent.name}', '{faq_agent.name}', '{plan_change_agent.name}', '{billing_agent.name}', '{tech_support_agent.name}', '{data_usage_agent.name}', '{roaming_agent.name}', '{human_support_agent.name}']. "
                f"Use other tools when needed to fulfill the user's request. "
                f"After you call any function (including 'handoff_to'), you must continue the turn and produce a concise assistant reply to the user summarizing the outcome or next steps."
            )
            response_id_holder = [None]

            def push_stream(kind: str, **kwargs):
                s = (
                    client.responses.create(stream=True, **kwargs)
                    if kind == "create"
                    else client.responses.submit_tool_outputs(stream=True, **kwargs)
                )
                # Track function calls by item_id and call_id
                pending_args: Dict[str, str] = {}
                call_meta: Dict[str, Dict[str, Any]] = {}
                for ev in s:
                    try:
                        data = getattr(ev, "model_dump", None)
                        event_dict = data() if callable(data) else json.loads(str(ev))
                    except Exception:
                        try:
                            event_dict = json.loads(getattr(ev, "json", lambda: "{}")())
                        except Exception:
                            event_dict = {"type": "unknown"}
                    # Capture response id
                    try:
                        if event_dict.get("type") == "response.created":
                            rid = ((event_dict.get("response") or {}).get("id") if isinstance(event_dict.get("response"), dict) else None)
                            if isinstance(rid, str):
                                response_id_holder[0] = rid
                    except Exception:
                        pass
                    et = event_dict.get("type", "")
                    # Handle function calling stream events (Responses API)
                    if et == "response.output_item.added":
                        try:
                            item = event_dict.get("item") or {}
                            if isinstance(item, dict) and item.get("type") == "function_call":
                                item_id = item.get("id") or item.get("item_id")
                                call_id = item.get("call_id")
                                name = item.get("name") or ""
                                if isinstance(item_id, str):
                                    call_meta[item_id] = {"name": name, "call_id": call_id}
                                    pending_args[item_id] = ""
                                # UI event: tool_call
                                q.put({"type": "response.event", "data": {"type": "tool_call", "agent": state.get("current_agent", triage_agent.name), "metadata": {"tool_name": name}}})
                        except Exception:
                            pass
                    elif et == "response.function_call_arguments.delta":
                        try:
                            item_id = event_dict.get("item_id")
                            delta = event_dict.get("delta") or ""
                            if isinstance(item_id, str):
                                if item_id not in pending_args:
                                    pending_args[item_id] = ""
                                pending_args[item_id] += str(delta)
                        except Exception:
                            pass
                    elif et == "response.output_item.done":
                        try:
                            item = event_dict.get("item") or {}
                            if isinstance(item, dict) and item.get("type") == "function_call":
                                item_id = item.get("id")
                                meta = call_meta.get(item_id or "") or {}
                                name = meta.get("name") or ""
                                call_id = meta.get("call_id")
                                args_json = pending_args.get(item_id or "", "{}")
                                try:
                                    args_obj = json.loads(args_json) if isinstance(args_json, str) else {}
                                except Exception:
                                    args_obj = {}
                                output_text = _execute_tool(name, args_obj)
                                # Emit tool_output (use friendly text if this is a handoff marker)
                                if isinstance(output_text, str) and output_text.startswith("__HANDOFF__::"):
                                    try:
                                        _, f, t = output_text.split("::", 2)
                                        friendly = f"Handed off from {f} to {t}."
                                    except Exception:
                                        friendly = "Handed off to target agent."
                                    tool_result_for_ui = friendly
                                else:
                                    tool_result_for_ui = output_text
                                q.put({"type": "response.event", "data": {"type": "tool_output", "agent": state.get("current_agent", triage_agent.name), "metadata": {"tool_result": tool_result_for_ui}}})
                                # Handle handoff marker
                                if isinstance(output_text, str) and output_text.startswith("__HANDOFF__::"):
                                    try:
                                        _, f, t = output_text.split("::", 2)
                                    except Exception:
                                        f, t = state.get("current_agent", triage_agent.name), triage_agent.name
                                    q.put({"type": "response.event", "data": {"type": "handoff", "source_agent": f, "target_agent": t}})
                                    output_to_model = f"Handed off from {f} to {t}."
                                else:
                                    output_to_model = output_text
                                rid = response_id_holder[0]
                                if isinstance(rid, str) and isinstance(call_id, str):
                                    for sev in client.responses.submit_tool_outputs(
                                        response_id=rid,
                                        tool_outputs=[{"tool_call_id": call_id, "output": output_to_model}],
                                        stream=True,
                                    ):
                                        try:
                                            data2 = getattr(sev, "model_dump", None)
                                            event_dict2 = data2() if callable(data2) else json.loads(str(sev))
                                        except Exception:
                                            try:
                                                event_dict2 = json.loads(getattr(sev, "json", lambda: "{}")())
                                            except Exception:
                                                event_dict2 = {"type": "unknown"}
                                        q.put(event_dict2)
                                # Skip forwarding this done event; it's handled
                                continue
                        except Exception:
                            pass
                    # Forward the event by default
                    q.put(event_dict)

            # Start with tools enabled
            push_stream(
                "create",
                model=model,
                instructions=dir_instructions,
                input=transcript,
                tools=_tools_schema(),
                tool_choice="auto",
                parallel_tool_calls=True,
            )
        except Exception as ex:
            # Surface a detailed error event to the client
            detail = str(ex)
            try:
                resp = getattr(ex, 'response', None)
                if resp is not None:
                    body = getattr(resp, 'text', None) or getattr(resp, 'content', None)
                    if body:
                        detail += f" | body={body}"
            except Exception:
                pass
            q.put({"type": "error", "error": detail})
        finally:
            q.put(None)

    async def event_generator():
        # If current agent is Human Support, do not invoke the model
        try:
            current_name = state.get("current_agent")  # type: ignore[assignment]
        except Exception:
            current_name = None
        if current_name == human_support_agent.name:
            conversation_store.save(conversation_id, state)
            complete = ChatResponse(
                conversation_id=conversation_id,
                current_agent=human_support_agent.name,
                messages=[],
                events=[],
                context=state["context"].model_dump() if hasattr(state["context"], "model_dump") else state["context"],
                agents=_build_agents_list(),
                guardrails=[],
            )
            yield _format_sse("response.completed", {"type": "response.completed", "response": complete.model_dump()})
            return

        # Forward OpenAI events live and accumulate final message text
        q: "queue.Queue[dict | None]" = queue.Queue()
        worker = threading.Thread(target=openai_stream_worker, args=(q,), daemon=True)
        worker.start()
        final_text = ""
        last_agent = state.get("current_agent", triage_agent.name)
        output_index = 0
        synthesized_messages: List[MessageResponse] = []
        handoff_to_human = False
        # Do not surface raw tool output to the end user chat bubble; only show model replies
        while True:
            ev = await asyncio.get_event_loop().run_in_executor(None, q.get)
            if ev is None:
                break
            ev_type = ev.get("type")
            # Track handoffs to update current agent promptly
            if ev_type == "response.event":
                try:
                    data = ev.get("data") or {}
                    if isinstance(data, dict) and data.get("type") == "handoff":
                        target = data.get("target_agent") or state.get("current_agent", triage_agent.name)
                        if isinstance(target, str) and target:
                            last_agent = target
                            try:
                                state["current_agent"] = target
                            except Exception:
                                pass
                            # If handed to Human Support, end streaming without auto AI text
                            if target == human_support_agent.name:
                                handoff_to_human = True
                            else:
                                # Emit an immediate greeting from the new AI agent
                                msg_id = f"msg_{uuid4().hex}"
                                greet = f"Hi, you are now connected to {target}. How can I help further?"
                                yield _format_sse(
                                    "response.output_item.added",
                                    {
                                        "type": "response.output_item.added",
                                        "output_index": output_index,
                                        "item": {
                                            "id": msg_id,
                                            "type": "message",
                                            "status": "in_progress",
                                            "role": "assistant",
                                            "content": [],
                                            "agent": target,
                                        },
                                    },
                                )
                                yield _format_sse(
                                    "response.content_part.added",
                                    {
                                        "type": "response.content_part.added",
                                        "item_id": msg_id,
                                        "output_index": output_index,
                                        "content_index": 0,
                                        "part": {"type": "output_text", "text": "", "annotations": []},
                                    },
                                )
                                yield _format_sse(
                                    "response.output_text.done",
                                    {
                                        "type": "response.output_text.done",
                                        "item_id": msg_id,
                                        "output_index": output_index,
                                        "content_index": 0,
                                        "text": greet,
                                    },
                                )
                                yield _format_sse(
                                    "response.content_part.done",
                                    {
                                        "type": "response.content_part.done",
                                        "item_id": msg_id,
                                        "output_index": output_index,
                                        "content_index": 0,
                                        "part": {"type": "output_text", "text": greet, "annotations": []},
                                    },
                                )
                                yield _format_sse(
                                    "response.output_item.done",
                                    {
                                        "type": "response.output_item.done",
                                        "output_index": output_index,
                                        "item": {
                                            "id": msg_id,
                                            "type": "message",
                                            "status": "completed",
                                            "role": "assistant",
                                            "content": [{"type": "output_text", "text": greet, "annotations": []}],
                                        },
                                    },
                                )
                                synthesized_messages.append(MessageResponse(content=greet, agent=target))
                                output_index += 1
                    # Tool outputs are forwarded as runner events only; never populate chat text directly
                except Exception:
                    pass
            # If handed off to Human Support, stop reading further model events
            if handoff_to_human:
                break
            # Accumulate text for completion envelope
            if ev_type == "response.output_text.delta":
                final_text += ev.get("delta", "")
            elif ev_type == "response.output_item.added":
                # Try to read agent role if present
                try:
                    item = ev.get("item") or {}
                    if isinstance(item, dict):
                        # Inject our agent for bubble color and labeling
                        item_agent = item.get("agent") or last_agent
                        if not item_agent:
                            item_agent = last_agent
                        item["agent"] = item_agent
                        last_agent = item_agent
                        ev["item"] = item
                except Exception:
                    pass
            # Forward all events except the upstream completion
            if ev_type == "response.completed":
                continue
            try:
                yield _format_sse(ev_type or "message", ev)
            except Exception:
                # Always keep the stream flowing
                yield _format_sse("message", {"type": ev_type or "message"})

        # Synthesize a completion envelope compatible with our UI
        messages: List[MessageResponse] = []
        if synthesized_messages:
            messages.extend(synthesized_messages)
        if final_text:
            messages.append(MessageResponse(content=final_text, agent=last_agent))
            # Update persistent state
            try:
                state["input_items"].append({"role": "assistant", "content": final_text})
                state["current_agent"] = last_agent
                conversation_store.save(conversation_id, state)
            except Exception:
                pass
        # Prefer the latest state for current agent
        final_agent_name = None
        try:
            final_agent_name = state.get("current_agent")
        except Exception:
            final_agent_name = None
        complete = ChatResponse(
            conversation_id=conversation_id,
            current_agent=final_agent_name or last_agent,
            messages=messages,
            events=[],
            context=state["context"].model_dump() if hasattr(state["context"], "model_dump") else state["context"],
            agents=_build_agents_list(),
            guardrails=[],
        )
        yield _format_sse("response.completed", {"type": "response.completed", "response": complete.model_dump()})

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_generator(), headers=headers)

@app.post("/human_reply", response_model=ChatResponse)
async def human_reply_endpoint(req: HumanReplyRequest):
    try:
        logger.info("/human_reply request: %s", {"conversation_id": req.conversation_id, "message": req.message})
    except Exception:
        pass
    conversation_id = req.conversation_id
    state = conversation_store.get(conversation_id)
    if state is None:
        # Initialize a new state defaulting to Human Support
        ctx = create_initial_context()
        state = {
            "input_items": [],
            "context": ctx,
            "current_agent": human_support_agent.name,
        }
    else:
        # Switch to human support if not already
        if state["current_agent"] != human_support_agent.name:
            state["current_agent"] = human_support_agent.name

    # Guardrail check on the human message using triage guardrails
    guardrail_checks: List[GuardrailCheck] = []
    try:
        _ = await Runner.run(triage_agent, [{"role": "user", "content": req.message}], context=state["context"])
        guardrails_ok = True
    except InputGuardrailTripwireTriggered as e:
        guardrails_ok = False
        failed = e.guardrail_result.guardrail
        gr_output = e.guardrail_result.output.output_info
        gr_reasoning = getattr(gr_output, "reasoning", "")
        gr_input = req.message
        gr_timestamp = time.time() * 1000
        for g in triage_agent.input_guardrails:
            guardrail_checks.append(GuardrailCheck(
                id=uuid4().hex,
                name=(getattr(g, "name", None) or getattr(getattr(g, "guardrail_function", None), "__name__", "Guardrail")),
                input=gr_input,
                reasoning=(gr_reasoning if g == failed else ""),
                passed=(g != failed),
                timestamp=gr_timestamp,
            ))

    if not guardrails_ok:
        # Do not append the message; surface failure to UI so operator can try again
        conversation_store.save(conversation_id, state)
        resp = ChatResponse(
            conversation_id=conversation_id,
            current_agent=human_support_agent.name,
            messages=[],
            events=[],
            context=state["context"].model_dump(),
            agents=_build_agents_list(),
            guardrails=guardrail_checks,
        )
        try:
            logger.info("/human_reply response (guardrail failed): %s", resp.model_dump())
        except Exception:
            pass
        return resp

    # Guardrails passed: append assistant message from human and mark resume hint
    state["input_items"].append({"role": "assistant", "content": req.message})
    try:
        setattr(state["context"], "resume_hint", True)
    except Exception:
        pass

    messages = [MessageResponse(content=req.message, agent=human_support_agent.name)]
    events = [
        AgentEvent(
            id=uuid4().hex,
            type="message",
            agent=human_support_agent.name,
            content=req.message,
        )
    ]

    conversation_store.save(conversation_id, state)
    resp = ChatResponse(
        conversation_id=conversation_id,
        current_agent=human_support_agent.name,
        messages=messages,
        events=events,
        context=state["context"].model_dump(),
        agents=_build_agents_list(),
        guardrails=[],
    )
    try:
        logger.info("/human_reply response: %s", resp.model_dump())
    except Exception:
        pass
    return resp


@app.post("/human_back", response_model=ChatResponse)
async def human_back_endpoint(req: HumanBackRequest):
    try:
        logger.info("/human_back request: %s", {"conversation_id": req.conversation_id})
    except Exception:
        pass
    conversation_id = req.conversation_id
    state = conversation_store.get(conversation_id)
    if state is None:
        # Nothing to resume; create a fresh triage session
        ctx = create_initial_context()
        state = {
            "input_items": [],
            "context": ctx,
            "current_agent": triage_agent.name,
        }
    # Prefer resuming with a remembered agent (e.g., the specialist before human handoff)
    resume_name = None
    try:
        resume_name = state.get("resume_agent_name")  # type: ignore
    except Exception:
        resume_name = None
    current_agent = _get_agent_by_name(resume_name) if resume_name else triage_agent
    old_context = state["context"].model_dump().copy()

    result = await Runner.run(current_agent, state["input_items"], context=state["context"])

    messages: List[MessageResponse] = []
    events: List[AgentEvent] = []

    for item in result.new_items:
        if isinstance(item, MessageOutputItem):
            text = ItemHelpers.text_message_output(item)
            messages.append(MessageResponse(content=text, agent=item.agent.name))
            events.append(AgentEvent(id=uuid4().hex, type="message", agent=item.agent.name, content=text))
        elif isinstance(item, HandoffOutputItem):
            events.append(
                AgentEvent(
                    id=uuid4().hex,
                    type="handoff",
                    agent=item.source_agent.name,
                    content=f"{item.source_agent.name} -> {item.target_agent.name}",
                    metadata={"source_agent": item.source_agent.name, "target_agent": item.target_agent.name},
                )
            )
            current_agent = item.target_agent
        elif isinstance(item, ToolCallItem):
            tool_name = getattr(item.raw_item, "name", None)
            raw_args = getattr(item.raw_item, "arguments", None)
            tool_args: Any = raw_args
            if isinstance(raw_args, str):
                try:
                    import json
                    tool_args = json.loads(raw_args)
                except Exception:
                    pass
            events.append(
                AgentEvent(
                    id=uuid4().hex,
                    type="tool_call",
                    agent=item.agent.name,
                    content=tool_name or "",
                    metadata={"tool_args": tool_args},
                )
            )
        elif isinstance(item, ToolCallOutputItem):
            events.append(
                AgentEvent(
                    id=uuid4().hex,
                    type="tool_output",
                    agent=item.agent.name,
                    content=str(item.output),
                    metadata={"tool_result": item.output},
                )
            )

    new_context = state["context"].dict()
    changes = {k: new_context[k] for k in new_context if old_context.get(k) != new_context[k]}
    if changes:
        events.append(
            AgentEvent(
                id=uuid4().hex,
                type="context_update",
                agent=current_agent.name,
                content="",
                metadata={"changes": changes},
            )
        )

    state["input_items"] = result.to_input_list()
    state["current_agent"] = current_agent.name
    # Clear resume hint and resume agent name after resuming
    try:
        if hasattr(state["context"], "resume_hint"):
            setattr(state["context"], "resume_hint", None)
    except Exception:
        pass
    # Clear resume hint after resuming
    if "resume_agent_name" in state:
        try:
            del state["resume_agent_name"]
        except Exception:
            state["resume_agent_name"] = None
    conversation_store.save(conversation_id, state)

    resp = ChatResponse(
        conversation_id=conversation_id,
        current_agent=current_agent.name,
        messages=messages,
        events=events,
        context=state["context"].dict(),
        agents=_build_agents_list(),
        guardrails=[],
    )
    try:
        logger.info("/human_back response: %s", resp.model_dump())
    except Exception:
        pass
    return resp
