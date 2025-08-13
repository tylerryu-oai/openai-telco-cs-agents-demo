from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from uuid import uuid4
import time
import logging

from main import (
    triage_agent,
    faq_agent,
    plan_change_agent,
    billing_agent,
    tech_support_agent,
    data_usage_agent,
    roaming_agent,
    human_support_agent,
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
        return ChatResponse(
            conversation_id=conversation_id,
            current_agent=current_agent.name,
            messages=[MessageResponse(content=refusal, agent=current_agent.name)],
            events=[],
            context=state["context"].model_dump(),
            agents=_build_agents_list(),
            guardrails=guardrail_checks,
        )

    messages: List[MessageResponse] = []
    events: List[AgentEvent] = []

    handoff_target_agent = None
    for item in result.new_items:
        if isinstance(item, MessageOutputItem):
            text = ItemHelpers.text_message_output(item)
            messages.append(MessageResponse(content=text, agent=item.agent.name))
            events.append(AgentEvent(id=uuid4().hex, type="message", agent=item.agent.name, content=text))
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
            events.append(
                AgentEvent(
                    id=uuid4().hex,
                    type="tool_call",
                    agent=item.agent.name,
                    content=tool_name or "",
                    metadata={"tool_args": tool_args},
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

    return ChatResponse(
        conversation_id=conversation_id,
        current_agent=current_agent.name,
        messages=messages,
        events=events,
        context=state["context"].dict(),
        agents=_build_agents_list(),
        guardrails=final_guardrails,
    )


@app.post("/human_reply", response_model=ChatResponse)
async def human_reply_endpoint(req: HumanReplyRequest):
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
        return ChatResponse(
            conversation_id=conversation_id,
            current_agent=human_support_agent.name,
            messages=[],
            events=[],
            context=state["context"].model_dump(),
            agents=_build_agents_list(),
            guardrails=guardrail_checks,
        )

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
    return ChatResponse(
        conversation_id=conversation_id,
        current_agent=human_support_agent.name,
        messages=messages,
        events=events,
        context=state["context"].model_dump(),
        agents=_build_agents_list(),
        guardrails=[],
    )


@app.post("/human_back", response_model=ChatResponse)
async def human_back_endpoint(req: HumanBackRequest):
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

    return ChatResponse(
        conversation_id=conversation_id,
        current_agent=current_agent.name,
        messages=messages,
        events=events,
        context=state["context"].dict(),
        agents=_build_agents_list(),
        guardrails=[],
    )
