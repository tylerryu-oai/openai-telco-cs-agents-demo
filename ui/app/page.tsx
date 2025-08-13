"use client";

import { useEffect, useState } from "react";
import { AgentPanel } from "@/components/agent-panel";
import { Chat } from "@/components/Chat";
import type { Agent, AgentEvent, GuardrailCheck, Message } from "@/lib/types";
import { callChatAPI, callHumanReplyAPI, callHumanBackAPI } from "@/lib/api";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [currentAgent, setCurrentAgent] = useState<string>("");
  const [guardrails, setGuardrails] = useState<GuardrailCheck[]>([]);
  const [context, setContext] = useState<Record<string, any>>({});
  const [conversationId, setConversationId] = useState<string | null>(null);
  // Loading state while awaiting assistant response
  const [isLoading, setIsLoading] = useState(false);
  const [isHumanSending, setIsHumanSending] = useState(false);
  const [humanFailedReason, setHumanFailedReason] = useState<string | null>(null);
  const [colorEnabled, setColorEnabled] = useState(true);

  // Boot the conversation
  useEffect(() => {
    (async () => {
      const bootStart = Date.now();
      const data = await callChatAPI("", conversationId ?? "");
      if (!data) return; // gracefully skip if backend is unavailable
      setConversationId(data.conversation_id);
      setCurrentAgent(data.current_agent);
      setContext(data.context);
      const initialEvents = (data.events || []).map((e: any) => ({
        ...e,
        timestamp: e.timestamp ?? Date.now(),
      }));
      setEvents(initialEvents);
      setAgents(data.agents || []);
      setGuardrails(data.guardrails || []);
      if (Array.isArray(data.messages)) {
        const latencyMs = Date.now() - bootStart;
        setMessages(
          data.messages.map((m: any) => ({
            id: Date.now().toString() + Math.random().toString(),
            content: m.content,
            role: "assistant",
            agent: m.agent,
            timestamp: new Date(),
            latencyMs,
          }))
        );
      }
    })();
  }, []);

  // Send a user message
  const handleSendMessage = async (content: string) => {
    const started = Date.now();
    const userMsg: Message = {
      id: Date.now().toString(),
      content,
      role: "user",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    const data = await callChatAPI(content, conversationId ?? "");
    if (!data) {
      // Backend error. Show a friendly assistant message and unlock input.
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString() + Math.random().toString(),
          content:
            "Sorry, we're having trouble processing your request right now. Please try again in a moment.",
          role: "assistant",
          agent: "System",
          timestamp: new Date(),
        },
      ]);
      setIsLoading(false);
      return;
    }

    if (!conversationId || conversationId !== data.conversation_id) {
      setConversationId(data.conversation_id);
    }
    setCurrentAgent(data.current_agent);
    setContext(data.context);
    if (data.events) {
      const stamped = data.events.map((e: any) => ({
        ...e,
        timestamp: e.timestamp ?? Date.now(),
      }));
      setEvents((prev) => [...prev, ...stamped]);
    }
    if (data.agents) setAgents(data.agents);
    // Update guardrails state
    if (data.guardrails) setGuardrails(data.guardrails);

    if (data.messages) {
      const latencyMs = Date.now() - started;
      const responses: Message[] = data.messages.map((m: any) => ({
        id: Date.now().toString() + Math.random().toString(),
        content: m.content,
        role: "assistant",
        agent: m.agent,
        timestamp: new Date(),
        latencyMs,
      }));
      setMessages((prev) => [...prev, ...responses]);
    }

    setIsLoading(false);
  };

  // Human operator reply
  const handleHumanReply = async (content: string): Promise<boolean> => {
    if (!conversationId) return false;
    setIsHumanSending(true);
    const data = await callHumanReplyAPI(content, conversationId);
    if (!data) { setIsHumanSending(false); return false; }
    setCurrentAgent(data.current_agent);
    setContext(data.context);
    // Guardrail failure detection for human replies
    let failed: string | null = null;
    if (Array.isArray(data.guardrails)) {
      const bad = data.guardrails.find((gr: any) => gr && gr.passed === false);
      if (bad) failed = bad.reasoning || "Message did not pass guardrails.";
    }
    setHumanFailedReason(failed);
    if (data.events) {
      const stamped = data.events.map((e: any) => ({ ...e, timestamp: e.timestamp ?? Date.now() }));
      setEvents((prev) => [...prev, ...stamped]);
    }
    if (data.agents) setAgents(data.agents);
    if (data.messages && data.messages.length > 0) {
      const responses: Message[] = data.messages.map((m: any) => ({
        id: Date.now().toString() + Math.random().toString(),
        content: m.content,
        role: "assistant",
        agent: m.agent,
        timestamp: new Date(),
      }));
      setMessages((prev) => [...prev, ...responses]);
      setIsHumanSending(false);
      return true;
    }
    setIsHumanSending(false);
    return false;
  };

  // Return to AI
  const handleReturnToAI = async (draft?: string) => {
    if (!conversationId) return;
    const started = Date.now();
    setIsHumanSending(true);
    if (draft && draft.trim()) {
      const r = await callHumanReplyAPI(draft.trim(), conversationId);
      if (r) {
        setCurrentAgent(r.current_agent);
        setContext(r.context);
        // Check for guardrail failure when sending draft
        let failed: string | null = null;
        if (Array.isArray(r.guardrails)) {
          const bad = r.guardrails.find((gr: any) => gr && gr.passed === false);
          if (bad) failed = bad.reasoning || "Message did not pass guardrails.";
        }
        if (failed) {
          setHumanFailedReason(failed);
          setIsHumanSending(false);
          return; // abort returning to AI to allow retry
        }
        setHumanFailedReason(null);
        if (r.events) {
          const stamped = r.events.map((e: any) => ({ ...e, timestamp: e.timestamp ?? Date.now() }));
          setEvents((prev) => [...prev, ...stamped]);
        }
        if (r.agents) setAgents(r.agents);
        if (r.messages) {
          const responses: Message[] = r.messages.map((m: any) => ({
            id: Date.now().toString() + Math.random().toString(),
            content: m.content,
            role: "assistant",
            agent: m.agent,
            timestamp: new Date(),
          }));
          setMessages((prev) => [...prev, ...responses]);
        }
      }
    }
    const data = await callHumanBackAPI(conversationId);
    if (data) {
      setCurrentAgent(data.current_agent);
      setContext(data.context);
      if (data.events) {
        const stamped = data.events.map((e: any) => ({ ...e, timestamp: e.timestamp ?? Date.now() }));
        setEvents((prev) => [...prev, ...stamped]);
      }
      if (data.agents) setAgents(data.agents);
      if (data.guardrails) setGuardrails(data.guardrails);
      if (data.messages) {
        const latencyMs = Date.now() - started;
        const responses: Message[] = data.messages.map((m: any) => ({
          id: Date.now().toString() + Math.random().toString(),
          content: m.content,
          role: "assistant",
          agent: m.agent,
          timestamp: new Date(),
          latencyMs,
        }));
        setMessages((prev) => [...prev, ...responses]);
      }
    }
    // Keep locked; Human panel will deactivate when currentAgent switches away
    setIsHumanSending(false);
  };

  return (
    <main className="flex h-screen gap-2 bg-gray-100 p-2">
      <AgentPanel
        agents={agents}
        currentAgent={currentAgent}
        events={events}
        guardrails={guardrails}
        context={context}
        onHumanReply={handleHumanReply}
        onReturnToAI={handleReturnToAI}
        isHumanSending={isHumanSending}
        humanFailedReason={humanFailedReason}
        colorEnabled={colorEnabled}
        onToggleColors={setColorEnabled}
      />
      <Chat
        messages={messages}
        onSendMessage={handleSendMessage}
        isLoading={isLoading}
        colorEnabled={colorEnabled}
      />
    </main>
  );
}
