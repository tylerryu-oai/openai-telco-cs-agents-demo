"use client";

import { Bot } from "lucide-react";
import type { Agent, AgentEvent, GuardrailCheck } from "@/lib/types";
import { AgentsList } from "./agents-list";
import { Guardrails } from "./guardrails";
import { ConversationContext } from "./conversation-context";
import { RunnerOutput } from "./runner-output";
import { HumanPanel } from "./human-panel";

interface AgentPanelProps {
  agents: Agent[];
  currentAgent: string;
  events: AgentEvent[];
  guardrails: GuardrailCheck[];
  context: Record<string, any>;
  onHumanReply: (message: string) => Promise<boolean>;
  onReturnToAI: (draft?: string) => Promise<void> | void;
  isHumanSending?: boolean;
  humanFailedReason?: string | null;
  colorEnabled: boolean;
  onToggleColors: (enabled: boolean) => void;
}

export function AgentPanel({
  agents,
  currentAgent,
  events,
  guardrails,
  context,
  onHumanReply,
  onReturnToAI,
  isHumanSending,
  humanFailedReason,
  colorEnabled,
  onToggleColors,
}: AgentPanelProps) {
  const activeAgent = agents.find((a) => a.name === currentAgent);
  const runnerEvents = events.filter((e) => e.type !== "message");
  const isHuman = currentAgent === "Human";

  return (
    <div className="w-3/5 h-full flex flex-col border-r border-gray-200 bg-white rounded-xl shadow-sm">
      <div className="bg-blue-600 text-white h-12 px-4 flex items-center gap-3 shadow-sm rounded-t-xl">
        <Bot className="h-5 w-5" />
        <h1 className="font-semibold text-sm sm:text-base lg:text-lg">Agent View</h1>
        <div className="ml-auto flex items-center gap-3">
          <label className="text-xs font-light tracking-wide opacity-90 flex items-center gap-1">
            <input
              type="checkbox"
              checked={colorEnabled}
              onChange={(e) => onToggleColors(e.target.checked)}
            />
            Colors
          </label>
          <span className="text-xs font-light tracking-wide opacity-80">Singtel (Demo)</span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-6 bg-gray-50/50">
        <AgentsList agents={agents} currentAgent={currentAgent} colorEnabled={colorEnabled} />
        <Guardrails
          guardrails={guardrails}
          inputGuardrails={activeAgent?.input_guardrails ?? []}
        />
        <ConversationContext context={context} />
        <HumanPanel isActive={isHuman} isSending={isHumanSending} failedReason={humanFailedReason} onSend={onHumanReply} onReturnToAI={onReturnToAI} />
        <RunnerOutput runnerEvents={runnerEvents} />
      </div>
    </div>
  );
}