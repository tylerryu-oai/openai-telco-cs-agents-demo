"use client";

import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Bot } from "lucide-react";
import { PanelSection } from "./panel-section";
import type { Agent } from "@/lib/types";
import { agentColorClasses, cn } from "@/lib/utils";

interface AgentsListProps {
  agents: Agent[];
  currentAgent: string;
  colorEnabled?: boolean;
}

export function AgentsList({ agents, currentAgent, colorEnabled = true }: AgentsListProps) {
  const activeAgent = agents.find((a) => a.name === currentAgent);
  return (
    <PanelSection
      title="Available Agents"
      icon={<Bot className="h-4 w-4 text-blue-600" />}
    >
      <div className="grid grid-cols-3 gap-3">
        {agents.map((agent) => {
          const colors = colorEnabled ? agentColorClasses(agent.name) : { bubbleBg: "bg-zinc-100", dotBg: "bg-zinc-300", ring: "ring-blue-500" };
          const isActive = agent.name === currentAgent;
          const isReachable = isActive || activeAgent?.handoffs.includes(agent.name);
          return (
            <Card
              key={agent.name}
              className={cn(
                "bg-white border-gray-200 transition-all",
                !isReachable && "opacity-50 filter grayscale cursor-not-allowed pointer-events-none",
                isActive && `ring-1 ${colors.ring} shadow-md`
              )}
            >
              <CardHeader className="p-3 pb-1">
                <CardTitle className="text-sm flex items-center text-zinc-900 gap-2">
                  <span className={cn("w-2.5 h-2.5 rounded-full", colors.dotBg)} />
                  {agent.name}
                </CardTitle>
              </CardHeader>
              <CardContent className="p-3 pt-1">
                <p className="text-xs font-light text-zinc-500">
                  {agent.description}
                </p>
                {isActive && (
                  <Badge className={cn("mt-2 text-white", colors.dotBg)}>
                    Active
                  </Badge>
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>
    </PanelSection>
  );
}