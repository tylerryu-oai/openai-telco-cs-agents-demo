"use client";

import { useState, useCallback } from "react";
import { PanelSection } from "./panel-section";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { User } from "lucide-react";

interface HumanPanelProps {
  isActive: boolean;
  isSending?: boolean;
  failedReason?: string | null;
  onSend: (message: string) => Promise<boolean>;
  onReturnToAI: (draft?: string) => Promise<void> | void;
}

export function HumanPanel({ isActive, isSending, failedReason, onSend, onReturnToAI }: HumanPanelProps) {
  const [text, setText] = useState("");

  const handleSend = useCallback(async () => {
    const msg = text.trim();
    if (!msg) return;
    const ok = await onSend(msg);
    if (ok) setText("");
  }, [text, onSend]);

  return (
    <PanelSection title="Human Operator" icon={<User className="h-4 w-4 text-blue-600" />}> 
      <Card className="bg-white border-gray-200 shadow-sm">
        <CardContent className="p-3">
          <div className="flex items-center gap-2 mb-2">
            {failedReason ? (
              <>
                <Badge className="bg-red-600">Failed</Badge>
                <span className="text-xs text-red-700">{failedReason}</span>
              </>
            ) : (
              <>
                <Badge className={isActive ? "bg-emerald-600" : "bg-gray-400"}>
                  {isActive ? "Active" : "Inactive"}
                </Badge>
                <span className="text-xs text-zinc-600">
                  {isActive ? "You can reply on behalf of the agent." : "Hand off to Human Support to enable."}
                </span>
              </>
            )}
          </div>
          <div className="flex gap-2 items-end">
            <textarea
              rows={2}
              className="flex-1 text-sm border border-gray-200 rounded-md p-2 bg-white"
              placeholder="Type a human reply..."
              value={text}
              onChange={(e) => setText(e.target.value)}
              disabled={!isActive || !!isSending}
            />
            <button
              className="h-9 px-3 rounded-md bg-black text-white disabled:bg-gray-300 disabled:text-gray-500"
              disabled={!isActive || !!isSending || !text.trim()}
              onClick={handleSend}
            >
              Send
            </button>
            <button
              className="h-9 px-3 rounded-md bg-blue-600 text-white disabled:bg-gray-300 disabled:text-gray-500"
              disabled={!isActive || !!isSending}
              onClick={async () => {
                const draft = text.trim();
                await onReturnToAI(draft || undefined);
                if (draft) setText("");
              }}
            >
              Return to AI
            </button>
          </div>
        </CardContent>
      </Card>
    </PanelSection>
  );
}


