"use client";

import { PanelSection } from "./panel-section";
import { Card, CardContent } from "@/components/ui/card";
import { BookText } from "lucide-react";

interface ConversationContextProps {
  context: Record<string, any>;
}

export function ConversationContext({ context }: ConversationContextProps) {
  return (
    <PanelSection
      title="Conversation Context"
      icon={<BookText className="h-4 w-4 text-blue-600" />}
    >
      <Card className="bg-gradient-to-r from-white to-gray-50 border-gray-200 shadow-sm">
        <CardContent className="p-3">
          <div className="grid grid-cols-2 gap-2">
            {Object.entries(context).map(([key, value]) => {
              const isNullish = value === null || value === undefined;
              const displayValue = isNullish
                ? "null"
                : typeof value === "boolean"
                ? value
                  ? "Yes"
                  : "No"
                : String(value);
              return (
                <div
                  key={key}
                  className="flex items-center gap-2 bg-white p-2 rounded-md border border-gray-200 shadow-sm transition-all"
                >
                  <div className="w-2 h-2 rounded-full bg-blue-500"></div>
                  <div className="text-xs">
                    <span className="text-zinc-500 font-light">{key}:</span>{" "}
                    <span
                      className={
                        !isNullish
                          ? "text-zinc-900 font-light"
                          : "text-gray-400 italic"
                      }
                    >
                      {displayValue}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>
    </PanelSection>
  );
}