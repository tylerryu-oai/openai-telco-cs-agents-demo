import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// Map agent names to consistent UI color classes
export function agentColorClasses(agentName?: string): {
  bubbleBg: string;
  dotBg: string;
  ring: string;
} {
  const name = (agentName || "").toLowerCase();
  // Define palette per agent
  const palette: Record<string, { bubbleBg: string; dotBg: string; ring: string }> = {
    "triage agent": { bubbleBg: "bg-indigo-100", dotBg: "bg-indigo-500", ring: "ring-indigo-400" },
    "faq agent": { bubbleBg: "bg-teal-100", dotBg: "bg-teal-500", ring: "ring-teal-400" },
    "plan change agent": { bubbleBg: "bg-violet-100", dotBg: "bg-violet-500", ring: "ring-violet-400" },
    "billing agent": { bubbleBg: "bg-amber-100", dotBg: "bg-amber-500", ring: "ring-amber-400" },
    "technical support agent": { bubbleBg: "bg-orange-100", dotBg: "bg-orange-500", ring: "ring-orange-400" },
    "data usage agent": { bubbleBg: "bg-cyan-100", dotBg: "bg-cyan-500", ring: "ring-cyan-400" },
    "roaming agent": { bubbleBg: "bg-fuchsia-100", dotBg: "bg-fuchsia-500", ring: "ring-fuchsia-400" },
    "human support": { bubbleBg: "bg-rose-100", dotBg: "bg-rose-500", ring: "ring-rose-400" },
  };
  return (
    palette[name] || { bubbleBg: "bg-zinc-100", dotBg: "bg-zinc-400", ring: "ring-zinc-300" }
  );
}
