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
    "triage": { bubbleBg: "bg-indigo-100", dotBg: "bg-indigo-500", ring: "ring-indigo-400" },
    "customer support": { bubbleBg: "bg-violet-100", dotBg: "bg-violet-500", ring: "ring-violet-400" },
    "technical support": { bubbleBg: "bg-orange-100", dotBg: "bg-orange-500", ring: "ring-orange-400" },
    "human": { bubbleBg: "bg-rose-100", dotBg: "bg-rose-500", ring: "ring-rose-400" },
  };
  return (
    palette[name] || { bubbleBg: "bg-zinc-100", dotBg: "bg-zinc-400", ring: "ring-zinc-300" }
  );
}
