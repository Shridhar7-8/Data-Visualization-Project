import type React from "react"
import { Tooltip, TooltipContent, TooltipProvider } from "@/components/ui/tooltip"

export function Chart({ children }: { children: React.ReactNode }) {
  return <TooltipProvider>{children}</TooltipProvider>
}

export function ChartContainer({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={className}>{children}</div>
}

export function ChartTooltip({ children }: { children: React.ReactNode }) {
  return <Tooltip>{children}</Tooltip>
}

export function ChartTooltipContent({
  children,
  formatter,
}: { children?: React.ReactNode; formatter?: (value: number) => [string, string] }) {
  return <TooltipContent>{children}</TooltipContent>
}
