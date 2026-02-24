"use client"

import { useState } from "react"
import {
  Activity,
  BarChart3,
  ChevronLeft,
  ChevronRight,
  FlaskConical,
  Info,
  RotateCcw,
  Zap,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

interface DashboardShellProps {
  currentView: "trading" | "backtest" | "about"
  onViewChange: (view: "trading" | "backtest" | "about") => void
  isRunning: boolean
  onToggleRunning: () => void
  onRestart?: () => void
  clock: number
  currentPrice: number
  children: React.ReactNode
}

export function DashboardShell({
  currentView,
  onViewChange,
  isRunning,
  onToggleRunning,
  onRestart,
  clock,
  currentPrice,
  children,
}: DashboardShellProps) {
  const [collapsed, setCollapsed] = useState(false)

  const formatClock = (ticks: number) => {
    const secs = Math.floor((ticks * 150) / 1000)
    const mins = Math.floor(secs / 60)
    const hrs = Math.floor(mins / 60)
    return `${hrs.toString().padStart(2, "0")}:${(mins % 60)
      .toString()
      .padStart(2, "0")}:${(secs % 60).toString().padStart(2, "0")}`
  }

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background text-foreground">
      {/* Sidebar */}
      <aside
        className={cn(
          "flex flex-col border-r border-border bg-secondary/50 transition-all duration-200",
          collapsed ? "w-14" : "w-48"
        )}
      >
        {/* Logo */}
        <div className="flex h-12 items-center gap-2 border-b border-border px-3">
          <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded bg-primary text-primary-foreground">
            <Zap className="h-4 w-4" />
          </div>
          {!collapsed && (
            <span className="font-mono text-sm font-bold tracking-tight text-foreground">
              NanoARB
            </span>
          )}
        </div>

        {/* Nav items */}
        <nav className="flex flex-1 flex-col gap-1 p-2">
          <button
            onClick={() => onViewChange("trading")}
            className={cn(
              "flex items-center gap-2 rounded-md px-2.5 py-2 text-sm transition-colors",
              currentView === "trading"
                ? "bg-primary/15 text-primary"
                : "text-muted-foreground hover:bg-secondary hover:text-foreground"
            )}
          >
            <Activity className="h-4 w-4 shrink-0" />
            {!collapsed && <span>Live Trading</span>}
          </button>
          <button
            onClick={() => onViewChange("backtest")}
            className={cn(
              "flex items-center gap-2 rounded-md px-2.5 py-2 text-sm transition-colors",
              currentView === "backtest"
                ? "bg-primary/15 text-primary"
                : "text-muted-foreground hover:bg-secondary hover:text-foreground"
            )}
          >
            <FlaskConical className="h-4 w-4 shrink-0" />
            {!collapsed && <span>Backtester</span>}
          </button>
          <button
            onClick={() => onViewChange("trading")}
            className={cn(
              "flex items-center gap-2 rounded-md px-2.5 py-2 text-sm transition-colors",
              "text-muted-foreground hover:bg-secondary hover:text-foreground"
            )}
          >
            <BarChart3 className="h-4 w-4 shrink-0" />
            {!collapsed && <span>Analytics</span>}
          </button>
          <button
            onClick={() => onViewChange("about")}
            className={cn(
              "flex items-center gap-2 rounded-md px-2.5 py-2 text-sm transition-colors",
              currentView === "about"
                ? "bg-primary/15 text-primary"
                : "text-muted-foreground hover:bg-secondary hover:text-foreground"
            )}
          >
            <Info className="h-4 w-4 shrink-0" />
            {!collapsed && <span>About</span>}
          </button>
        </nav>

        {/* Collapse toggle */}
        <div className="border-t border-border p-2">
          <Button
            variant="ghost"
            size="sm"
            className="w-full justify-center text-muted-foreground"
            onClick={() => setCollapsed(!collapsed)}
          >
            {collapsed ? (
              <ChevronRight className="h-4 w-4" />
            ) : (
              <ChevronLeft className="h-4 w-4" />
            )}
          </Button>
        </div>
      </aside>

      {/* Main area */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Top bar */}
        <header className="flex h-12 items-center justify-between border-b border-border bg-secondary/30 px-4">
          <div className="flex items-center gap-4">
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex cursor-help items-center gap-2">
                  <span className="text-xs text-muted-foreground">INSTRUMENT</span>
                  <span className="font-mono text-sm font-semibold text-foreground">
                    ES CME Futures
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent>
                CME E-mini S&P 500 futures — the most liquid futures contract globally
              </TooltipContent>
            </Tooltip>
            <Separator orientation="vertical" className="h-5" />
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex cursor-help items-center gap-2">
                  <span className="text-xs text-muted-foreground">PRICE</span>
                  <span className="font-mono text-sm font-semibold text-primary">
                    {currentPrice.toFixed(2)}
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent>
                Real-time microprice derived from order book bid/ask imbalance
              </TooltipContent>
            </Tooltip>
            <Separator orientation="vertical" className="h-5" />
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex cursor-help items-center gap-2">
                  <span className="text-xs text-muted-foreground">SESSION</span>
                  <span className="font-mono text-xs text-foreground">
                    {formatClock(clock)}
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent>
                Elapsed simulation time since engine start
              </TooltipContent>
            </Tooltip>
          </div>

          <div className="flex items-center gap-3">
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex cursor-help items-center gap-2">
                  <div
                    className={cn(
                      "h-2 w-2 rounded-full",
                      isRunning ? "bg-green-500 animate-pulse" : "bg-red-500"
                    )}
                  />
                  <span className="text-xs text-muted-foreground">
                    {isRunning ? "LIVE" : "PAUSED"}
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent>
                {isRunning
                  ? "Engine is streaming live simulation data via SSE"
                  : "Click Resume to connect to the Rust engine"}
              </TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <Badge
                  variant="outline"
                  className="cursor-help border-primary/30 font-mono text-xs text-primary"
                >
                  Mamba SSM
                </Badge>
              </TooltipTrigger>
              <TooltipContent className="max-w-xs">
                Mamba State-Space Model — 10-50× faster than Transformers for sequence modeling. Predicts LOB price direction at 580ns inference latency.
              </TooltipContent>
            </Tooltip>
            <Button
              variant="outline"
              size="sm"
              onClick={onToggleRunning}
              className="h-7 text-xs"
            >
              {isRunning ? "Pause" : "Resume"}
            </Button>
            {onRestart && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={onRestart}
                    className="h-7 gap-1 text-xs"
                  >
                    <RotateCcw className="h-3 w-3" />
                    Restart
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  Reset simulation: clear trades, P&L, and start fresh from tick 0
                </TooltipContent>
              </Tooltip>
            )}
          </div>
        </header>

        {/* Content */}
        <main className="flex-1 overflow-auto">{children}</main>
      </div>
    </div>
  )
}
