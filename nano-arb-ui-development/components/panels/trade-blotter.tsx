"use client"

import { useRef, useEffect } from "react"
import type { Trade } from "@/lib/mock-data"
import { ScrollArea } from "@/components/ui/scroll-area"

interface TradeBlotterProps {
  trades: Trade[]
}

export function TradeBlotter({ trades }: TradeBlotterProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to latest trade
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [trades.length])

  const formatTime = (clock: number) => {
    const secs = Math.floor((clock * 150) / 1000)
    const mins = Math.floor(secs / 60)
    return `${mins.toString().padStart(2, "0")}:${(secs % 60)
      .toString()
      .padStart(2, "0")}.${((clock * 150) % 1000).toString().padStart(3, "0")}`
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between pb-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Trade Blotter
        </h3>
        <span className="font-mono text-[10px] text-muted-foreground">
          {trades.length} fills
        </span>
      </div>

      {/* Header row */}
      <div className="grid grid-cols-7 gap-2 border-b border-border pb-1 text-[10px] uppercase tracking-wider text-muted-foreground">
        <span>Time</span>
        <span>Side</span>
        <span className="text-right">Price</span>
        <span className="text-right">Qty</span>
        <span className="text-right">P&L</span>
        <span className="text-right">Latency</span>
        <span className="text-center">Signal</span>
      </div>

      <ScrollArea className="flex-1">
        <div className="flex flex-col">
          {trades.map((trade) => (
            <div
              key={trade.id}
              className="grid grid-cols-7 items-center gap-2 border-b border-border/50 py-[3px] font-mono text-xs transition-colors hover:bg-secondary/30"
            >
              <span className="text-muted-foreground">
                {formatTime(trade.time)}
              </span>
              <span
                className={
                  trade.side === "BUY" ? "text-green-400" : "text-red-400"
                }
              >
                {trade.side}
              </span>
              <span className="text-right text-foreground">
                {trade.price.toFixed(2)}
              </span>
              <span className="text-right text-foreground">{trade.qty}</span>
              <span
                className={`text-right font-medium ${trade.pnl >= 0 ? "text-green-400" : "text-red-400"}`}
              >
                {trade.pnl >= 0 ? "+" : ""}
                {trade.pnl.toFixed(2)}
              </span>
              <span
                className={`text-right ${
                  trade.latencyUs < 1
                    ? "text-green-400"
                    : trade.latencyUs < 2
                      ? "text-yellow-400"
                      : "text-red-400"
                }`}
              >
                {trade.latencyUs.toFixed(2)}us
              </span>
              <span className="text-center">
                <span
                  className={`inline-block rounded px-1.5 py-0.5 text-[9px] font-medium ${
                    trade.signalSource === "ML"
                      ? "bg-primary/15 text-primary"
                      : trade.signalSource === "Skew"
                        ? "bg-yellow-500/15 text-yellow-400"
                        : "bg-secondary text-muted-foreground"
                  }`}
                >
                  {trade.signalSource}
                </span>
              </span>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
      </ScrollArea>
    </div>
  )
}
