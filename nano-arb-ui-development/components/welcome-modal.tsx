"use client"

import { useState, useEffect } from "react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Zap, Activity, Brain, Shield, Timer } from "lucide-react"

const STORAGE_KEY = "nanoarb-welcome-seen"

export function WelcomeModal() {
  const [open, setOpen] = useState(false)

  useEffect(() => {
    if (typeof window !== "undefined" && !localStorage.getItem(STORAGE_KEY)) {
      setOpen(true)
    }
  }, [])

  const handleDismiss = () => {
    localStorage.setItem(STORAGE_KEY, "true")
    setOpen(false)
  }

  return (
    <Dialog open={open} onOpenChange={(v) => { if (!v) handleDismiss() }}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary text-primary-foreground">
              <Zap className="h-5 w-5" />
            </div>
            <div>
              <DialogTitle className="text-xl">Welcome to NanoARB</DialogTitle>
              <DialogDescription>
                Nanosecond-Level HFT Market-Making Engine
              </DialogDescription>
            </div>
          </div>
        </DialogHeader>

        <div className="space-y-4 py-2">
          <p className="text-sm text-muted-foreground">
            You're looking at a <strong className="text-foreground">production-grade trading engine</strong> built
            entirely in Rust, performing automated market-making on CME futures
            with sub-microsecond latency.
          </p>

          <div className="grid grid-cols-2 gap-3">
            {[
              {
                icon: Activity,
                title: "Live Simulation",
                desc: "Real-time order book, trades, and P&L streamed from an AWS server",
              },
              {
                icon: Brain,
                title: "ML-Driven",
                desc: "Mamba State-Space Model generates buy/sell signals at 580ns inference",
              },
              {
                icon: Timer,
                title: "Sub-Microsecond",
                desc: "780ns median tick-to-trade latency, measured end-to-end",
              },
              {
                icon: Shield,
                title: "Risk Controls",
                desc: "Position limits, drawdown kill-switch, and inventory management",
              },
            ].map((item) => (
              <div
                key={item.title}
                className="rounded-lg border border-border p-3"
              >
                <div className="flex items-center gap-2">
                  <item.icon className="h-4 w-4 text-primary" />
                  <span className="text-sm font-medium text-foreground">
                    {item.title}
                  </span>
                </div>
                <p className="mt-1 text-xs text-muted-foreground">{item.desc}</p>
              </div>
            ))}
          </div>

          <p className="text-xs text-muted-foreground">
            Click <strong className="text-foreground">Resume</strong> in the top bar to start streaming
            live data, or visit the <strong className="text-foreground">About</strong> tab for a full technical deep-dive.
          </p>
        </div>

        <DialogFooter>
          <Button onClick={handleDismiss} className="w-full">
            Explore Dashboard
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
