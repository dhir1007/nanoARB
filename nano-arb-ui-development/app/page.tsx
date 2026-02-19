"use client"

import { useState } from "react"
import { useEngineConnection } from "@/hooks/use-engine-connection"
import { DashboardShell } from "@/components/dashboard-shell"
import { TradingDashboard } from "@/components/trading-dashboard"
import { BacktestDashboard } from "@/components/backtest-dashboard"

export default function Home() {
  const [currentView, setCurrentView] = useState<"trading" | "backtest">(
    "trading"
  )
  const simulation = useEngineConnection()

  return (
    <DashboardShell
      currentView={currentView}
      onViewChange={setCurrentView}
      isRunning={simulation.isRunning}
      onToggleRunning={simulation.toggleRunning}
      clock={simulation.clock}
      currentPrice={simulation.currentPrice}
    >
      {currentView === "trading" ? (
        <TradingDashboard state={simulation} />
      ) : (
        <BacktestDashboard />
      )}
    </DashboardShell>
  )
}
