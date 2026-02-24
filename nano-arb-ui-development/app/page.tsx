"use client"

import { useState } from "react"
import { useEngineConnection } from "@/hooks/use-engine-connection"
import { DashboardShell } from "@/components/dashboard-shell"
import { TradingDashboard } from "@/components/trading-dashboard"
import { BacktestDashboard } from "@/components/backtest-dashboard"
import { AboutPage } from "@/components/about-page"
import { WelcomeModal } from "@/components/welcome-modal"

export default function Home() {
  const [currentView, setCurrentView] = useState<
    "trading" | "backtest" | "about"
  >("trading")
  const simulation = useEngineConnection()

  return (
    <>
      <WelcomeModal />
      <DashboardShell
        currentView={currentView}
        onViewChange={setCurrentView}
        isRunning={simulation.isRunning}
        onToggleRunning={simulation.toggleRunning}
        onRestart={currentView === "trading" ? simulation.restart : undefined}
        clock={simulation.clock}
        currentPrice={simulation.currentPrice}
      >
        {currentView === "trading" && <TradingDashboard state={simulation} />}
        {currentView === "backtest" && <BacktestDashboard />}
        {currentView === "about" && <AboutPage />}
      </DashboardShell>
    </>
  )
}
