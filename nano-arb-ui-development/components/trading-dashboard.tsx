"use client"

import type { SimulationState } from "@/hooks/use-engine-connection"
import { OrderBookPanel } from "@/components/panels/order-book"
import { PriceChartPanel } from "@/components/panels/price-chart"
import { PnlPanel } from "@/components/panels/pnl-panel"
import { LatencyPanel } from "@/components/panels/latency-panel"
import { RiskPanel } from "@/components/panels/risk-panel"
import { TradeBlotter } from "@/components/panels/trade-blotter"

interface TradingDashboardProps {
  state: SimulationState
}

export function TradingDashboard({ state }: TradingDashboardProps) {
  return (
    <div className="grid h-full grid-cols-12 grid-rows-[1fr_1fr_auto] gap-px bg-border">
      {/* Row 1: Order Book | Price Chart | Risk Panel */}
      <div className="col-span-3 bg-background p-2">
        <OrderBookPanel orderBook={state.orderBook} />
      </div>
      <div className="col-span-6 bg-background p-2">
        <PriceChartPanel
          priceTicks={state.priceTicks}
          currentPrice={state.currentPrice}
        />
      </div>
      <div className="col-span-3 bg-background p-2">
        <RiskPanel riskState={state.riskState} />
      </div>

      {/* Row 2: Trade Blotter | P&L Panel */}
      <div className="col-span-5 bg-background p-2">
        <TradeBlotter trades={state.trades} />
      </div>
      <div className="col-span-7 bg-background p-2">
        <PnlPanel pnlCurve={state.pnlCurve} metrics={state.metrics} />
      </div>

      {/* Row 3: Latency Monitor (full width) */}
      <div className="col-span-12 bg-background p-2">
        <LatencyPanel latencySamples={state.latencySamples} />
      </div>
    </div>
  )
}
