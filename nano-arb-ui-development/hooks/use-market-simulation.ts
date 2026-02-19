"use client"

import { useState, useEffect, useCallback, useRef } from "react"
import {
  type OrderBook,
  type PriceTick,
  type Trade,
  type LatencySample,
  type RiskState,
  type PerformanceMetrics,
  type PnlPoint,
  type RiskAlert,
  generateOrderBook,
  generatePriceTick,
  generateTrade,
  generateLatencySample,
  generateInitialRiskState,
  getInitialPrice,
} from "@/lib/mock-data"

export interface SimulationState {
  isRunning: boolean
  clock: number
  currentPrice: number
  orderBook: OrderBook
  priceTicks: PriceTick[]
  trades: Trade[]
  latencySamples: LatencySample[]
  riskState: RiskState
  metrics: PerformanceMetrics
  pnlCurve: PnlPoint[]
}

let alertIdCounter = 0

export function useMarketSimulation() {
  const [isRunning, setIsRunning] = useState(true)
  const priceRef = useRef(getInitialPrice())
  const clockRef = useRef(0)
  const cumulativePnlRef = useRef(0)
  const tradeCountRef = useRef(0)
  const winCountRef = useRef(0)

  const [state, setState] = useState<SimulationState>(() => {
    const initialPrice = priceRef.current
    return {
      isRunning: true,
      clock: 0,
      currentPrice: initialPrice,
      orderBook: generateOrderBook(initialPrice),
      priceTicks: [],
      trades: [],
      latencySamples: [],
      riskState: generateInitialRiskState(),
      metrics: {
        totalPnl: 0,
        sharpeRatio: 0,
        winRate: 0,
        totalTrades: 0,
        maxDrawdown: 0,
        fillRate: 96.2,
        avgTradeUs: 1.45,
      },
      pnlCurve: [],
    }
  })

  const toggleRunning = useCallback(() => {
    setIsRunning((prev) => !prev)
  }, [])

  // Main simulation tick - runs every 150ms for real-time feel
  useEffect(() => {
    if (!isRunning) return

    const interval = setInterval(() => {
      clockRef.current += 1
      const clock = clockRef.current

      // Generate new price tick
      const newTick = generatePriceTick(priceRef.current, clock)
      priceRef.current = newTick.price

      // Generate new latency sample
      const newLatency = generateLatencySample(clock)

      // Maybe generate a trade (roughly every 3-5 ticks)
      let newTrade: Trade | null = null
      if (Math.random() < 0.3) {
        newTrade = generateTrade(priceRef.current, clock)
        tradeCountRef.current += 1
        cumulativePnlRef.current += newTrade.pnl
        if (newTrade.pnl > 0) winCountRef.current += 1
      }

      setState((prev) => {
        // Update price ticks (keep last 200)
        const priceTicks = [...prev.priceTicks, newTick].slice(-200)

        // Update trades (keep last 100)
        const trades = newTrade
          ? [...prev.trades, newTrade].slice(-100)
          : prev.trades

        // Update latency samples (keep last 100)
        const latencySamples = [...prev.latencySamples, newLatency].slice(-100)

        // Update order book
        const orderBook = generateOrderBook(priceRef.current)

        // Update P&L curve
        const pnlCurve = [
          ...prev.pnlCurve,
          { time: clock, pnl: cumulativePnlRef.current },
        ].slice(-200)

        // Calculate metrics
        const totalTrades = tradeCountRef.current
        const winRate =
          totalTrades > 0
            ? Math.round((winCountRef.current / totalTrades) * 10000) / 100
            : 0
        const avgLatency =
          latencySamples.length > 0
            ? latencySamples.reduce((s, l) => s + l.totalUs, 0) /
              latencySamples.length
            : 0

        // Calculate running Sharpe (simplified)
        const returns = pnlCurve.slice(-50).map((p, i, arr) =>
          i === 0 ? 0 : p.pnl - arr[i - 1].pnl
        )
        const meanReturn =
          returns.length > 1
            ? returns.slice(1).reduce((s, r) => s + r, 0) / (returns.length - 1)
            : 0
        const stdReturn =
          returns.length > 2
            ? Math.sqrt(
                returns
                  .slice(1)
                  .reduce((s, r) => s + (r - meanReturn) ** 2, 0) /
                  (returns.length - 2)
              )
            : 1
        const sharpe =
          stdReturn > 0
            ? Math.round((meanReturn / stdReturn) * Math.sqrt(252) * 100) / 100
            : 0

        // Max drawdown from P&L curve
        let peak = -Infinity
        let maxDd = 0
        for (const p of pnlCurve) {
          peak = Math.max(peak, p.pnl)
          maxDd = Math.max(maxDd, peak - p.pnl)
        }

        // Update risk state
        const positionDelta = newTrade
          ? (newTrade.side === "BUY" ? newTrade.qty : -newTrade.qty)
          : 0
        const newPosition = prev.riskState.positionSize + positionDelta
        const skew = newPosition / prev.riskState.positionLimit
        const ddPct = maxDd / 10000 // Rough pct

        // Generate risk alerts occasionally
        const newAlerts = [...prev.riskState.alerts]
        if (Math.abs(newPosition) > prev.riskState.positionLimit * 0.8 && Math.random() < 0.1) {
          alertIdCounter++
          newAlerts.push({
            id: `A-${alertIdCounter}`,
            time: clock,
            level: "warning",
            message: `Position nearing limit: ${newPosition}/${prev.riskState.positionLimit}`,
          })
        }
        if (ddPct > 3 && Math.random() < 0.05) {
          alertIdCounter++
          newAlerts.push({
            id: `A-${alertIdCounter}`,
            time: clock,
            level: "critical",
            message: `Drawdown elevated: $${maxDd.toFixed(0)}`,
          })
        }

        const riskState: RiskState = {
          ...prev.riskState,
          positionSize: newPosition,
          currentDrawdown: Math.round(ddPct * 100) / 100,
          inventorySkew: Math.round(skew * 1000) / 1000,
          longExposure: prev.riskState.longExposure + (positionDelta > 0 ? positionDelta * priceRef.current * 50 : 0),
          shortExposure: prev.riskState.shortExposure + (positionDelta < 0 ? Math.abs(positionDelta) * priceRef.current * 50 : 0),
          netExposure: newPosition * priceRef.current * 50,
          alerts: newAlerts.slice(-20),
        }

        const metrics: PerformanceMetrics = {
          totalPnl: Math.round(cumulativePnlRef.current * 100) / 100,
          sharpeRatio: sharpe,
          winRate,
          totalTrades,
          maxDrawdown: Math.round(maxDd * 100) / 100,
          fillRate: Math.round((96 + Math.random() * 3) * 10) / 10,
          avgTradeUs: Math.round(avgLatency * 100) / 100,
        }

        return {
          isRunning: true,
          clock,
          currentPrice: priceRef.current,
          orderBook,
          priceTicks,
          trades,
          latencySamples,
          riskState,
          metrics,
          pnlCurve,
        }
      })
    }, 150)

    return () => clearInterval(interval)
  }, [isRunning])

  return { ...state, isRunning, toggleRunning }
}
