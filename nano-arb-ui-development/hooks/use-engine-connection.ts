"use client"

import { useState, useEffect, useCallback, useRef } from "react"
import type {
  OrderBook,
  PriceTick,
  Trade,
  LatencySample,
  RiskState,
  PerformanceMetrics,
  PnlPoint,
} from "@/lib/mock-data"

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:9090"

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
  /** Accumulated P&L points for full-session view (cleared on restart) */
  fullPnlCurve: PnlPoint[]
  /** Active data source label */
  dataSource?: string
  /** Active strategy name */
  strategy?: string
  /** Whether running in historical replay mode */
  isReplay?: boolean
  /** Replay progress 0.0 to 1.0 */
  replayProgress?: number
}

const INITIAL_STATE: SimulationState = {
  isRunning: false,
  clock: 0,
  currentPrice: 5425.75,
  orderBook: { bids: [], asks: [], spread: 0, midPrice: 5425.75 },
  priceTicks: [],
  trades: [],
  latencySamples: [],
  riskState: {
    positionSize: 0,
    positionLimit: 50,
    currentDrawdown: 0,
    maxDrawdown: 5,
    killSwitchActive: true,
    killSwitchTripped: false,
    longExposure: 0,
    shortExposure: 0,
    netExposure: 0,
    inventorySkew: 0,
    alerts: [],
  },
  metrics: {
    totalPnl: 0,
    sharpeRatio: 0,
    winRate: 0,
    totalTrades: 0,
    maxDrawdown: 0,
    fillRate: 0,
    avgTradeUs: 0,
  },
  pnlCurve: [],
  fullPnlCurve: [],
}

export function useEngineConnection() {
  const [state, setState] = useState<SimulationState>(INITIAL_STATE)
  const [isRunning, setIsRunning] = useState(false)
  const esRef = useRef<EventSource | null>(null)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const backoffMs = useRef(1000)

  const connect = useCallback(() => {
    if (esRef.current) {
      esRef.current.close()
    }

    const es = new EventSource(`${API_BASE}/api/stream`)
    esRef.current = es

    es.addEventListener("state", (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data) as Omit<SimulationState, "fullPnlCurve">
        setState((prev) => {
          const seen = new Set(prev.fullPnlCurve.map((p) => p.time))
          const newPoints = (data.pnlCurve ?? []).filter((p) => !seen.has(p.time))
          const merged =
            data.clock < prev.clock || data.clock < 10
              ? [...newPoints]
              : [...prev.fullPnlCurve, ...newPoints]
          const fullPnlCurve = merged.sort((a, b) => a.time - b.time)
          return { ...data, fullPnlCurve }
        })
        setIsRunning(data.isRunning ?? true)
        backoffMs.current = 1000
      } catch {
        // ignore malformed events
      }
    })

    es.onopen = () => {
      setIsRunning(true)
      backoffMs.current = 1000
    }

    es.onerror = () => {
      es.close()
      esRef.current = null
      setIsRunning(false)

      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      reconnectTimer.current = setTimeout(() => {
        backoffMs.current = Math.min(backoffMs.current * 2, 30000)
        connect()
      }, backoffMs.current)
    }
  }, [])

  const disconnect = useCallback(() => {
    if (esRef.current) {
      esRef.current.close()
      esRef.current = null
    }
    if (reconnectTimer.current) {
      clearTimeout(reconnectTimer.current)
      reconnectTimer.current = null
    }
    setIsRunning(false)
  }, [])

  const toggleRunning = useCallback(() => {
    if (isRunning) {
      disconnect()
    } else {
      connect()
    }
  }, [isRunning, connect, disconnect])

  const restart = useCallback(async () => {
    try {
      await fetch(`${API_BASE}/api/restart`, { method: "POST" })
      setState((prev) => ({ ...prev, fullPnlCurve: [] }))
    } catch {
      // ignore
    }
  }, [])

  useEffect(() => {
    connect()
    return () => disconnect()
  }, [connect, disconnect])

  return { ...state, isRunning, toggleRunning, restart }
}
