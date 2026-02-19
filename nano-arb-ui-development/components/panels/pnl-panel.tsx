"use client"

import { useMemo } from "react"
import type { PnlPoint, PerformanceMetrics } from "@/lib/mock-data"
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
} from "recharts"

interface PnlPanelProps {
  pnlCurve: PnlPoint[]
  metrics: PerformanceMetrics
}

const TEAL = "#2dd4bf"
const GREEN = "#22c55e"
const RED = "#ef4444"
const GRID_COLOR = "#1e293b"
const MUTED = "#64748b"

export function PnlPanel({ pnlCurve, metrics }: PnlPanelProps) {
  const chartData = useMemo(
    () => pnlCurve.map((p, i) => ({ idx: i, pnl: p.pnl })),
    [pnlCurve]
  )

  const pnlColor = metrics.totalPnl >= 0 ? GREEN : RED

  const statCards = [
    {
      label: "Total P&L",
      value: `$${metrics.totalPnl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
      color: metrics.totalPnl >= 0 ? "text-green-400" : "text-red-400",
    },
    {
      label: "Sharpe",
      value: metrics.sharpeRatio.toFixed(2),
      color: metrics.sharpeRatio > 1 ? "text-green-400" : "text-foreground",
    },
    {
      label: "Win Rate",
      value: `${metrics.winRate.toFixed(1)}%`,
      color: metrics.winRate > 50 ? "text-green-400" : "text-red-400",
    },
    {
      label: "Trades",
      value: metrics.totalTrades.toLocaleString(),
      color: "text-foreground",
    },
    {
      label: "Max DD",
      value: `$${metrics.maxDrawdown.toFixed(2)}`,
      color: "text-red-400",
    },
    {
      label: "Fill Rate",
      value: `${metrics.fillRate}%`,
      color: "text-primary",
    },
  ]

  return (
    <div className="flex h-full flex-col">
      <h3 className="pb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        P&L & Performance
      </h3>

      {/* Stat cards grid */}
      <div className="grid grid-cols-6 gap-2 pb-2">
        {statCards.map((card) => (
          <div
            key={card.label}
            className="rounded border border-border bg-secondary/50 px-2 py-1.5"
          >
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
              {card.label}
            </p>
            <p className={`font-mono text-sm font-semibold ${card.color}`}>
              {card.value}
            </p>
          </div>
        ))}
      </div>

      {/* P&L curve chart */}
      <div className="flex-1 min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={chartData}
            margin={{ top: 5, right: 10, bottom: 0, left: 0 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke={GRID_COLOR}
              vertical={false}
            />
            <XAxis dataKey="idx" tick={false} axisLine={{ stroke: GRID_COLOR }} />
            <YAxis
              tick={{
                fill: MUTED,
                fontSize: 10,
                fontFamily: "var(--font-jetbrains-mono)",
              }}
              axisLine={false}
              tickLine={false}
              width={50}
              tickFormatter={(v: number) => `$${v.toFixed(0)}`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#0f172a",
                border: "1px solid #1e293b",
                borderRadius: 6,
                fontSize: 11,
                fontFamily: "var(--font-jetbrains-mono)",
              }}
              labelFormatter={() => ""}
              formatter={(value: number) => [
                `$${value.toFixed(2)}`,
                "P&L",
              ]}
            />
            <Area
              type="monotone"
              dataKey="pnl"
              stroke={pnlColor}
              strokeWidth={1.5}
              fill={pnlColor}
              fillOpacity={0.1}
              dot={false}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
