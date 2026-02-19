"use client"

import { useMemo } from "react"
import type { LatencySample } from "@/lib/mock-data"
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
} from "recharts"

interface LatencyPanelProps {
  latencySamples: LatencySample[]
}

const TEAL = "#2dd4bf"
const GREEN = "#22c55e"
const YELLOW = "#eab308"
const RED = "#ef4444"
const GRID_COLOR = "#1e293b"
const MUTED = "#64748b"

export function LatencyPanel({ latencySamples }: LatencyPanelProps) {
  // Calculate percentiles
  const stats = useMemo(() => {
    if (latencySamples.length === 0)
      return { p50: 0, p95: 0, p99: 0, min: 0, max: 0 }

    const sorted = [...latencySamples]
      .map((s) => s.totalUs)
      .sort((a, b) => a - b)
    const len = sorted.length

    return {
      p50: sorted[Math.floor(len * 0.5)] ?? 0,
      p95: sorted[Math.floor(len * 0.95)] ?? 0,
      p99: sorted[Math.floor(len * 0.99)] ?? 0,
      min: sorted[0] ?? 0,
      max: sorted[len - 1] ?? 0,
    }
  }, [latencySamples])

  // Histogram buckets
  const histogram = useMemo(() => {
    const buckets = [
      { label: "0-0.5", min: 0, max: 0.5, count: 0 },
      { label: "0.5-1.0", min: 0.5, max: 1.0, count: 0 },
      { label: "1.0-1.5", min: 1.0, max: 1.5, count: 0 },
      { label: "1.5-2.0", min: 1.5, max: 2.0, count: 0 },
      { label: "2.0-2.5", min: 2.0, max: 2.5, count: 0 },
      { label: "2.5+", min: 2.5, max: Infinity, count: 0 },
    ]

    for (const sample of latencySamples) {
      for (const bucket of buckets) {
        if (sample.totalUs >= bucket.min && sample.totalUs < bucket.max) {
          bucket.count++
          break
        }
      }
    }
    return buckets
  }, [latencySamples])

  // Component breakdown (average)
  const breakdown = useMemo(() => {
    if (latencySamples.length === 0) return []
    const len = latencySamples.length
    const avg = (key: keyof LatencySample) =>
      latencySamples.reduce((s, l) => s + (l[key] as number), 0) / len

    return [
      { name: "Market Data", value: Math.round(avg("marketDataUs") * 100) / 100, color: TEAL },
      { name: "ML Inference", value: Math.round(avg("mlInferenceUs") * 100) / 100, color: "#a78bfa" },
      { name: "Quote Calc", value: Math.round(avg("quoteCalcUs") * 100) / 100, color: YELLOW },
      { name: "Order Submit", value: Math.round(avg("orderSubmitUs") * 100) / 100, color: "#f97316" },
    ]
  }, [latencySamples])

  // Time series for line chart (last 60)
  const timeSeries = useMemo(
    () =>
      latencySamples.slice(-60).map((s, i) => ({
        idx: i,
        latency: s.totalUs,
      })),
    [latencySamples]
  )

  const latencyColor = (val: number) => {
    if (val < 1) return GREEN
    if (val < 2) return YELLOW
    return RED
  }

  return (
    <div className="flex h-full flex-col" style={{ minHeight: 160 }}>
      <div className="flex items-center justify-between pb-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Latency Monitor
        </h3>
        <div className="flex items-center gap-4">
          {[
            { label: "p50", value: stats.p50 },
            { label: "p95", value: stats.p95 },
            { label: "p99", value: stats.p99 },
            { label: "min", value: stats.min },
            { label: "max", value: stats.max },
          ].map((s) => (
            <div key={s.label} className="flex items-center gap-1">
              <span className="text-[10px] uppercase text-muted-foreground">
                {s.label}
              </span>
              <span
                className="font-mono text-xs font-semibold"
                style={{ color: latencyColor(s.value) }}
              >
                {s.value.toFixed(2)}us
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="grid flex-1 grid-cols-3 gap-3 min-h-0">
        {/* Histogram */}
        <div className="min-h-0">
          <p className="pb-1 text-[10px] uppercase tracking-wider text-muted-foreground">
            Distribution
          </p>
          <ResponsiveContainer width="100%" height="90%">
            <BarChart data={histogram} margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
              <XAxis
                dataKey="label"
                tick={{ fill: MUTED, fontSize: 9 }}
                axisLine={{ stroke: GRID_COLOR }}
                tickLine={false}
              />
              <YAxis hide />
              <Bar
                dataKey="count"
                fill={TEAL}
                radius={[2, 2, 0, 0]}
                opacity={0.8}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Time series */}
        <div className="min-h-0">
          <p className="pb-1 text-[10px] uppercase tracking-wider text-muted-foreground">
            Tick-to-Trade Latency
          </p>
          <ResponsiveContainer width="100%" height="90%">
            <LineChart data={timeSeries} margin={{ top: 0, right: 5, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GRID_COLOR} vertical={false} />
              <XAxis dataKey="idx" tick={false} axisLine={{ stroke: GRID_COLOR }} />
              <YAxis
                tick={{ fill: MUTED, fontSize: 9, fontFamily: "var(--font-jetbrains-mono)" }}
                axisLine={false}
                tickLine={false}
                width={30}
                tickFormatter={(v: number) => `${v.toFixed(1)}`}
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
                formatter={(value: number) => [`${value.toFixed(2)}us`, "Latency"]}
              />
              <Line
                type="monotone"
                dataKey="latency"
                stroke={TEAL}
                strokeWidth={1.5}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Component breakdown */}
        <div className="flex flex-col gap-2">
          <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
            Component Breakdown
          </p>
          {breakdown.map((comp) => (
            <div key={comp.name} className="flex flex-col gap-0.5">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-muted-foreground">
                  {comp.name}
                </span>
                <span className="font-mono text-[10px] font-medium text-foreground">
                  {comp.value.toFixed(2)}us
                </span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-secondary">
                <div
                  className="h-full rounded-full transition-all duration-300"
                  style={{
                    width: `${Math.min((comp.value / 1.0) * 100, 100)}%`,
                    backgroundColor: comp.color,
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
