"use client"

import { useMemo } from "react"
import type { PriceTick } from "@/lib/mock-data"
import { PanelInfo, PANEL_INFO } from "@/components/panel-info"
import {
  ComposedChart,
  Area,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  ReferenceDot,
  Tooltip,
} from "recharts"

interface PriceChartPanelProps {
  priceTicks: PriceTick[]
  currentPrice: number
}

const TEAL = "#2dd4bf"
const GREEN = "#22c55e"
const RED = "#ef4444"
const GRID_COLOR = "#1e293b"
const MUTED = "#64748b"

export function PriceChartPanel({
  priceTicks,
  currentPrice,
}: PriceChartPanelProps) {
  const firstPrice = priceTicks.length > 0 ? priceTicks[0].price : currentPrice
  const change = currentPrice - firstPrice
  const changePct = firstPrice > 0 ? (change / firstPrice) * 100 : 0

  const chartData = useMemo(
    () =>
      priceTicks.map((tick, i) => ({
        idx: i,
        price: tick.price,
        volume: tick.volume,
        signal: tick.signal,
        prediction: tick.prediction,
      })),
    [priceTicks]
  )

  const buySignals = useMemo(
    () => chartData.filter((d) => d.signal === "buy"),
    [chartData]
  )
  const sellSignals = useMemo(
    () => chartData.filter((d) => d.signal === "sell"),
    [chartData]
  )

  const priceMin = useMemo(
    () =>
      chartData.length > 0
        ? Math.min(...chartData.map((d) => d.price)) - 2
        : currentPrice - 5,
    [chartData, currentPrice]
  )
  const priceMax = useMemo(
    () =>
      chartData.length > 0
        ? Math.max(...chartData.map((d) => d.price)) + 2
        : currentPrice + 5,
    [chartData, currentPrice]
  )

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center justify-between pb-2">
        <div className="flex items-center gap-3">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            ES Futures
          </h3>
          <PanelInfo {...PANEL_INFO.priceChart} />
          <span className="font-mono text-lg font-bold text-foreground">
            {currentPrice.toFixed(2)}
          </span>
          <span
            className={`font-mono text-sm font-medium ${change >= 0 ? "text-green-400" : "text-red-400"}`}
          >
            {change >= 0 ? "+" : ""}
            {change.toFixed(2)} ({changePct >= 0 ? "+" : ""}
            {changePct.toFixed(3)}%)
          </span>
        </div>
        <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
          <span className="flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-full bg-green-500" />
            Buy Signal
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-full bg-red-500" />
            Sell Signal
          </span>
          <span className="flex items-center gap-1">
            <span
              className="inline-block h-0.5 w-3"
              style={{ backgroundColor: TEAL }}
            />
            Mamba SSM
          </span>
        </div>
      </div>

      {/* Chart */}
      <div className="flex-1 min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 5, right: 10, bottom: 0, left: 0 }}>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke={GRID_COLOR}
              vertical={false}
            />
            <XAxis
              dataKey="idx"
              tick={false}
              axisLine={{ stroke: GRID_COLOR }}
              tickLine={false}
            />
            <YAxis
              yAxisId="price"
              domain={[priceMin, priceMax]}
              tick={{ fill: MUTED, fontSize: 10, fontFamily: "var(--font-jetbrains-mono)" }}
              axisLine={false}
              tickLine={false}
              width={55}
              tickFormatter={(v: number) => v.toFixed(2)}
            />
            <YAxis
              yAxisId="volume"
              orientation="right"
              domain={[0, "auto"]}
              hide
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
              formatter={(value: number, name: string) => {
                if (name === "price") return [value.toFixed(2), "Price"]
                if (name === "volume") return [value, "Volume"]
                return [value, name]
              }}
            />

            {/* Volume bars */}
            <Bar
              yAxisId="volume"
              dataKey="volume"
              fill={TEAL}
              opacity={0.15}
              barSize={2}
            />

            {/* Price line */}
            <Area
              yAxisId="price"
              type="monotone"
              dataKey="price"
              stroke={TEAL}
              strokeWidth={1.5}
              fill={TEAL}
              fillOpacity={0.05}
              dot={false}
              isAnimationActive={false}
            />

            {/* ML Signal dots */}
            {buySignals.map((d) => (
              <ReferenceDot
                key={`buy-${d.idx}`}
                yAxisId="price"
                x={d.idx}
                y={d.price}
                r={3}
                fill={GREEN}
                stroke="none"
              />
            ))}
            {sellSignals.map((d) => (
              <ReferenceDot
                key={`sell-${d.idx}`}
                yAxisId="price"
                x={d.idx}
                y={d.price}
                r={3}
                fill={RED}
                stroke="none"
              />
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
