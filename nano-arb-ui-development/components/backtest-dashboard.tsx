"use client"

import { useState, useCallback } from "react"
import {
  type BacktestConfig,
  type BacktestResult,
} from "@/lib/mock-data"

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:9090"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Switch } from "@/components/ui/switch"
import { Separator } from "@/components/ui/separator"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  AreaChart,
  Area,
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
import { Play, RotateCcw } from "lucide-react"

const TEAL = "#2dd4bf"
const GREEN = "#22c55e"
const RED = "#ef4444"
const GRID_COLOR = "#1e293b"
const MUTED = "#64748b"

export function BacktestDashboard() {
  const [config, setConfig] = useState<BacktestConfig>({
    symbol: "ES",
    startDate: "2024-01-01",
    endDate: "2024-12-31",
    initialCapital: 1000000,
    spreadMultiplier: 1.0,
    inventoryLimit: 50,
    skewFactor: 0.5,
    useML: true,
    maxDrawdown: 5.0,
    positionLimit: 50,
  })

  const [isRunning, setIsRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState<BacktestResult | null>(null)

  const runBacktest = useCallback(async () => {
    setIsRunning(true)
    setProgress(0)
    setResult(null)

    // Animate progress bar while waiting for the backend
    let p = 0
    const interval = setInterval(() => {
      p += Math.random() * 8
      if (p > 90) p = 90
      setProgress(Math.floor(p))
    }, 200)

    try {
      const res = await fetch(`${API_BASE}/api/backtest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      })

      clearInterval(interval)

      if (!res.ok) {
        const msg = await res.text()
        throw new Error(msg || `HTTP ${res.status}`)
      }

      setProgress(100)
      const data: BacktestResult = await res.json()
      setTimeout(() => {
        setResult(data)
        setIsRunning(false)
      }, 200)
    } catch (err) {
      clearInterval(interval)
      setProgress(0)
      setIsRunning(false)
      console.error("Backtest failed:", err)
    }
  }, [config])

  const resetBacktest = () => {
    setResult(null)
    setProgress(0)
    setIsRunning(false)
  }

  return (
    <div className="flex h-full">
      {/* Config sidebar */}
      <aside className="flex w-72 shrink-0 flex-col border-r border-border bg-secondary/20 p-4">
        <h3 className="pb-4 text-sm font-semibold text-foreground">
          Backtest Configuration
        </h3>

        <div className="flex flex-col gap-4">
          {/* Symbol */}
          <div>
            <label className="pb-1 text-[10px] uppercase tracking-wider text-muted-foreground">
              Symbol
            </label>
            <Select
              value={config.symbol}
              onValueChange={(v) => setConfig({ ...config, symbol: v })}
            >
              <SelectTrigger className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="ES">ES - E-mini S&P 500</SelectItem>
                <SelectItem value="NQ">NQ - E-mini Nasdaq</SelectItem>
                <SelectItem value="CL">CL - Crude Oil</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Date range */}
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="pb-1 text-[10px] uppercase tracking-wider text-muted-foreground">
                Start
              </label>
              <input
                type="date"
                value={config.startDate}
                onChange={(e) =>
                  setConfig({ ...config, startDate: e.target.value })
                }
                className="flex h-8 w-full rounded-md border border-input bg-background px-2 text-xs text-foreground"
              />
            </div>
            <div>
              <label className="pb-1 text-[10px] uppercase tracking-wider text-muted-foreground">
                End
              </label>
              <input
                type="date"
                value={config.endDate}
                onChange={(e) =>
                  setConfig({ ...config, endDate: e.target.value })
                }
                className="flex h-8 w-full rounded-md border border-input bg-background px-2 text-xs text-foreground"
              />
            </div>
          </div>

          {/* Capital */}
          <div>
            <label className="pb-1 text-[10px] uppercase tracking-wider text-muted-foreground">
              Initial Capital
            </label>
            <input
              type="number"
              value={config.initialCapital}
              onChange={(e) =>
                setConfig({
                  ...config,
                  initialCapital: Number(e.target.value),
                })
              }
              className="flex h-8 w-full rounded-md border border-input bg-background px-2 font-mono text-xs text-foreground"
            />
          </div>

          <Separator />

          {/* Strategy params */}
          <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
            Strategy Parameters
          </p>

          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-[10px] text-muted-foreground">
                Spread Mult
              </label>
              <input
                type="number"
                step="0.1"
                value={config.spreadMultiplier}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    spreadMultiplier: Number(e.target.value),
                  })
                }
                className="flex h-7 w-full rounded-md border border-input bg-background px-2 font-mono text-xs text-foreground"
              />
            </div>
            <div>
              <label className="text-[10px] text-muted-foreground">
                Inv Limit
              </label>
              <input
                type="number"
                value={config.inventoryLimit}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    inventoryLimit: Number(e.target.value),
                  })
                }
                className="flex h-7 w-full rounded-md border border-input bg-background px-2 font-mono text-xs text-foreground"
              />
            </div>
          </div>

          <div>
            <label className="text-[10px] text-muted-foreground">
              Skew Factor
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={config.skewFactor}
              onChange={(e) =>
                setConfig({
                  ...config,
                  skewFactor: Number(e.target.value),
                })
              }
              className="w-full accent-primary"
            />
            <div className="flex justify-between font-mono text-[9px] text-muted-foreground">
              <span>0.0</span>
              <span className="text-foreground">{config.skewFactor}</span>
              <span>1.0</span>
            </div>
          </div>

          {/* ML toggle */}
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-foreground">Mamba SSM</p>
              <p className="text-[10px] text-muted-foreground">
                ML-enhanced quotes
              </p>
            </div>
            <Switch
              checked={config.useML}
              onCheckedChange={(v) => setConfig({ ...config, useML: v })}
            />
          </div>

          <Separator />

          {/* Risk params */}
          <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
            Risk Parameters
          </p>

          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className="text-[10px] text-muted-foreground">
                Max DD %
              </label>
              <input
                type="number"
                step="0.5"
                value={config.maxDrawdown}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    maxDrawdown: Number(e.target.value),
                  })
                }
                className="flex h-7 w-full rounded-md border border-input bg-background px-2 font-mono text-xs text-foreground"
              />
            </div>
            <div>
              <label className="text-[10px] text-muted-foreground">
                Pos Limit
              </label>
              <input
                type="number"
                value={config.positionLimit}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    positionLimit: Number(e.target.value),
                  })
                }
                className="flex h-7 w-full rounded-md border border-input bg-background px-2 font-mono text-xs text-foreground"
              />
            </div>
          </div>
        </div>

        {/* Action buttons */}
        <div className="mt-auto flex flex-col gap-2 pt-4">
          {isRunning && (
            <div className="flex flex-col gap-1">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-muted-foreground">
                  Running backtest...
                </span>
                <span className="font-mono text-[10px] text-primary">
                  {progress}%
                </span>
              </div>
              <Progress value={progress} className="h-1.5" />
            </div>
          )}
          <Button
            onClick={runBacktest}
            disabled={isRunning}
            className="gap-2"
            size="sm"
          >
            <Play className="h-3.5 w-3.5" />
            {isRunning ? "Running..." : "Run Backtest"}
          </Button>
          {result && (
            <Button
              onClick={resetBacktest}
              variant="outline"
              size="sm"
              className="gap-2"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              Reset
            </Button>
          )}
        </div>
      </aside>

      {/* Results area */}
      <div className="flex-1 overflow-auto p-4">
        {!result && !isRunning && (
          <div className="flex h-full flex-col items-center justify-center text-muted-foreground">
            <Play className="mb-3 h-10 w-10 opacity-30" />
            <p className="text-sm">
              Configure parameters and run a backtest to see results
            </p>
          </div>
        )}

        {isRunning && (
          <div className="flex h-full flex-col items-center justify-center">
            <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
            <p className="mt-3 text-sm text-muted-foreground">
              Simulating {config.symbol} market-making strategy...
            </p>
            <p className="font-mono text-xs text-muted-foreground">
              Processing 252 trading days
            </p>
          </div>
        )}

        {result && <BacktestResults result={result} config={config} />}
      </div>
    </div>
  )
}

// ─── Results Component ──────────────────────────────────────────────────────

function BacktestResults({
  result,
  config,
}: {
  result: BacktestResult
  config: BacktestConfig
}) {
  const summaryCards = [
    {
      label: "Total Return",
      value: `${result.totalReturn >= 0 ? "+" : ""}${result.totalReturn.toFixed(2)}%`,
      color: result.totalReturn >= 0 ? "text-green-400" : "text-red-400",
    },
    {
      label: "Sharpe Ratio",
      value: result.sharpe.toFixed(2),
      color: result.sharpe > 1.5 ? "text-green-400" : "text-foreground",
    },
    {
      label: "Max Drawdown",
      value: `${result.maxDrawdown.toFixed(2)}%`,
      color: "text-red-400",
    },
    {
      label: "Win Rate",
      value: `${result.winRate.toFixed(1)}%`,
      color: result.winRate > 50 ? "text-green-400" : "text-red-400",
    },
    {
      label: "Profit Factor",
      value: result.profitFactor.toFixed(2),
      color: result.profitFactor > 1.5 ? "text-green-400" : "text-foreground",
    },
    {
      label: "Total Trades",
      value: result.totalTrades.toLocaleString(),
      color: "text-foreground",
    },
  ]

  return (
    <div className="flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center gap-3">
        <h3 className="text-sm font-semibold text-foreground">
          Backtest Results
        </h3>
        <Badge
          variant="outline"
          className="border-primary/30 font-mono text-[10px] text-primary"
        >
          {config.symbol}
        </Badge>
        <Badge
          variant="outline"
          className="font-mono text-[10px] text-muted-foreground"
        >
          {config.startDate} to {config.endDate}
        </Badge>
        {config.useML && (
          <Badge
            variant="outline"
            className="border-primary/30 font-mono text-[10px] text-primary"
          >
            Mamba SSM
          </Badge>
        )}
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-6 gap-3">
        {summaryCards.map((card) => (
          <div
            key={card.label}
            className="rounded-lg border border-border bg-secondary/30 px-3 py-2"
          >
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
              {card.label}
            </p>
            <p className={`font-mono text-lg font-bold ${card.color}`}>
              {card.value}
            </p>
          </div>
        ))}
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-2 gap-4">
        {/* Equity curve */}
        <div className="rounded-lg border border-border bg-secondary/20 p-3">
          <p className="pb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Equity Curve
          </p>
          <div style={{ height: 220 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={result.equityCurve}
                margin={{ top: 5, right: 10, bottom: 0, left: 10 }}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke={GRID_COLOR}
                  vertical={false}
                />
                <XAxis
                  dataKey="day"
                  tick={{ fill: MUTED, fontSize: 9 }}
                  axisLine={{ stroke: GRID_COLOR }}
                  tickLine={false}
                />
                <YAxis
                  tick={{
                    fill: MUTED,
                    fontSize: 9,
                    fontFamily: "var(--font-jetbrains-mono)",
                  }}
                  axisLine={false}
                  tickLine={false}
                  width={60}
                  tickFormatter={(v: number) =>
                    `$${(v / 1000).toFixed(0)}K`
                  }
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0f172a",
                    border: "1px solid #1e293b",
                    borderRadius: 6,
                    fontSize: 11,
                    fontFamily: "var(--font-jetbrains-mono)",
                  }}
                  formatter={(value: number) => [
                    `$${value.toLocaleString()}`,
                    "Equity",
                  ]}
                  labelFormatter={(label: number) => `Day ${label}`}
                />
                <Area
                  type="monotone"
                  dataKey="equity"
                  stroke={GREEN}
                  strokeWidth={1.5}
                  fill={GREEN}
                  fillOpacity={0.1}
                  dot={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Drawdown chart */}
        <div className="rounded-lg border border-border bg-secondary/20 p-3">
          <p className="pb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Drawdown
          </p>
          <div style={{ height: 220 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={result.drawdownCurve}
                margin={{ top: 5, right: 10, bottom: 0, left: 10 }}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke={GRID_COLOR}
                  vertical={false}
                />
                <XAxis
                  dataKey="day"
                  tick={{ fill: MUTED, fontSize: 9 }}
                  axisLine={{ stroke: GRID_COLOR }}
                  tickLine={false}
                />
                <YAxis
                  tick={{
                    fill: MUTED,
                    fontSize: 9,
                    fontFamily: "var(--font-jetbrains-mono)",
                  }}
                  axisLine={false}
                  tickLine={false}
                  width={35}
                  tickFormatter={(v: number) => `${v}%`}
                  reversed
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0f172a",
                    border: "1px solid #1e293b",
                    borderRadius: 6,
                    fontSize: 11,
                    fontFamily: "var(--font-jetbrains-mono)",
                  }}
                  formatter={(value: number) => [
                    `${value.toFixed(2)}%`,
                    "Drawdown",
                  ]}
                  labelFormatter={(label: number) => `Day ${label}`}
                />
                <Area
                  type="monotone"
                  dataKey="drawdown"
                  stroke={RED}
                  strokeWidth={1.5}
                  fill={RED}
                  fillOpacity={0.15}
                  dot={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Bottom row: Trade distribution + Monthly returns */}
      <div className="grid grid-cols-2 gap-4">
        {/* Trade distribution */}
        <div className="rounded-lg border border-border bg-secondary/20 p-3">
          <p className="pb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Trade P&L Distribution
          </p>
          <div style={{ height: 180 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={result.tradeDistribution}
                margin={{ top: 5, right: 5, bottom: 0, left: 5 }}
              >
                <XAxis
                  dataKey="bucket"
                  tick={{ fill: MUTED, fontSize: 9 }}
                  axisLine={{ stroke: GRID_COLOR }}
                  tickLine={false}
                />
                <YAxis hide />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0f172a",
                    border: "1px solid #1e293b",
                    borderRadius: 6,
                    fontSize: 11,
                  }}
                  formatter={(value: number) => [value, "Trades"]}
                />
                <Bar
                  dataKey="count"
                  fill={TEAL}
                  radius={[3, 3, 0, 0]}
                  opacity={0.8}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Monthly returns */}
        <div className="rounded-lg border border-border bg-secondary/20 p-3">
          <p className="pb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Monthly Returns
          </p>
          <div className="grid grid-cols-6 gap-1.5">
            {result.monthlyReturns.map((m) => (
              <div
                key={m.month}
                className="flex flex-col items-center rounded border border-border px-2 py-2"
                style={{
                  backgroundColor:
                    m.return >= 0
                      ? `rgba(34, 197, 94, ${Math.min(Math.abs(m.return) / 6, 0.3)})`
                      : `rgba(239, 68, 68, ${Math.min(Math.abs(m.return) / 6, 0.3)})`,
                }}
              >
                <span className="text-[10px] text-muted-foreground">
                  {m.month}
                </span>
                <span
                  className={`font-mono text-xs font-semibold ${m.return >= 0 ? "text-green-400" : "text-red-400"}`}
                >
                  {m.return >= 0 ? "+" : ""}
                  {m.return.toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
