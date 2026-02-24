"use client"

import { Info } from "lucide-react"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"

interface PanelInfoProps {
  title: string
  description: string
  engineering: string
}

export function PanelInfo({ title, description, engineering }: PanelInfoProps) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          className="inline-flex h-4 w-4 shrink-0 items-center justify-center rounded-full text-muted-foreground/50 transition-colors hover:bg-secondary hover:text-muted-foreground"
          aria-label={`Info about ${title}`}
        >
          <Info className="h-3 w-3" />
        </button>
      </PopoverTrigger>
      <PopoverContent side="bottom" align="start" className="w-80 space-y-2">
        <p className="text-xs font-semibold text-foreground">{title}</p>
        <p className="text-xs leading-relaxed text-muted-foreground">
          {description}
        </p>
        <div className="rounded border border-border bg-secondary/50 px-2.5 py-2">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-primary">
            Systems Engineering
          </p>
          <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
            {engineering}
          </p>
        </div>
      </PopoverContent>
    </Popover>
  )
}

export const PANEL_INFO = {
  orderBook: {
    title: "Order Book",
    description:
      "20-level bid/ask depth reconstructed in real-time from CME MDP 3.0 market data events. Shows aggregated quantity at each price level with visual depth bars.",
    engineering:
      "BTreeMap-based price level indexing with O(log n) updates. Zero-copy SBE protocol parsing via the nom crate. 45ns median LOB update latency benchmarked with Criterion.",
  },
  priceChart: {
    title: "ES Futures Price Chart",
    description:
      "Real-time E-mini S&P 500 futures price with ML-generated buy/sell signals. Green dots = buy signals, red dots = sell signals from the Mamba State-Space Model.",
    engineering:
      "Mamba SSM ingests 100-tick LOB tensor windows (100×40) and outputs directional predictions at 580ns median inference via ONNX Runtime in Rust. Streamed to the UI via Server-Sent Events (SSE).",
  },
  risk: {
    title: "Risk Management",
    description:
      "Real-time position limits, drawdown monitoring, inventory skew tracking, and kill-switch status. Ensures the strategy stays within defined risk bounds.",
    engineering:
      "Kill-switch triggers on max drawdown (5%) or position limit breach. Inventory skew penalizes directional exposure via quadratic cost. All risk checks run in the hot path at < 50ns overhead.",
  },
  tradeBlotter: {
    title: "Trade Blotter",
    description:
      "Every simulated fill with timestamp, side, price, quantity, per-trade P&L, execution latency, and the ML signal source that triggered the trade.",
    engineering:
      "Event-driven execution simulator with queue position tracking, partial fills, and configurable latency injection (50μs–500μs colo-to-exchange RTT). Models adverse selection and maker/taker fee schedules.",
  },
  pnl: {
    title: "P&L & Performance",
    description:
      "Cumulative profit/loss curve with key metrics: Sharpe ratio, win rate, max drawdown, total trades, and fill rate. Updated in real-time as fills occur.",
    engineering:
      "Rolling window statistics with O(1) incremental updates using Welford's online algorithm. Annualized Sharpe calculation, real-time max drawdown tracking via high-water mark.",
  },
  latency: {
    title: "Latency Monitor",
    description:
      "Component-level timing breakdown: p50, p95, p99, min, and max latencies across the full pipeline. Histogram shows distribution of tick-to-trade latencies.",
    engineering:
      "std::time::Instant measurements around each pipeline stage — market data parsing (45ns), feature extraction (120ns), ML inference (580ns), order routing. Sub-microsecond granularity with zero allocation in the hot path.",
  },
} as const
