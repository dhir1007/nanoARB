import { TrendingUp, TrendingDown } from "lucide-react"

const metrics = [
  {
    label: "Sharpe Ratio",
    value: "2.4",
    change: "+0.3 vs baseline",
    positive: true,
  },
  {
    label: "Max Drawdown",
    value: "-1.2%",
    change: "Within 2% limit",
    positive: true,
  },
  {
    label: "Win Rate",
    value: "63.8%",
    change: "Per-trade basis",
    positive: true,
  },
  {
    label: "Avg Fill Rate",
    value: "89.4%",
    change: "Passive fills",
    positive: true,
  },
  {
    label: "Daily P&L Vol",
    value: "$2.1K",
    change: "Simulated notional",
    positive: true,
  },
  {
    label: "Latency Impact",
    value: "-12bps",
    change: "vs. no-latency sim",
    positive: false,
  },
]

export function BacktestSection() {
  return (
    <section id="backtest" className="border-t border-border/50 py-24">
      <div className="mx-auto max-w-7xl px-6">
        <div className="mb-16 text-center">
          <h2 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
            Backtest Results
          </h2>
          <p className="mt-4 text-lg text-muted-foreground">
            Walk-forward validated with realistic latency simulation on ES
            futures
          </p>
        </div>

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {metrics.map((metric) => (
            <div
              key={metric.label}
              className="rounded-lg border border-border/50 bg-card/50 p-6"
            >
              <div className="mb-1 text-sm text-muted-foreground">
                {metric.label}
              </div>
              <div className="mb-2 font-mono text-3xl font-bold text-foreground">
                {metric.value}
              </div>
              <div className="flex items-center gap-1.5">
                {metric.positive ? (
                  <TrendingUp className="h-3 w-3 text-chart-2" />
                ) : (
                  <TrendingDown className="h-3 w-3 text-chart-5" />
                )}
                <span
                  className={`text-xs ${
                    metric.positive ? "text-chart-2" : "text-chart-5"
                  }`}
                >
                  {metric.change}
                </span>
              </div>
            </div>
          ))}
        </div>

        {/* Strategy explanation */}
        <div className="mt-12 rounded-lg border border-border/50 bg-card/50 p-8">
          <h3 className="mb-4 text-xl font-semibold text-foreground">
            Market-Making Strategy
          </h3>
          <div className="grid gap-6 md:grid-cols-3">
            <div>
              <h4 className="mb-2 text-sm font-semibold text-primary">
                Quote Generation
              </h4>
              <p className="text-sm leading-relaxed text-muted-foreground">
                Bid/ask quotes are placed symmetrically around the ML-predicted
                fair value, with spread determined by volatility estimates and
                inventory position.
              </p>
            </div>
            <div>
              <h4 className="mb-2 text-sm font-semibold text-primary">
                Inventory Skew
              </h4>
              <p className="text-sm leading-relaxed text-muted-foreground">
                Quotes are skewed based on current inventory to encourage mean
                reversion. Larger positions result in more aggressive pricing on
                the reducing side.
              </p>
            </div>
            <div>
              <h4 className="mb-2 text-sm font-semibold text-primary">
                Risk Controls
              </h4>
              <p className="text-sm leading-relaxed text-muted-foreground">
                Hard position limits, maximum drawdown kill-switch, and
                per-symbol exposure caps ensure the strategy stays within
                defined risk parameters at all times.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
