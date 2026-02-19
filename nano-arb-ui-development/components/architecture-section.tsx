import { Badge } from "@/components/ui/badge"

const crates = [
  {
    name: "nanoarb-common",
    description: "Shared types, configs, and utilities across all crates",
    tag: "Foundation",
  },
  {
    name: "nanoarb-feed",
    description:
      "CME MDP 3.0 binary protocol parser with zero-copy deserialization",
    tag: "Data",
  },
  {
    name: "nanoarb-book",
    description:
      "Real-time order book reconstruction using BTreeMap with O(log n) ops",
    tag: "Data",
  },
  {
    name: "nanoarb-model",
    description: "Mamba SSM inference engine via ONNX Runtime (<800ns latency)",
    tag: "ML",
  },
  {
    name: "nanoarb-strategy",
    description:
      "Market-making strategy with inventory skew and ML-enhanced quoting",
    tag: "Trading",
  },
  {
    name: "nanoarb-engine",
    description:
      "Core event loop, risk management, and order execution pipeline",
    tag: "Core",
  },
  {
    name: "nanoarb-backtest",
    description:
      "Event-driven backtester with realistic latency simulation",
    tag: "Testing",
  },
]

const tagColors: Record<string, string> = {
  Foundation: "border-muted-foreground/30 text-muted-foreground",
  Data: "border-chart-1/30 text-chart-1",
  ML: "border-chart-4/30 text-chart-4",
  Trading: "border-chart-2/30 text-chart-2",
  Core: "border-primary/30 text-primary",
  Testing: "border-chart-5/30 text-chart-5",
}

export function ArchitectureSection() {
  return (
    <section id="architecture" className="border-t border-border/50 py-24">
      <div className="mx-auto max-w-7xl px-6">
        <div className="mb-16 text-center">
          <h2 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
            System Architecture
          </h2>
          <p className="mt-4 text-lg text-muted-foreground">
            7-crate Rust workspace with clear separation of concerns
          </p>
        </div>

        {/* Pipeline diagram */}
        <div className="mb-16 overflow-x-auto rounded-lg border border-border/50 bg-card/50 p-8">
          <div className="mx-auto flex min-w-[700px] items-center justify-center gap-3">
            {[
              { label: "CME Feed", sub: "MDP 3.0" },
              { label: "Parser", sub: "Zero-copy" },
              { label: "Order Book", sub: "BTreeMap" },
              { label: "ML Model", sub: "Mamba SSM" },
              { label: "Strategy", sub: "Inventory Skew" },
              { label: "Risk Mgmt", sub: "Kill-switch" },
              { label: "Exchange", sub: "CME" },
            ].map((step, i, arr) => (
              <div key={step.label} className="flex items-center gap-3">
                <div className="flex flex-col items-center gap-1 rounded-md border border-border bg-secondary px-4 py-3 text-center">
                  <span className="text-xs font-semibold text-foreground">
                    {step.label}
                  </span>
                  <span className="font-mono text-[10px] text-muted-foreground">
                    {step.sub}
                  </span>
                </div>
                {i < arr.length - 1 && (
                  <span className="font-mono text-sm text-primary">
                    {"->"}
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Crate grid */}
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {crates.map((crate) => (
            <div
              key={crate.name}
              className="rounded-lg border border-border/50 bg-card/50 p-5 transition-colors hover:border-primary/20"
            >
              <div className="mb-3 flex items-center justify-between">
                <code className="font-mono text-sm font-semibold text-foreground">
                  {crate.name}
                </code>
                <Badge
                  variant="outline"
                  className={`text-[10px] ${tagColors[crate.tag] || ""}`}
                >
                  {crate.tag}
                </Badge>
              </div>
              <p className="text-xs leading-relaxed text-muted-foreground">
                {crate.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
