"use client"

const benchmarks = [
  {
    label: "Market Data Parsing",
    value: "~120ns",
    ns: 120,
    max: 2000,
    description: "CME MDP 3.0 binary protocol zero-copy parsing",
  },
  {
    label: "Order Book Update",
    value: "~85ns",
    ns: 85,
    max: 2000,
    description: "BTreeMap insert/remove with O(log n) complexity",
  },
  {
    label: "ML Inference",
    value: "<800ns",
    ns: 800,
    max: 2000,
    description: "Mamba SSM forward pass via ONNX Runtime",
  },
  {
    label: "Strategy Decision",
    value: "~200ns",
    ns: 200,
    max: 2000,
    description: "Quote generation with inventory skew calculation",
  },
  {
    label: "Risk Check",
    value: "~50ns",
    ns: 50,
    max: 2000,
    description: "Position limits and drawdown validation",
  },
  {
    label: "Total Tick-to-Trade",
    value: "<2\u00B5s",
    ns: 2000,
    max: 2000,
    description: "End-to-end from market data to trading decision",
    highlight: true,
  },
]

export function PerformanceSection() {
  return (
    <section id="performance" className="border-t border-border/50 py-24">
      <div className="mx-auto max-w-7xl px-6">
        <div className="mb-16 text-center">
          <h2 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
            Performance Benchmarks
          </h2>
          <p className="mt-4 text-lg text-muted-foreground">
            Every nanosecond counts in high-frequency trading
          </p>
        </div>

        <div className="mx-auto max-w-4xl">
          <div className="rounded-lg border border-border/50 bg-card/50 overflow-hidden">
            {/* Table header */}
            <div className="grid grid-cols-[1fr_100px_1fr] gap-4 border-b border-border/50 bg-secondary/50 px-6 py-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              <span>Component</span>
              <span className="text-right">Latency</span>
              <span className="pl-4">Relative</span>
            </div>

            {/* Rows */}
            {benchmarks.map((bench) => (
              <div
                key={bench.label}
                className={`grid grid-cols-[1fr_100px_1fr] items-center gap-4 border-b border-border/30 px-6 py-4 last:border-b-0 ${
                  bench.highlight ? "bg-primary/5" : ""
                }`}
              >
                <div>
                  <div
                    className={`text-sm font-medium ${
                      bench.highlight ? "text-primary" : "text-foreground"
                    }`}
                  >
                    {bench.label}
                  </div>
                  <div className="mt-0.5 text-xs text-muted-foreground">
                    {bench.description}
                  </div>
                </div>
                <div
                  className={`text-right font-mono text-sm font-bold ${
                    bench.highlight ? "text-primary" : "text-foreground"
                  }`}
                >
                  {bench.value}
                </div>
                <div className="pl-4">
                  <div className="h-2 w-full rounded-full bg-secondary">
                    <div
                      className={`h-2 rounded-full transition-all ${
                        bench.highlight ? "bg-primary" : "bg-primary/60"
                      }`}
                      style={{
                        width: `${Math.max(
                          (bench.ns / bench.max) * 100,
                          4
                        )}%`,
                      }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* ML Pipeline callout */}
          <div className="mt-8 rounded-lg border border-primary/20 bg-primary/5 p-6">
            <h3 className="mb-2 text-lg font-semibold text-foreground">
              ML Pipeline
            </h3>
            <p className="text-sm leading-relaxed text-muted-foreground">
              Training is done in Python with PyTorch, exported to ONNX format,
              and loaded into the Rust engine for inference. Walk-forward
              validation with purged cross-validation ensures the model
              generalizes to unseen market regimes.
            </p>
            <div className="mt-4 flex flex-wrap gap-3">
              {["PyTorch Training", "ONNX Export", "Rust Inference"].map(
                (step, i) => (
                  <div key={step} className="flex items-center gap-2">
                    <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary/20 font-mono text-xs font-bold text-primary">
                      {i + 1}
                    </span>
                    <span className="text-sm text-foreground">{step}</span>
                  </div>
                )
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
