import {
  Activity,
  Brain,
  Shield,
  BarChart3,
  Layers,
  Cpu,
} from "lucide-react"

const features = [
  {
    icon: Cpu,
    title: "Zero-Copy Protocol Parser",
    description:
      "CME MDP 3.0 binary protocol parser with zero-copy deserialization for minimal allocation overhead.",
  },
  {
    icon: Brain,
    title: "Mamba SSM Inference",
    description:
      "State Space Model for price prediction with sub-800ns inference latency via ONNX Runtime in Rust.",
  },
  {
    icon: Activity,
    title: "Real-Time Order Book",
    description:
      "BTreeMap-based order book reconstruction with O(log n) operations for live market depth tracking.",
  },
  {
    icon: BarChart3,
    title: "Event-Driven Backtester",
    description:
      "Realistic latency simulation with walk-forward validation and purged cross-validation.",
  },
  {
    icon: Shield,
    title: "Risk Management",
    description:
      "Position limits, maximum drawdown kill-switch, and real-time P&L tracking with fill rate monitoring.",
  },
  {
    icon: Layers,
    title: "Production Stack",
    description:
      "Docker containerization, Prometheus metrics, and Grafana dashboards for full observability.",
  },
]

export function FeaturesSection() {
  return (
    <section id="features" className="border-t border-border/50 py-24">
      <div className="mx-auto max-w-7xl px-6">
        <div className="mb-16 text-center">
          <h2 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
            Core Capabilities
          </h2>
          <p className="mt-4 text-lg text-muted-foreground">
            Built from the ground up for ultra-low latency market making
          </p>
        </div>

        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((feature) => (
            <div
              key={feature.title}
              className="group rounded-lg border border-border/50 bg-card/50 p-6 transition-colors hover:border-primary/30 hover:bg-card"
            >
              <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-md bg-primary/10 text-primary transition-colors group-hover:bg-primary/20">
                <feature.icon className="h-5 w-5" />
              </div>
              <h3 className="mb-2 text-lg font-semibold text-foreground">
                {feature.title}
              </h3>
              <p className="text-sm leading-relaxed text-muted-foreground">
                {feature.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
