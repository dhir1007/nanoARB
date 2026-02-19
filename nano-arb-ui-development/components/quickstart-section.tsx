"use client"

import { useState } from "react"
import { Check, Copy } from "lucide-react"
import { Badge } from "@/components/ui/badge"

const commands = [
  {
    label: "Build",
    code: `# Clone the repository
git clone https://github.com/dhir1007/nanoARB.git
cd nanoARB

# Build all crates in release mode
cargo build --release`,
  },
  {
    label: "Test",
    code: `# Run all 115 unit tests
cargo test --all

# Run with output for debugging
cargo test --all -- --nocapture`,
  },
  {
    label: "Backtest",
    code: `# Run the event-driven backtester
cargo run --release --bin backtest -- \\
  --data ./data/es_futures.csv \\
  --config ./config/strategy.toml`,
  },
  {
    label: "Docker",
    code: `# Build and run with monitoring stack
docker-compose up -d

# Access Grafana dashboards
# http://localhost:3000
# Prometheus: http://localhost:9090`,
  },
  {
    label: "ML Train",
    code: `# Train the Mamba SSM model
cd ml/
python train.py --config config.yaml

# Export to ONNX for Rust inference
python export_onnx.py --checkpoint best.pt`,
  },
]

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <button
      onClick={handleCopy}
      className="flex h-7 w-7 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
      aria-label="Copy to clipboard"
    >
      {copied ? <Check className="h-3.5 w-3.5 text-chart-2" /> : <Copy className="h-3.5 w-3.5" />}
    </button>
  )
}

export function QuickstartSection() {
  const [activeTab, setActiveTab] = useState(0)

  return (
    <section id="quickstart" className="border-t border-border/50 py-24">
      <div className="mx-auto max-w-7xl px-6">
        <div className="mb-16 text-center">
          <h2 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
            Quick Start
          </h2>
          <p className="mt-4 text-lg text-muted-foreground">
            Get up and running in minutes
          </p>
        </div>

        <div className="mx-auto max-w-3xl">
          {/* Tabs */}
          <div className="mb-0 flex gap-1 overflow-x-auto rounded-t-lg border border-b-0 border-border/50 bg-secondary/50 p-1">
            {commands.map((cmd, i) => (
              <button
                key={cmd.label}
                onClick={() => setActiveTab(i)}
                className={`rounded-md px-4 py-2 text-sm font-medium transition-colors ${
                  activeTab === i
                    ? "bg-background text-foreground"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {cmd.label}
              </button>
            ))}
          </div>

          {/* Code block */}
          <div className="rounded-b-lg border border-border/50 bg-[oklch(0.08_0.005_250)]">
            <div className="flex items-center justify-between border-b border-border/30 px-4 py-2">
              <Badge
                variant="outline"
                className="border-border/50 text-[10px] text-muted-foreground"
              >
                {commands[activeTab].label === "ML Train" ? "python" : "bash"}
              </Badge>
              <CopyButton text={commands[activeTab].code} />
            </div>
            <pre className="overflow-x-auto p-6">
              <code className="font-mono text-sm leading-relaxed text-foreground/80">
                {commands[activeTab].code}
              </code>
            </pre>
          </div>
        </div>

        {/* Tech stack */}
        <div className="mt-20">
          <h3 className="mb-6 text-center text-sm font-semibold uppercase tracking-wider text-muted-foreground">
            Built With
          </h3>
          <div className="flex flex-wrap items-center justify-center gap-3">
            {[
              "Rust",
              "Python",
              "PyTorch",
              "ONNX Runtime",
              "Docker",
              "Prometheus",
              "Grafana",
              "CME MDP 3.0",
              "BTreeMap",
              "Mamba SSM",
            ].map((tech) => (
              <span
                key={tech}
                className="rounded-full border border-border/50 bg-secondary/50 px-4 py-1.5 font-mono text-xs text-muted-foreground"
              >
                {tech}
              </span>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
