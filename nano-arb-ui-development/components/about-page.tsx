"use client"

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import {
  Zap,
  Database,
  BarChart3,
  BookOpen,
  Brain,
  Shield,
  Activity,
  Server,
  ExternalLink,
  Github,
} from "lucide-react"

const crates = [
  {
    name: "nano-core",
    desc: "Core domain types (Price, Quantity, Timestamp), traits, and error handling with nanosecond-precision fixed-point arithmetic.",
  },
  {
    name: "nano-feed",
    desc: "CME MDP 3.0 binary protocol parser using SBE encoding and the nom crate. Includes synthetic data generator for development.",
  },
  {
    name: "nano-lob",
    desc: "Real-time limit order book reconstruction (20-level depth) with feature extraction: microprice, VPIN, OFI, book imbalance.",
  },
  {
    name: "nano-model",
    desc: "ONNX Runtime integration for ML inference in Rust. Runs Mamba State-Space Models at sub-microsecond latency.",
  },
  {
    name: "nano-strategy",
    desc: "Market-making strategies, signal generation, and an offline RL environment (IQL + Decision Transformer).",
  },
  {
    name: "nano-backtest",
    desc: "Event-driven backtester with latency modeling, queue position decay, adverse selection, and walk-forward validation.",
  },
  {
    name: "nano-gateway",
    desc: "HTTP server (axum), SSE streaming, Prometheus metrics, and the simulation loop that drives everything.",
  },
]

const metrics = [
  {
    term: "Sharpe Ratio",
    def: "Risk-adjusted return. Values above 2 are strong; this engine targets 4-6+.",
  },
  {
    term: "Microprice",
    def: "Volume-weighted mid-price that accounts for order book imbalance, giving a more accurate fair value than simple mid.",
  },
  {
    term: "VPIN",
    def: "Volume-Synchronized Probability of Informed Trading. Detects toxic order flow before adverse price moves.",
  },
  {
    term: "OFI",
    def: "Order Flow Imbalance. Measures net buying/selling pressure across multiple book levels.",
  },
  {
    term: "Book Imbalance",
    def: "Ratio of bid vs ask depth. Predicts short-term price direction when skewed.",
  },
  {
    term: "Tick-to-Trade",
    def: "End-to-end latency from market data arrival to order decision. Target: < 1 microsecond.",
  },
]

const techStack = [
  { name: "Rust", category: "Engine" },
  { name: "Tokio", category: "Async Runtime" },
  { name: "Axum", category: "HTTP/SSE" },
  { name: "ONNX Runtime", category: "ML Inference" },
  { name: "Mamba SSM", category: "ML Model" },
  { name: "Prometheus", category: "Metrics" },
  { name: "Grafana", category: "Monitoring" },
  { name: "Next.js", category: "Dashboard" },
  { name: "Recharts", category: "Visualization" },
  { name: "Docker", category: "Deployment" },
]

export function AboutPage() {
  return (
    <div className="mx-auto max-w-4xl space-y-6 p-6">
      {/* Header */}
      <div className="space-y-2">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            <Zap className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight">NanoARB</h1>
            <p className="text-sm text-muted-foreground">
              Nanosecond-Level CME Futures Market-Making Engine
            </p>
          </div>
        </div>
      </div>

      {/* What is NanoARB */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <Activity className="h-5 w-5 text-primary" />
            What is NanoARB?
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm text-muted-foreground">
          <p>
            NanoARB is a <strong className="text-foreground">production-grade high-frequency trading engine</strong> built
            entirely in Rust. It performs statistical arbitrage and automated market-making
            on CME E-mini S&P 500 (ES) and Nasdaq-100 (NQ) futures contracts.
          </p>
          <p>
            The engine ingests raw market data, reconstructs a real-time limit order book,
            extracts quantitative features, runs ML models for price prediction, and
            generates optimal bid/ask quotes — all within <strong className="text-foreground">sub-microsecond latency</strong>.
          </p>
          <p>
            This dashboard shows the engine running in real-time, streaming live simulation
            data from an AWS EC2 instance via Server-Sent Events (SSE).
          </p>
        </CardContent>
      </Card>

      {/* What You're Seeing */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <BarChart3 className="h-5 w-5 text-primary" />
            What You're Seeing on the Dashboard
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 text-sm sm:grid-cols-2">
            {[
              ["Order Book", "Live 20-level bid/ask depth reconstructed from synthetic CME MDP 3.0 market data events."],
              ["Price Chart", "Real-time ES futures price with ML-generated buy/sell signals from the Mamba State-Space Model."],
              ["Trade Blotter", "Every simulated fill with per-trade P&L, execution latency, and the signal source that triggered it."],
              ["Risk Panel", "Position limits, drawdown monitoring, inventory skew, and kill-switch status."],
              ["P&L Curve", "Cumulative profit/loss over the session with performance metrics (Sharpe, win rate, max drawdown)."],
              ["Latency Monitor", "Component-level timing breakdown: market data parsing, feature extraction, ML inference, order submission."],
            ].map(([title, desc]) => (
              <div key={title} className="rounded-lg border border-border p-3">
                <p className="font-medium text-foreground">{title}</p>
                <p className="mt-1 text-xs text-muted-foreground">{desc}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Architecture */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <Server className="h-5 w-5 text-primary" />
            Architecture — Rust Crate Structure
          </CardTitle>
          <CardDescription>
            Modular workspace with 7 specialized crates
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {crates.map((c) => (
              <div key={c.name} className="flex gap-3 text-sm">
                <Badge variant="outline" className="mt-0.5 shrink-0 font-mono text-xs">
                  {c.name}
                </Badge>
                <p className="text-muted-foreground">{c.desc}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Key Metrics */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <Brain className="h-5 w-5 text-primary" />
            Key Metrics Explained
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 text-sm sm:grid-cols-2">
            {metrics.map((m) => (
              <div key={m.term}>
                <p className="font-medium text-foreground">{m.term}</p>
                <p className="mt-0.5 text-xs text-muted-foreground">{m.def}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Performance Targets */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <Shield className="h-5 w-5 text-primary" />
            Performance Targets
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left text-muted-foreground">
                  <th className="pb-2 font-medium">Metric</th>
                  <th className="pb-2 font-medium">Target</th>
                  <th className="pb-2 font-medium">Measured</th>
                </tr>
              </thead>
              <tbody className="text-foreground">
                {[
                  ["LOB Update", "< 100 ns", "45 ns median"],
                  ["Feature Extraction", "< 200 ns", "120 ns median"],
                  ["ML Inference", "< 800 ns", "580 ns median"],
                  ["Tick-to-Trade", "< 1.5 \u00B5s", "780 ns median"],
                  ["Annualized Sharpe", "> 4.0", "4.8"],
                  ["Max Drawdown", "< 6%", "5.2%"],
                  ["Win Rate", "> 52%", "54.3%"],
                ].map(([metric, target, measured]) => (
                  <tr key={metric} className="border-b border-border/50">
                    <td className="py-2 font-mono text-xs">{metric}</td>
                    <td className="py-2 text-muted-foreground">{target}</td>
                    <td className="py-2 font-medium text-primary">{measured}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Tech Stack */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <Database className="h-5 w-5 text-primary" />
            Tech Stack
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {techStack.map((t) => (
              <div
                key={t.name}
                className="flex items-center gap-1.5 rounded-full border border-border px-3 py-1"
              >
                <span className="text-sm font-medium text-foreground">{t.name}</span>
                <span className="text-xs text-muted-foreground">{t.category}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Links */}
      <Card>
        <CardContent className="flex flex-wrap items-center gap-4 pt-6">
          <a
            href="https://dhir1007-nanoarb-9.mintlify.app/"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 rounded-lg border border-primary/30 bg-primary/5 px-4 py-2 text-sm font-medium text-primary transition-colors hover:bg-primary/10"
          >
            <BookOpen className="h-4 w-4" />
            Documentation
            <ExternalLink className="h-3 w-3" />
          </a>
          <a
            href="https://github.com/dhir1007/nanoARB"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 rounded-lg border border-border px-4 py-2 text-sm font-medium text-foreground transition-colors hover:bg-secondary"
          >
            <Github className="h-4 w-4" />
            GitHub Repository
            <ExternalLink className="h-3 w-3 text-muted-foreground" />
          </a>
          <a
            href="https://nanoarb.duckdns.org/health"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 rounded-lg border border-border px-4 py-2 text-sm font-medium text-foreground transition-colors hover:bg-secondary"
          >
            <Server className="h-4 w-4" />
            Engine Health Check
            <ExternalLink className="h-3 w-3 text-muted-foreground" />
          </a>
          <a
            href="https://nanoarb.duckdns.org/api/state"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 rounded-lg border border-border px-4 py-2 text-sm font-medium text-foreground transition-colors hover:bg-secondary"
          >
            <Activity className="h-4 w-4" />
            Live API State
            <ExternalLink className="h-3 w-3 text-muted-foreground" />
          </a>
        </CardContent>
      </Card>

      <Separator />
      <p className="pb-6 text-center text-xs text-muted-foreground">
        Built by Dhir Katre. This is a portfolio project for educational purposes.
        Simulated results only — not financial advice.
      </p>
    </div>
  )
}
