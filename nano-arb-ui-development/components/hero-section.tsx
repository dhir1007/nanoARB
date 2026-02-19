import Link from "next/link"
import { ArrowRight, Github, Zap, Timer, FlaskConical, Box } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"

const stats = [
  { icon: Timer, value: "<2\u00B5s", label: "Tick-to-Trade" },
  { icon: Zap, value: "<800ns", label: "ML Inference" },
  { icon: FlaskConical, value: "115", label: "Unit Tests" },
  { icon: Box, value: "7", label: "Rust Crates" },
]

export function HeroSection() {
  return (
    <section className="relative overflow-hidden">
      {/* Background grid effect */}
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:64px_64px]" />
      <div className="pointer-events-none absolute left-1/2 top-0 h-[500px] w-[800px] -translate-x-1/2 bg-primary/5 blur-[120px]" />

      <div className="relative mx-auto max-w-7xl px-6 pb-20 pt-24 lg:pb-32 lg:pt-36">
        <div className="flex flex-col items-center text-center">
          <Badge
            variant="outline"
            className="mb-6 border-primary/30 bg-primary/5 px-4 py-1.5 text-primary"
          >
            Production-Grade HFT Engine
          </Badge>

          <h1 className="max-w-4xl text-balance text-4xl font-bold tracking-tight text-foreground sm:text-5xl lg:text-7xl">
            Nanosecond-Level{" "}
            <span className="text-primary">Market Making</span>
          </h1>

          <p className="mt-6 max-w-2xl text-pretty text-lg leading-relaxed text-muted-foreground lg:text-xl">
            A production-grade high-frequency trading engine built entirely in
            Rust, designed for CME futures market-making with sub-microsecond
            latency and ML-enhanced price prediction.
          </p>

          <div className="mt-10 flex flex-wrap items-center justify-center gap-4">
            <Button size="lg" asChild>
              <Link
                href="https://github.com/dhir1007/nanoARB"
                target="_blank"
                rel="noopener noreferrer"
              >
                <Github className="mr-2 h-4 w-4" />
                View on GitHub
              </Link>
            </Button>
            <Button variant="outline" size="lg" asChild>
              <Link href="#architecture">
                Explore Architecture
                <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          </div>

          {/* Stats row */}
          <div className="mt-20 grid w-full max-w-3xl grid-cols-2 gap-6 lg:grid-cols-4">
            {stats.map((stat) => (
              <div
                key={stat.label}
                className="flex flex-col items-center gap-2 rounded-lg border border-border/50 bg-card/50 p-6 backdrop-blur-sm"
              >
                <stat.icon className="h-5 w-5 text-primary" />
                <span className="font-mono text-2xl font-bold text-foreground">
                  {stat.value}
                </span>
                <span className="text-sm text-muted-foreground">
                  {stat.label}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
