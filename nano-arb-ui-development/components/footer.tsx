import Link from "next/link"
import { Zap, Github } from "lucide-react"

export function Footer() {
  return (
    <footer className="border-t border-border/50 py-12">
      <div className="mx-auto max-w-7xl px-6">
        <div className="flex flex-col items-center justify-between gap-6 sm:flex-row">
          <div className="flex items-center gap-2">
            <div className="flex h-7 w-7 items-center justify-center rounded-md bg-primary">
              <Zap className="h-3.5 w-3.5 text-primary-foreground" />
            </div>
            <span className="text-sm font-semibold text-foreground">
              NanoARB
            </span>
          </div>

          <p className="text-center text-xs leading-relaxed text-muted-foreground sm:text-left">
            Open-source HFT market-making engine. For educational and research
            purposes only. Not financial advice.
          </p>

          <Link
            href="https://github.com/dhir1007/nanoARB"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-sm text-muted-foreground transition-colors hover:text-foreground"
          >
            <Github className="h-4 w-4" />
            <span>Source Code</span>
          </Link>
        </div>
      </div>
    </footer>
  )
}
