"use client"

import type { RiskState } from "@/lib/mock-data"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import { ShieldCheck, ShieldAlert, AlertTriangle } from "lucide-react"

interface RiskPanelProps {
  riskState: RiskState
}

export function RiskPanel({ riskState }: RiskPanelProps) {
  const positionPct =
    (Math.abs(riskState.positionSize) / riskState.positionLimit) * 100
  const ddPct = (riskState.currentDrawdown / riskState.maxDrawdown) * 100
  const skewPct = (riskState.inventorySkew + 1) * 50 // -1..1 -> 0..100

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between pb-2">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Risk Management
        </h3>
        {riskState.killSwitchTripped ? (
          <Badge variant="destructive" className="gap-1 text-[10px]">
            <ShieldAlert className="h-3 w-3" /> TRIPPED
          </Badge>
        ) : riskState.killSwitchActive ? (
          <Badge
            variant="outline"
            className="gap-1 border-green-500/30 text-[10px] text-green-400"
          >
            <ShieldCheck className="h-3 w-3" /> ARMED
          </Badge>
        ) : (
          <Badge variant="outline" className="gap-1 text-[10px] text-muted-foreground">
            DISABLED
          </Badge>
        )}
      </div>

      <div className="flex flex-col gap-3">
        {/* Position */}
        <div>
          <div className="flex items-center justify-between pb-1">
            <span className="text-[10px] uppercase text-muted-foreground">
              Position
            </span>
            <span className="font-mono text-xs text-foreground">
              {riskState.positionSize} /{" "}
              <span className="text-muted-foreground">
                {riskState.positionLimit}
              </span>
            </span>
          </div>
          <Progress
            value={Math.min(positionPct, 100)}
            className="h-2"
          />
        </div>

        {/* Drawdown */}
        <div>
          <div className="flex items-center justify-between pb-1">
            <span className="text-[10px] uppercase text-muted-foreground">
              Drawdown
            </span>
            <span className="font-mono text-xs text-red-400">
              {riskState.currentDrawdown.toFixed(2)}% /{" "}
              <span className="text-muted-foreground">
                {riskState.maxDrawdown}%
              </span>
            </span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-secondary">
            <div
              className="h-full rounded-full transition-all duration-300"
              style={{
                width: `${Math.min(ddPct, 100)}%`,
                backgroundColor: ddPct > 80 ? "#ef4444" : ddPct > 50 ? "#eab308" : "#22c55e",
              }}
            />
          </div>
        </div>

        {/* Inventory skew gauge */}
        <div>
          <div className="flex items-center justify-between pb-1">
            <span className="text-[10px] uppercase text-muted-foreground">
              Inventory Skew
            </span>
            <span className="font-mono text-xs text-foreground">
              {riskState.inventorySkew.toFixed(3)}
            </span>
          </div>
          <div className="relative h-2 overflow-hidden rounded-full bg-secondary">
            {/* Center marker */}
            <div className="absolute left-1/2 top-0 h-full w-px bg-muted-foreground/50" />
            <div
              className="absolute top-0 h-full rounded-full bg-primary transition-all duration-300"
              style={{
                left: skewPct < 50 ? `${skewPct}%` : "50%",
                width: `${Math.abs(skewPct - 50)}%`,
              }}
            />
          </div>
          <div className="flex justify-between pt-0.5">
            <span className="text-[9px] text-muted-foreground">Short</span>
            <span className="text-[9px] text-muted-foreground">Long</span>
          </div>
        </div>

        {/* Exposure summary */}
        <div className="rounded border border-border bg-secondary/30 p-2">
          <p className="pb-1.5 text-[10px] uppercase tracking-wider text-muted-foreground">
            Exposure
          </p>
          <div className="flex flex-col gap-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-muted-foreground">Long</span>
              <span className="font-mono text-[10px] text-green-400">
                ${(riskState.longExposure / 1000).toFixed(0)}K
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-muted-foreground">Short</span>
              <span className="font-mono text-[10px] text-red-400">
                ${(riskState.shortExposure / 1000).toFixed(0)}K
              </span>
            </div>
            <div className="flex items-center justify-between border-t border-border pt-1">
              <span className="text-[10px] font-medium text-muted-foreground">
                Net
              </span>
              <span
                className={`font-mono text-[10px] font-semibold ${riskState.netExposure >= 0 ? "text-green-400" : "text-red-400"}`}
              >
                ${(riskState.netExposure / 1000).toFixed(0)}K
              </span>
            </div>
          </div>
        </div>

        {/* Risk Alerts */}
        <div className="flex flex-1 flex-col min-h-0">
          <p className="pb-1 text-[10px] uppercase tracking-wider text-muted-foreground">
            Alerts
          </p>
          <ScrollArea className="flex-1 max-h-32">
            {riskState.alerts.length === 0 ? (
              <p className="py-2 text-center text-[10px] text-muted-foreground">
                No active alerts
              </p>
            ) : (
              <div className="flex flex-col gap-1">
                {[...riskState.alerts].reverse().map((alert) => (
                  <div
                    key={alert.id}
                    className="flex items-start gap-1.5 rounded bg-secondary/50 px-2 py-1"
                  >
                    <AlertTriangle
                      className={`mt-0.5 h-3 w-3 shrink-0 ${
                        alert.level === "critical"
                          ? "text-red-400"
                          : alert.level === "warning"
                            ? "text-yellow-400"
                            : "text-muted-foreground"
                      }`}
                    />
                    <span className="text-[10px] leading-tight text-foreground">
                      {alert.message}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </ScrollArea>
        </div>
      </div>
    </div>
  )
}
