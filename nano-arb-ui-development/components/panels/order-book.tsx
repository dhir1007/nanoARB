"use client"

import type { OrderBook } from "@/lib/mock-data"
import { ScrollArea } from "@/components/ui/scroll-area"
import { PanelInfo, PANEL_INFO } from "@/components/panel-info"

interface OrderBookPanelProps {
  orderBook: OrderBook
}

export function OrderBookPanel({ orderBook }: OrderBookPanelProps) {
  const maxSize = Math.max(
    ...orderBook.bids.map((b) => b.size),
    ...orderBook.asks.map((a) => a.size)
  )

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between pb-2">
        <div className="flex items-center gap-1.5">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Order Book
          </h3>
          <PanelInfo {...PANEL_INFO.orderBook} />
        </div>
        <span className="font-mono text-xs text-muted-foreground">
          Spread:{" "}
          <span className="text-primary">{orderBook.spread.toFixed(2)}</span>
        </span>
      </div>

      {/* Header */}
      <div className="grid grid-cols-4 gap-1 border-b border-border pb-1 text-[10px] uppercase tracking-wider text-muted-foreground">
        <span className="text-right">Orders</span>
        <span className="text-right">Size</span>
        <span className="text-right">Price</span>
        <span>Depth</span>
      </div>

      <ScrollArea className="flex-1">
        {/* Asks (reversed so lowest ask is at bottom) */}
        <div className="flex flex-col">
          {[...orderBook.asks].reverse().map((level, i) => (
            <div
              key={`ask-${i}`}
              className="group relative grid grid-cols-4 items-center gap-1 py-[3px] font-mono text-xs"
            >
              <span className="text-right text-muted-foreground">
                {level.orders}
              </span>
              <span className="text-right text-red-400">{level.size}</span>
              <span className="text-right font-medium text-red-400">
                {level.price.toFixed(2)}
              </span>
              <div className="relative h-3 overflow-hidden rounded-sm bg-secondary">
                <div
                  className="absolute inset-y-0 left-0 rounded-sm bg-red-500/25"
                  style={{ width: `${(level.size / maxSize) * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>

        {/* Spread indicator */}
        <div className="flex items-center justify-center gap-2 border-y border-primary/20 py-1.5">
          <span className="font-mono text-sm font-bold text-primary">
            {orderBook.midPrice.toFixed(2)}
          </span>
        </div>

        {/* Bids */}
        <div className="flex flex-col">
          {orderBook.bids.map((level, i) => (
            <div
              key={`bid-${i}`}
              className="group relative grid grid-cols-4 items-center gap-1 py-[3px] font-mono text-xs"
            >
              <span className="text-right text-muted-foreground">
                {level.orders}
              </span>
              <span className="text-right text-green-400">{level.size}</span>
              <span className="text-right font-medium text-green-400">
                {level.price.toFixed(2)}
              </span>
              <div className="relative h-3 overflow-hidden rounded-sm bg-secondary">
                <div
                  className="absolute inset-y-0 left-0 rounded-sm bg-green-500/25"
                  style={{ width: `${(level.size / maxSize) * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </ScrollArea>
    </div>
  )
}
