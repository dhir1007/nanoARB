// NanoARB HFT Engine - Shared TypeScript interfaces
// These types mirror the Rust `engine_state.rs` structs (camelCase JSON).

export interface OrderBookLevel {
  price: number
  size: number
  orders: number
}

export interface OrderBook {
  bids: OrderBookLevel[]
  asks: OrderBookLevel[]
  spread: number
  midPrice: number
}

export interface PriceTick {
  time: number
  price: number
  volume: number
  signal: "buy" | "sell" | "neutral"
  prediction: number
}

export interface Trade {
  id: string
  time: number
  side: "BUY" | "SELL"
  price: number
  qty: number
  pnl: number
  latencyUs: number
  signalSource: "ML" | "Skew" | "Spread"
}

export interface LatencySample {
  time: number
  totalUs: number
  marketDataUs: number
  mlInferenceUs: number
  quoteCalcUs: number
  orderSubmitUs: number
}

export interface RiskState {
  positionSize: number
  positionLimit: number
  currentDrawdown: number
  maxDrawdown: number
  killSwitchActive: boolean
  killSwitchTripped: boolean
  longExposure: number
  shortExposure: number
  netExposure: number
  inventorySkew: number
  alerts: RiskAlert[]
}

export interface RiskAlert {
  id: string
  time: number
  level: "info" | "warning" | "critical"
  message: string
}

export interface PerformanceMetrics {
  totalPnl: number
  sharpeRatio: number
  winRate: number
  totalTrades: number
  maxDrawdown: number
  fillRate: number
  avgTradeUs: number
}

export interface PnlPoint {
  time: number
  pnl: number
}

export interface BacktestConfig {
  symbol: string
  startDate: string
  endDate: string
  initialCapital: number
  spreadMultiplier: number
  inventoryLimit: number
  skewFactor: number
  useML: boolean
  maxDrawdown: number
  positionLimit: number
}

export interface BacktestResult {
  equityCurve: { day: number; equity: number }[]
  drawdownCurve: { day: number; drawdown: number }[]
  totalReturn: number
  sharpe: number
  maxDrawdown: number
  winRate: number
  profitFactor: number
  avgTradePnl: number
  totalTrades: number
  monthlyReturns: { month: string; return: number }[]
  tradeDistribution: { bucket: string; count: number }[]
}
