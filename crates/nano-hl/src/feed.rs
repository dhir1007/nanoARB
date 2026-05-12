//! Hyperliquid WebSocket feed.
//!
//! Connects to HL, subscribes to l2Book updates, and forwards raw
//! messages to a channel for the engine to consume.

use anyhow::{Context, Result};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};

// ── WebSocket URLs ──────────────────────────────────────────────────────────

pub const HL_TESTNET_WS: &str = "wss://api.hyperliquid-testnet.xyz/ws";
pub const HL_MAINNET_WS: &str = "wss://api.hyperliquid.xyz/ws";

/// How often to send a heartbeat ping (HL disconnects idle connections after 60s)
const HEARTBEAT_SECS: u64 = 20;

// ── Raw message types from Hyperliquid ─────────────────────────────────────

/// A single price level in the HL order book.
/// px and sz come as strings because HL doesn't use numbers in JSON for prices.
#[derive(Debug, Clone, Deserialize)]
pub struct HlLevel {
    pub px: String, // price, e.g. "99500.5"
    pub sz: String, // size,  e.g. "0.500"
    pub n: u32,     // number of orders at this level
}

/// The "data" field inside an l2Book message.
#[derive(Debug, Clone, Deserialize)]
pub struct HlBookData {
    pub coin: String,
    /// levels[0] = bids (descending), levels[1] = asks (ascending)
    pub levels: (Vec<HlLevel>, Vec<HlLevel>),
    pub time: u64, // milliseconds since epoch
}

/// Top-level HL WebSocket message envelope.
#[derive(Debug, Deserialize)]
pub struct HlMessage {
    pub channel: String,
    pub data: serde_json::Value,
}

// ── Feed connection ─────────────────────────────────────────────────────────

/// Connect to Hyperliquid and stream l2Book updates for `coin` onto `tx`.
/// This function runs forever (reconnecting on error) until the receiver is dropped.
pub async fn run_feed(
    coin: String,
    testnet: bool,
    tx: mpsc::Sender<HlBookData>,
) {
    loop {
        let url = if testnet { HL_TESTNET_WS } else { HL_MAINNET_WS };
        tracing::info!(coin = %coin, testnet, url, "Connecting to Hyperliquid WebSocket");

        match connect_once(&coin, url, &tx).await {
            Ok(()) => {
                tracing::warn!("WebSocket stream ended cleanly, reconnecting in 2s");
            }
            Err(e) => {
                tracing::error!("WebSocket error: {e:#}, reconnecting in 2s");
            }
        }

        // If the receiver side closed (engine shut down), stop reconnecting
        if tx.is_closed() {
            tracing::info!("Feed receiver dropped, shutting down feed task");
            return;
        }

        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    }
}

/// Single connection attempt. Returns Ok(()) when stream ends cleanly.
async fn connect_once(
    coin: &str,
    url: &str,
    tx: &mpsc::Sender<HlBookData>,
) -> Result<()> {
    let (ws_stream, _) = connect_async(url)
        .await
        .context("WebSocket connect failed")?;

    let (mut write, mut read) = ws_stream.split();

    // Subscribe to l2Book
    let sub = serde_json::json!({
        "method": "subscribe",
        "subscription": { "type": "l2Book", "coin": coin }
    });
    write
        .send(Message::Text(sub.to_string()))
        .await
        .context("Failed to send subscription")?;

    tracing::info!(coin, "Subscribed to l2Book");

    // Heartbeat task: send a ping every HEARTBEAT_SECS seconds
    let mut heartbeat = tokio::time::interval(std::time::Duration::from_secs(HEARTBEAT_SECS));
    heartbeat.tick().await; // consume the immediate first tick

    loop {
        tokio::select! {
            msg = read.next() => {
                match msg {
                    None => {
                        tracing::warn!("WebSocket stream closed by server");
                        return Ok(());
                    }
                    Some(Err(e)) => return Err(e.into()),
                    Some(Ok(Message::Text(text))) => {
                        handle_text_message(&text, tx).await;
                    }
                    Some(Ok(Message::Ping(data))) => {
                        write.send(Message::Pong(data)).await.ok();
                    }
                    Some(Ok(Message::Close(_))) => {
                        tracing::info!("Server sent Close frame");
                        return Ok(());
                    }
                    _ => {} // binary / pong / etc.
                }
            }
            _ = heartbeat.tick() => {
                // Send a ping to keep connection alive
                write.send(Message::Ping(vec![])).await.ok();
                tracing::trace!("Sent heartbeat ping");
            }
        }
    }
}

/// Parse a raw text message and forward l2Book updates.
async fn handle_text_message(text: &str, tx: &mpsc::Sender<HlBookData>) {
    let msg: HlMessage = match serde_json::from_str(text) {
        Ok(m) => m,
        Err(e) => {
            tracing::trace!("Ignoring unparseable message: {e}");
            return;
        }
    };

    if msg.channel != "l2Book" {
        // subscriptionResponse, pong, etc. — ignore
        return;
    }

    let book_data: HlBookData = match serde_json::from_value(msg.data) {
        Ok(d) => d,
        Err(e) => {
            tracing::warn!("Failed to parse l2Book data: {e}");
            return;
        }
    };

    if tx.send(book_data).await.is_err() {
        tracing::debug!("Feed channel closed");
    }
}