//! HTTP server for metrics and health checks.

use std::sync::Arc;
use tokio::sync::RwLock;

use crate::metrics::MetricsRegistry;

/// Server state
pub struct ServerState {
    /// Metrics registry
    pub metrics: Arc<MetricsRegistry>,
    /// Application status
    pub status: RwLock<AppStatus>,
}

/// Application status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppStatus {
    /// Starting up
    Starting,
    /// Running normally
    Running,
    /// Degraded but operational
    Degraded,
    /// Shutting down
    ShuttingDown,
    /// Stopped
    Stopped,
}

impl ServerState {
    /// Create a new server state
    #[must_use]
    pub fn new(metrics: Arc<MetricsRegistry>) -> Self {
        Self {
            metrics,
            status: RwLock::new(AppStatus::Starting),
        }
    }

    /// Set application status
    pub async fn set_status(&self, status: AppStatus) {
        let mut s = self.status.write().await;
        *s = status;
    }

    /// Get application status
    pub async fn get_status(&self) -> AppStatus {
        *self.status.read().await
    }

    /// Check if application is healthy
    pub async fn is_healthy(&self) -> bool {
        matches!(
            self.get_status().await,
            AppStatus::Running | AppStatus::Starting
        )
    }
}

/// Health check response
#[derive(Debug, serde::Serialize)]
pub struct HealthResponse {
    /// Status string
    pub status: &'static str,
    /// Application version
    pub version: &'static str,
    /// Uptime in seconds
    pub uptime_secs: u64,
}

/// Start the metrics server
pub async fn start_metrics_server(state: Arc<ServerState>, port: u16) -> anyhow::Result<()> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    let listener = TcpListener::bind(format!("0.0.0.0:{port}")).await?;
    tracing::info!("Metrics server listening on port {}", port);

    loop {
        let (mut socket, _) = listener.accept().await?;
        let state = state.clone();

        tokio::spawn(async move {
            let mut buffer = [0u8; 1024];
            if socket.read(&mut buffer).await.is_err() {
                return;
            }

            let request = String::from_utf8_lossy(&buffer);
            let response = if request.contains("GET /metrics") {
                let metrics = state.metrics.encode();
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\r\n{}",
                    metrics.len(),
                    metrics
                )
            } else if request.contains("GET /health") {
                let healthy = state.is_healthy().await;
                let status = if healthy { "ok" } else { "unhealthy" };
                let code = if healthy { 200 } else { 503 };
                let body = format!(r#"{{"status":"{status}","version":"0.1.0"}}"#);
                format!(
                    "HTTP/1.1 {} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                    code,
                    body.len(),
                    body
                )
            } else {
                "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n".to_string()
            };

            let _ = socket.write_all(response.as_bytes()).await;
        });
    }
}
