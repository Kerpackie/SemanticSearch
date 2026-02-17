use std::env;
use std::sync::{Arc, Mutex};
use tonic::transport::Server;
use arbiter::reranker::model::RerankerModel;
use arbiter::reranker::proto::RerankerServer;
use arbiter::reranker::service::RerankerService;
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing subscriber for structured logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .with_thread_ids(false)
        .with_line_number(true)
        .init();

    tracing::info!("Initializing reranker model and device...");
    println!("Initializing reranker model and device...");
    // Initialize the reranker model.
    // Default: cross-encoder/ms-marco-MiniLM-L-12-v2 - good quality with better score calibration.
    // Override via RERANKER_MODEL env var.
    // Alternatives:
    // - "cross-encoder/ms-marco-MiniLM-L-6-v2" for faster inference
    // - "cross-encoder/ms-marco-TinyBERT-L-2-v2" for very fast inference
    let model_id = env::var("RERANKER_MODEL")
        .unwrap_or_else(|_| "cross-encoder/ms-marco-MiniLM-L-12-v2".to_string());
    tracing::info!(model_id = %model_id, "Loading reranker model");
    let model = RerankerModel::new(&model_id)?;
    tracing::info!(
        device = ?model.device.location(),
        "Reranker model loaded successfully"
    );
    println!(
        "Reranker model loaded successfully on device: {:?}.",
        model.device.location()
    );

    // Wrap the model in a standard Mutex and an Arc for safe, shared access across threads.
    let shared_model = Arc::new(Mutex::new(model));

    // Create the service instance, passing the shared model.
    let reranker_service = RerankerService {
        model: shared_model,
    };

    // Read port from GRPC_PORT env var (set by Aspire) or default to 50053
    let port = env::var("GRPC_PORT").unwrap_or_else(|_| "50053".to_string());
    let addr = format!("[::1]:{}", port).parse()?;
    tracing::info!(
        address = %addr,
        port = %port,
        "gRPC RerankerServer starting"
    );
    println!("gRPC RerankerServer listening on {}", addr);

    // Set up the gRPC health checking service.
    let (health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_serving::<RerankerServer<RerankerService>>()
        .await;

    // Build and run the gRPC server.
    Server::builder()
        .add_service(RerankerServer::new(reranker_service))
        .add_service(health_service)
        .serve(addr)
        .await?;

    Ok(())
}