use std::sync::{Arc, Mutex};
use tonic::transport::Server;
use arbiter::reranker::model::RerankerModel;
use arbiter::reranker::proto::RerankerServer;
use arbiter::reranker::service::RerankerService;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing reranker model and device...");
    // Initialize the reranker model.
    // Using cross-encoder/ms-marco-MiniLM-L-6-v2 - fast and good quality for semantic search reranking.
    // Alternatives:
    // - "cross-encoder/ms-marco-MiniLM-L-12-v2" for better quality
    // - "cross-encoder/ms-marco-TinyBERT-L-2-v2" for faster inference
    let model = RerankerModel::new("cross-encoder/ms-marco-MiniLM-L-6-v2")?;
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

    let addr = "[::1]:50052".parse()?;
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