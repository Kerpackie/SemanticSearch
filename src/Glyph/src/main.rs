use std::sync::{Arc, Mutex};
use tonic::transport::Server;
use Glyph::embedder::model::EmbeddingModel;
use Glyph::embedder::proto::embedder_server::EmbedderServer;
use Glyph::embedder::service::EmbedderService;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing model and device...");
    // Initialize the embedding model.
    let model = EmbeddingModel::new("BAAI/bge-base-en-v1.5")?;
    println!(
        "Model loaded successfully on device: {:?}.",
        model.device.location()
    );

    // Wrap the model in a standard Mutex and an Arc for safe, shared access across threads.
    let shared_model = Arc::new(Mutex::new(model));

    // Create the service instance, passing the shared model.
    let embedder_service = EmbedderService {
        model: shared_model,
    };

    let addr = "[::1]:50051".parse()?;
    println!("gRPC EmbedderServer listening on {}", addr);

    // Set up the gRPC health checking service.
    let (health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_serving::<EmbedderServer<EmbedderService>>()
        .await;

    // Build and run the gRPC server.
    Server::builder()
        .add_service(EmbedderServer::new(embedder_service))
        .add_service(health_service)
        .serve(addr)
        .await?;

    Ok(())
}