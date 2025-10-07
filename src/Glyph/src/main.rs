mod utils;
pub mod embedder;

use std::sync::Arc;
use crate::embedder::model::EmbeddingModel;
use crate::embedder::service::EmbedderService;
use crate::embedder::proto::EmbedderServer;
use tonic::transport::Server;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing model and device...");
    let model = EmbeddingModel::new("BAAI/bge-base-en-v1.5")?;
    println!(
        "Model loaded successfully on device: {:?}.",
        model.device.location()
    );
    let shared_model = Arc::new(model);
    let embedder_service = EmbedderService {
        model: shared_model,
    };
    let addr = "[::1]:50051".parse()?;
    println!("gRPC EmbedderServer listening on {}", addr);
    let (health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_serving::<EmbedderServer<EmbedderService>>()
        .await;
    Server::builder()
        .add_service(EmbedderServer::new(embedder_service))
        .add_service(health_service)
        .serve(addr)
        .await?;

    Ok(())
}