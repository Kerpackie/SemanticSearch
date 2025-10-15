mod clipembedder; // Renamed from 'clipembedder'
mod utils;

use crate::clipembedder::model::ClipEmbeddingModel;
use crate::clipembedder::proto::{clip_embedder_server, ClipEmbedderServer};
use crate::clipembedder::service::ClipEmbedderService;
use std::sync::{Arc, Mutex};
use tonic::transport::Server;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Initializing CLIP model and device...");
    let model = ClipEmbeddingModel::new("openai/clip-vit-base-patch32")?;
    println!(
        "CLIP Model loaded successfully on device: {:?}.",
        model.device.location()
    );

    let shared_model = Arc::new(Mutex::new(model));

    let clip_service = ClipEmbedderService {
        model: shared_model,
    };

    let addr = "[::1]:50051".parse()?;
    println!("gRPC ClipEmbedderServer listening on {}", addr);

    let (health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_serving::<ClipEmbedderServer<ClipEmbedderService>>()
        .await;

    Server::builder()
        .add_service(ClipEmbedderServer::new(clip_service))
        .add_service(health_service)
        .serve(addr)
        .await?;

    Ok(())
}