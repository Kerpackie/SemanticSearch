pub mod embedder {
    tonic::include_proto!("embedder");
}
pub use embedder::*;
pub use embedder::embedder_server::{Embedder, EmbedderServer};

