pub mod reranker {
    tonic::include_proto!("reranker");
}
pub use reranker::*;
pub use reranker::reranker_server::{Reranker, RerankerServer};
