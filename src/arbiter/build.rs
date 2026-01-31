fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compile the helloworld service
    tonic_prost_build::compile_protos("proto/helloworld.proto")?;

    // Compile the health service
    tonic_prost_build::compile_protos("proto/health.proto")?;

    // Compile the reranker service
    tonic_prost_build::compile_protos("proto/reranker.proto")?;

    Ok(())
}