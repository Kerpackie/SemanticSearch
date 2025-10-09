fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compile the helloworld service
    tonic_prost_build::compile_protos("proto/helloworld.proto")?;

    // Compile the health service
    tonic_prost_build::compile_protos("proto/health.proto")?;

    tonic_prost_build::compile_protos("proto/embedding.proto")?;

    Ok(())
}