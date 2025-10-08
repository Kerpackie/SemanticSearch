# --- Stage 1: Build ---
# This stage compiles the Rust application.
FROM rust:1.77.2-slim-bookworm AS builder

# Install necessary build dependencies.
# - build-essential: for C compilers, etc.
# - protobuf-compiler: needed for tonic-build.
# - libssl-dev: for networking dependencies.
RUN apt-get update && apt-get install -y build-essential protobuf-compiler libssl-dev

# Set the working directory.
WORKDIR /usr/src/app

# Create a new dummy project to cache dependencies.
# This layer is only rebuilt when Cargo.toml or Cargo.lock changes.
RUN cargo init --bin
COPY Cargo.toml Cargo.lock ./
# Build dependencies only.
RUN cargo build --release
# Cleanup the dummy source file.
RUN rm src/*.rs

# Copy the actual application source code.
COPY . .

# Build the application for release.
RUN cargo build --release

# --- Stage 2: Runtime ---
# This stage creates the final, minimal image.
FROM debian:bookworm-slim AS runtime

# Set the working directory for the application binary.
WORKDIR /usr/local/bin

# Install runtime dependencies (OpenSSL).
RUN apt-get update && apt-get install -y libssl3 && rm -rf /var/lib/apt/lists/*

# Copy the compiled binary from the 'builder' stage.
COPY --from=builder /usr/src/app/target/release/Glyph .

# Set up the Hugging Face cache directory inside the container.
# Using an environment variable is the standard way to configure this.
ENV HUGGING_FACE_HUB_CACHE=/huggingface/cache
RUN mkdir -p /huggingface/cache

# Expose the gRPC port.
EXPOSE 50051

# Set the command to run the server.
CMD ["./Glyph"]