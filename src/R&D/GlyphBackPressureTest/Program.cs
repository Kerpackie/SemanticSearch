using System.Diagnostics;
using Embedder;
using Grpc.Core;
using Grpc.Net.Client;

namespace GlyphBackPressureTest;

public class Program
{
    private const int TotalRequests = 100_000;
    private const string ServerAddress = "http://localhost:50051";

    public static async Task Main(string[] args)
    {
        Console.WriteLine($"Starting gRPC backpressure test against {ServerAddress}");
        Console.WriteLine($"Will attempt to send {TotalRequests:N0} documents.");

        // Create a channel and client
        using var channel = GrpcChannel.ForAddress(ServerAddress);
        var client = new Embedder.Embedder.EmbedderClient(channel);

        // Start the bi-directional streaming call
        using var call = client.IndexTexts();

        // --- Task 1: Listen for responses from the server ---
        long responsesReceived = 0;
        var stopwatch = Stopwatch.StartNew();
        
        var responseTask = Task.Run(async () =>
        {
            try
            {
                await foreach (var response in call.ResponseStream.ReadAllAsync())
                {
                    responsesReceived++;
                    if (responsesReceived % 1000 == 0)
                    {
                        var elapsedSeconds = stopwatch.Elapsed.TotalSeconds;
                        var rate = responsesReceived / elapsedSeconds;
                        Console.WriteLine($"[RECEIVER] Received {responsesReceived:N0} responses. Rate: {rate:F2}/sec");
                    }
                }
            }
            catch (RpcException ex) when (ex.StatusCode == StatusCode.Cancelled)
            {
                Console.WriteLine("[RECEIVER] Stream cancelled by the server or client.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[RECEIVER] An error occurred: {ex.Message}");
            }
        });

        // --- Task 2: Send requests to the server as fast as possible ---
        var requestTask = Task.Run(async () =>
        {
            Console.WriteLine("[SENDER] Starting to send requests...");
            for (int i = 0; i < TotalRequests; i++)
            {
                var request = new IndexRequest
                {
                    DocumentId = $"doc_{i + 1}",
                    Text = $"This is the content for document number {i + 1}. It is a sample text to test the embedding model."
                };
                
                try
                {
                    // We don't await here in the loop to send as fast as possible.
                    // The client library buffers requests and sends them over the stream.
                    // When the buffer is full, this call will implicitly wait (apply backpressure).
                    await call.RequestStream.WriteAsync(request);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[SENDER] Failed to write to stream: {ex.Message}");
                    break;
                }
            }
            
            // Signal that we're done sending requests.
            await call.RequestStream.CompleteAsync();
            Console.WriteLine($"[SENDER] Finished sending all {TotalRequests:N0} requests.");
        });

        // Wait for both tasks to complete
        await Task.WhenAll(requestTask, responseTask);

        stopwatch.Stop();
        Console.WriteLine("\n--- Test Finished ---");
        Console.WriteLine($"Successfully sent and received {responsesReceived:N0} documents.");
        Console.WriteLine($"Total time: {stopwatch.Elapsed.TotalSeconds:F2} seconds.");
        var finalRate = responsesReceived / stopwatch.Elapsed.TotalSeconds;
        Console.WriteLine($"Average throughput: {finalRate:F2} docs/sec.");
    }
}