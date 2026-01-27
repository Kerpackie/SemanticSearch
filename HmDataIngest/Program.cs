using System.Collections.Concurrent;
using System.Globalization;
using CsvHelper;
using CsvHelper.Configuration;
using Embedder;
using Embedder.Api;
using Grpc.Core;
using Grpc.Net.Client;
using Mneme.Api;

namespace HmDataIngest;

class Program
{
    const string CsvPath = "/Users/kerpackie/RiderProjects/SemanticSearch/DatabaseSeeder/Data/articles.csv";
    const string EmbedderUrl = "http://localhost:50051"; 
    const string MnemeUrl = "http://localhost:5074";    
    const int MnemeBatchSize = 50;

    static ConcurrentDictionary<string, ArticleCsv> _pendingDocs = new();

    static async Task Main(string[] args)
    {
        Console.WriteLine("Starting H&M Ingestion Pipeline with Enhanced Metadata...");

        using var embedderChannel = GrpcChannel.ForAddress(EmbedderUrl);
        var embedderClient = new Embedder.Api.Embedder.EmbedderClient(embedderChannel);

        using var mnemeChannel = GrpcChannel.ForAddress(MnemeUrl);
        var mnemeClient = new ProductSearch.ProductSearchClient(mnemeChannel);

        using var embeddingCall = embedderClient.IndexTexts();

        var processingTask = ProcessEmbeddingsAndUpload(embeddingCall.ResponseStream, mnemeClient);

        await StreamCsvToEmbedder(embeddingCall.RequestStream);

        await embeddingCall.RequestStream.CompleteAsync();
        await processingTask;

        Console.WriteLine("\nIngestion Complete!");
    }

    static async Task StreamCsvToEmbedder(IClientStreamWriter<IndexRequest> requestStream)
    {
        var config = new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            PrepareHeaderForMatch = args => args.Header.ToLower(),
            MissingFieldFound = null 
        };

        using var reader = new StreamReader(CsvPath);
        using var csv = new CsvReader(reader, config);

        var records = csv.GetRecordsAsync<ArticleCsv>();
        int count = 0;

        Console.WriteLine("Reading CSV...");

        await foreach (var row in records)
        {
            if (string.IsNullOrWhiteSpace(row.DetailDesc)) continue;

            // ENHANCED SEMANTIC STRING
            // We combine attributes into a natural language string.
            // Old: "Strap top. Top in jersey..."
            // New: "Strap top. Top in jersey... Color: Black. Type: Vest top. Pattern: Solid. Group: Ladieswear."
            var fullText = $"{row.ProdName}. {row.DetailDesc} " +
                           $"Color: {row.Color}. " +
                           $"Type: {row.Type}. " +
                           $"Pattern: {row.Pattern}. " +
                           $"Group: {row.Group}.";
            
            _pendingDocs.TryAdd(row.ArticleId, row);

            await requestStream.WriteAsync(new IndexRequest
            {
                DocumentId = row.ArticleId,
                Text = fullText
            });

            count++;
            if (count % 1000 == 0) Console.Write($"\rSent {count} articles to embedder...");
        }

        Console.WriteLine($"\nFinished sending {count} articles.");
    }

    static async Task ProcessEmbeddingsAndUpload(IAsyncStreamReader<IndexResponse> responseStream, ProductSearch.ProductSearchClient mnemeClient)
    {
        var batch = new List<Product>();
        int totalIndexed = 0;

        await foreach (var response in responseStream.ReadAllAsync())
        {
            if (!response.Success)
            {
                _pendingDocs.TryRemove(response.DocumentId, out _);
                continue;
            }

            if (_pendingDocs.TryRemove(response.DocumentId, out var meta))
            {
                var uuid = GenerateUuidFromStr(meta.ArticleId);

                var product = new Product
                {
                    Id = uuid.ToString(),
                    Name = meta.ProdName,
                    Description = meta.DetailDesc,
                };

                // Add vectors
                product.TextVector.AddRange(response.Embedding.Values);

                // Add Metadata (So we can filter by these later in Qdrant)
                product.Metadata.Add("color", meta.Color);
                product.Metadata.Add("type", meta.Type);
                product.Metadata.Add("pattern", meta.Pattern);
                product.Metadata.Add("group", meta.Group);
                product.Metadata.Add("original_id", meta.ArticleId);

                batch.Add(product);
            }

            if (batch.Count >= MnemeBatchSize)
            {
                await UploadBatch(mnemeClient, batch);
                totalIndexed += batch.Count;
                Console.Write($"\rIndexed {totalIndexed} products...");
                batch.Clear();
            }
        }

        if (batch.Count > 0)
        {
            await UploadBatch(mnemeClient, batch);
            totalIndexed += batch.Count;
        }
    }

    static async Task UploadBatch(ProductSearch.ProductSearchClient client, List<Product> batch)
    {
        try
        {
            var req = new UpsertProductsRequest();
            req.Products.AddRange(batch);
            await client.UpsertProductsAsync(req);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\nError: {ex.Message}");
        }
    }

    static Guid GenerateUuidFromStr(string input)
    {
        using (System.Security.Cryptography.MD5 md5 = System.Security.Cryptography.MD5.Create())
        {
            byte[] hash = md5.ComputeHash(System.Text.Encoding.Default.GetBytes(input));
            return new Guid(hash);
        }
    }
}