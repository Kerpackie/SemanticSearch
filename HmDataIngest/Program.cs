using System.Collections.Concurrent;
using System.Globalization;
using CsvHelper;
using CsvHelper.Configuration;
using Grpc.Core;
using Grpc.Net.Client;
using Google.Protobuf; 
using Embedder.Api;    
using ClipEmbedder.Api;
using Mneme.Api;       

namespace HmDataIngest;

public class ProductState
{
    public string Id { get; set; } = "";
    public string Name { get; set; } = "";
    public string Description { get; set; } = "";
    public Dictionary<string, string> Metadata { get; set; } = new();
    
    public float[]? TextVector { get; set; }
    public float[]? ImageVector { get; set; }
    public bool HasImage { get; set; } = false;
    public byte[]? ImageBytes { get; set; } // Temp storage for image data
}

class Program
{
    const string CsvPath = "/Users/kerpackie/RiderProjects/SemanticSearch/DatabaseSeeder/Data/articles.csv";
    const string ImagesRootPath = "/Users/kerpackie/Downloads/h-and-m-personalized-fashion-recommendations/images"; 
    
    const string TextEmbedderUrl = "http://localhost:50051"; 
    const string ImageEmbedderUrl = "http://localhost:50053"; 
    const string MnemeUrl = "http://localhost:5074"; 
    
    // Batch size for processing. Kept small to ensure stability.
    const int ProcessingBatchSize = 50; 

    static async Task Main(string[] args)
    {
        Console.WriteLine("Starting Sequential Batch Ingestion Pipeline...");

        using var textChannel = GrpcChannel.ForAddress(TextEmbedderUrl);
        var textClient = new Embedder.Api.Embedder.EmbedderClient(textChannel);

        using var imageChannel = GrpcChannel.ForAddress(ImageEmbedderUrl);
        var imageClient = new ClipEmbedder.Api.ClipEmbedder.ClipEmbedderClient(imageChannel);

        using var mnemeChannel = GrpcChannel.ForAddress(MnemeUrl);
        var mnemeClient = new ProductSearch.ProductSearchClient(mnemeChannel);

        await ProcessInBatches(textClient, imageClient, mnemeClient);
        
        Console.WriteLine("\nIngestion Complete!");
    }

    static async Task ProcessInBatches(
        Embedder.Api.Embedder.EmbedderClient textClient,
        ClipEmbedder.Api.ClipEmbedder.ClipEmbedderClient imageClient,
        ProductSearch.ProductSearchClient mnemeClient)
    {
        var config = new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            PrepareHeaderForMatch = args => args.Header.ToLower(),
            MissingFieldFound = null 
        };

        using var reader = new StreamReader(CsvPath);
        using var csv = new CsvReader(reader, config);
        
        var records = csv.GetRecordsAsync<ArticleCsv>();
        var currentBatch = new List<ProductState>();
        int totalProcessed = 0;

        await foreach (var row in records)
        {
            if (string.IsNullOrWhiteSpace(row.DetailDesc)) continue;

            // 1. Prepare Product State
            string folderPrefix = row.ArticleId.Length >= 3 ? row.ArticleId.Substring(0, 3) : "000";
            string imagePath = Path.Combine(ImagesRootPath, folderPrefix, $"{row.ArticleId}.jpg");
            bool imageExists = File.Exists(imagePath);

            var p = new ProductState
            {
                Id = row.ArticleId,
                Name = row.ProdName,
                Description = row.DetailDesc,
                HasImage = imageExists,
                Metadata = new Dictionary<string, string>
                {
                    { "color", row.Color },
                    { "type", row.Type },
                    { "pattern", row.Pattern },
                    { "group", row.Group }
                }
            };

            // Pre-load image bytes so we don't do file I/O during the network phase
            if (imageExists)
            {
                try { p.ImageBytes = await File.ReadAllBytesAsync(imagePath); }
                catch { p.HasImage = false; }
            }

            currentBatch.Add(p);

            // 2. Process Batch if Full
            if (currentBatch.Count >= ProcessingBatchSize)
            {
                await ProcessBatch(currentBatch, textClient, imageClient, mnemeClient);
                totalProcessed += currentBatch.Count;
                Console.Write($"\rProcessed {totalProcessed} products...");
                currentBatch.Clear();
            }
        }

        // Process remaining
        if (currentBatch.Count > 0)
        {
            await ProcessBatch(currentBatch, textClient, imageClient, mnemeClient);
            totalProcessed += currentBatch.Count;
        }
    }

    static async Task ProcessBatch(
        List<ProductState> batch,
        Embedder.Api.Embedder.EmbedderClient textClient,
        ClipEmbedder.Api.ClipEmbedder.ClipEmbedderClient imageClient,
        ProductSearch.ProductSearchClient mnemeClient)
    {
        // PHASE 1: Text Embeddings
        // We open a new stream for every batch. This is slightly less efficient but VERY robust against crashes.
        // If a stream dies, it only affects 50 items, and we retry or fail cleanly.
        using (var textCall = textClient.IndexTexts())
        {
            var responseTask = ReadTextResponses(textCall.ResponseStream, batch);
            
            foreach (var p in batch)
            {
                var fullText = $"{p.Name}. {p.Description}. Color: {p.Metadata["color"]}. Type: {p.Metadata["type"]}.";
                await textCall.RequestStream.WriteAsync(new Embedder.Api.IndexRequest { DocumentId = p.Id, Text = fullText });
            }
            await textCall.RequestStream.CompleteAsync();
            await responseTask;
        }

        // PHASE 2: Image Embeddings
        var imagesToEmbed = batch.Where(p => p.HasImage && p.ImageBytes != null).ToList();
        if (imagesToEmbed.Any())
        {
            using (var imageCall = imageClient.IndexImages())
            {
                var responseTask = ReadImageResponses(imageCall.ResponseStream, batch);
                
                foreach (var p in imagesToEmbed)
                {
                    await imageCall.RequestStream.WriteAsync(new ClipEmbedder.Api.IndexImageRequest 
                    { 
                        DocumentId = p.Id, 
                        Image = ByteString.CopyFrom(p.ImageBytes!) 
                    });
                }
                await imageCall.RequestStream.CompleteAsync();
                await responseTask;
            }
        }

        // PHASE 3: Upload to Mneme
        var uploadList = new List<Product>();
        foreach (var p in batch)
        {
            // Only upload if we at least got the text vector (mandatory)
            if (p.TextVector != null)
            {
                var proto = MapToProto(p);
                uploadList.Add(proto);
            }
        }

        if (uploadList.Count > 0)
        {
            try
            {
                var req = new UpsertProductsRequest();
                req.Products.AddRange(uploadList);
                var res = await mnemeClient.UpsertProductsAsync(req);
                if (!res.Success) Console.WriteLine($"\nBatch Upload Failed: {res.Message}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\nBatch Upload RPC Error: {ex.Message}");
            }
        }
    }

    static async Task ReadTextResponses(IAsyncStreamReader<Embedder.Api.IndexResponse> stream, List<ProductState> batch)
    {
        // Create a lookup for O(1) access
        var batchMap = batch.ToDictionary(p => p.Id);
        
        try
        {
            await foreach (var resp in stream.ReadAllAsync())
            {
                if (resp.Success && batchMap.TryGetValue(resp.DocumentId, out var p))
                {
                    p.TextVector = resp.Embedding.Values.ToArray();
                }
            }
        }
        catch (RpcException) { /* Log if needed */ }
    }

    static async Task ReadImageResponses(IAsyncStreamReader<ClipEmbedder.Api.IndexResponse> stream, List<ProductState> batch)
    {
        var batchMap = batch.ToDictionary(p => p.Id);
        
        try
        {
            await foreach (var resp in stream.ReadAllAsync())
            {
                if (resp.Success && batchMap.TryGetValue(resp.DocumentId, out var p))
                {
                    p.ImageVector = resp.Embedding.Values.ToArray();
                }
            }
        }
        catch (RpcException) { /* Log if needed */ }
    }

    static Product MapToProto(ProductState p)
    {
        var uuid = GenerateUuidFromStr(p.Id);
        var proto = new Product
        {
            Id = uuid.ToString(),
            Name = p.Name,
            Description = p.Description
        };

        if (p.TextVector != null) proto.TextVector.AddRange(p.TextVector);
        if (p.ImageVector != null) proto.ImageVector.AddRange(p.ImageVector);
        
        foreach(var kvp in p.Metadata) proto.Metadata.Add(kvp.Key, kvp.Value);

        return proto;
    }

    static Guid GenerateUuidFromStr(string input)
    {
        using (var md5 = System.Security.Cryptography.MD5.Create())
        {
            byte[] hash = md5.ComputeHash(System.Text.Encoding.Default.GetBytes(input));
            return new Guid(hash);
        }
    }
}