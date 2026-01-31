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
    public byte[]? ImageBytes { get; set; }
}

class Program
{
    const string CsvPath = "/Users/kerpackie/RiderProjects/SemanticSearch/DatabaseSeeder/Data/articles.csv";
    const string ImagesRootPath = "/Users/kerpackie/Downloads/h-and-m-personalized-fashion-recommendations/images"; 
    
    const string TextEmbedderUrl = "http://localhost:50051"; 
    const string ImageEmbedderUrl = "http://localhost:50053"; 
    const string MnemeUrl = "http://localhost:5074"; 
    
    // --- CONFIGURATION ---
    const int ProcessingBatchSize = 50; 
    const int MaxConcurrentBatches = 5;

    // SKIP CONFIGURATION: Change this to resume processing
    // e.g. Set to 5000 to skip the first 5000 rows
    const int StartFromIndex = 88150; 

    static SemaphoreSlim _concurrencyLimit = new SemaphoreSlim(MaxConcurrentBatches);
    static int _totalProcessed = 0;
    static object _lock = new object();

    static async Task Main(string[] args)
    {
        Console.WriteLine($"Starting Parallel Batch Ingestion (Max {MaxConcurrentBatches} concurrent batches)...");
        
        if (StartFromIndex > 0)
        {
            Console.WriteLine($"[RESUME MODE] Skipping first {StartFromIndex} records...");
        }

        using var textChannel = GrpcChannel.ForAddress(TextEmbedderUrl);
        using var imageChannel = GrpcChannel.ForAddress(ImageEmbedderUrl);
        using var mnemeChannel = GrpcChannel.ForAddress(MnemeUrl);

        var textClient = new Embedder.Api.Embedder.EmbedderClient(textChannel);
        var imageClient = new ClipEmbedder.Api.ClipEmbedder.ClipEmbedderClient(imageChannel);
        var mnemeClient = new ProductSearch.ProductSearchClient(mnemeChannel);

        await ProcessPipeline(textClient, imageClient, mnemeClient);
        
        Console.WriteLine("\nIngestion Complete!");
    }

    static async Task ProcessPipeline(
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
        var activeTasks = new List<Task>();
        
        // Counter to track our position in the CSV
        int globalIndex = 0; 

        await foreach (var row in records)
        {
            // 1. Check if we need to skip this row
            if (globalIndex < StartFromIndex)
            {
                globalIndex++;
                // Optional: Print a dot every 1000 skips so you know it's working
                if (globalIndex % 1000 == 0) Console.Write("."); 
                continue;
            }

            if (string.IsNullOrWhiteSpace(row.DetailDesc)) 
            {
                // Even if we skip due to bad data, we increment index to keep count accurate to the file rows
                globalIndex++;
                continue;
            }

            var p = new ProductState
            {
                Id = row.ArticleId,
                Name = row.ProdName,
                Description = row.DetailDesc,
                Metadata = new Dictionary<string, string>
                {
                    { "color", row.Color },
                    { "type", row.Type },
                    { "pattern", row.Pattern },
                    { "group", row.Group }
                }
            };
            currentBatch.Add(p);
            globalIndex++;

            if (currentBatch.Count >= ProcessingBatchSize)
            {
                await _concurrencyLimit.WaitAsync();

                var batchToProcess = new List<ProductState>(currentBatch);
                currentBatch.Clear();

                activeTasks.RemoveAll(t => t.IsCompleted);

                activeTasks.Add(Task.Run(async () => 
                {
                    try 
                    {
                        await ProcessBatch(batchToProcess, textClient, imageClient, mnemeClient);
                    }
                    finally 
                    {
                        _concurrencyLimit.Release();
                    }
                }));
            }
        }

        if (currentBatch.Count > 0)
        {
             await ProcessBatch(currentBatch, textClient, imageClient, mnemeClient);
        }

        await Task.WhenAll(activeTasks);
    }

    static async Task ProcessBatch(
        List<ProductState> batch,
        Embedder.Api.Embedder.EmbedderClient textClient,
        ClipEmbedder.Api.ClipEmbedder.ClipEmbedderClient imageClient,
        ProductSearch.ProductSearchClient mnemeClient)
    {
        // 0. Load Images
        foreach (var p in batch)
        {
            string folderPrefix = p.Id.Length >= 3 ? p.Id.Substring(0, 3) : "000";
            string imagePath = Path.Combine(ImagesRootPath, folderPrefix, $"{p.Id}.jpg");
            if (File.Exists(imagePath))
            {
                try {
                    p.ImageBytes = await File.ReadAllBytesAsync(imagePath);
                    p.HasImage = true;
                } catch { }
            }
        }

        // 1. Text Embeddings
        try 
        {
            using var textCall = textClient.IndexTexts();
            var readTask = ReadTextResponses(textCall.ResponseStream, batch);
            
            foreach (var p in batch)
            {
                var fullText = $"{p.Name}. {p.Description}. Color: {p.Metadata["color"]}. Type: {p.Metadata["type"]}.";
                await textCall.RequestStream.WriteAsync(new Embedder.Api.IndexRequest { DocumentId = p.Id, Text = fullText });
            }
            await textCall.RequestStream.CompleteAsync();
            await readTask;
        }
        catch (RpcException ex) 
        { 
            Console.WriteLine($"\n[Text Error] {ex.Status}");
            return; 
        }

        // 2. Image Embeddings
        var imagesToEmbed = batch.Where(p => p.HasImage && p.ImageBytes != null).ToList();
        if (imagesToEmbed.Any())
        {
            try
            {
                using var imageCall = imageClient.IndexImages();
                var readTask = ReadImageResponses(imageCall.ResponseStream, batch);
                
                foreach (var p in imagesToEmbed)
                {
                    await imageCall.RequestStream.WriteAsync(new ClipEmbedder.Api.IndexImageRequest 
                    { 
                        DocumentId = p.Id, 
                        Image = ByteString.CopyFrom(p.ImageBytes!) 
                    });
                }
                await imageCall.RequestStream.CompleteAsync();
                await readTask;
            }
            catch (RpcException ex) { Console.WriteLine($"\n[Image Error] {ex.Status}"); }
        }

        // 3. Upload
        var uploadList = new List<Product>();
        foreach (var p in batch)
        {
            if (p.TextVector != null) 
            {
                uploadList.Add(MapToProto(p));
            }
        }

        if (uploadList.Count > 0)
        {
            try
            {
                var req = new UpsertProductsRequest();
                req.Products.AddRange(uploadList);
                await mnemeClient.UpsertProductsAsync(req);
                
                lock(_lock)
                {
                    _totalProcessed += uploadList.Count;
                    // UPDATED: Print the total index relative to the file, not just this run
                    Console.Write($"\rIndexed {_totalProcessed + StartFromIndex} products (Offset: {StartFromIndex}, Run: {_totalProcessed})...");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\nBatch Upload Error: {ex.Message}");
            }
        }
    }

    // ... (ReadTextResponses, ReadImageResponses, MapToProto, GenerateUuidFromStr same as before)
    static async Task ReadTextResponses(IAsyncStreamReader<Embedder.Api.IndexResponse> stream, List<ProductState> batch)
    {
        var batchMap = batch.ToDictionary(p => p.Id);
        await foreach (var resp in stream.ReadAllAsync())
        {
            if (resp.Success && batchMap.TryGetValue(resp.DocumentId, out var p))
                p.TextVector = resp.Embedding.Values.ToArray();
        }
    }

    static async Task ReadImageResponses(IAsyncStreamReader<ClipEmbedder.Api.IndexResponse> stream, List<ProductState> batch)
    {
        var batchMap = batch.ToDictionary(p => p.Id);
        await foreach (var resp in stream.ReadAllAsync())
        {
            if (resp.Success && batchMap.TryGetValue(resp.DocumentId, out var p))
                p.ImageVector = resp.Embedding.Values.ToArray();
        }
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