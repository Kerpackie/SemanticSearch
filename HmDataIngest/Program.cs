using System.Collections.Concurrent;
using System.Globalization;
using System.Threading.Channels;
using CsvHelper;
using CsvHelper.Configuration;
using Grpc.Core;
using Grpc.Net.Client;
using Google.Protobuf; 
using Embedder.Api;    
using ClipEmbedder.Api;
using Mneme.Api;       

namespace HmDataIngest;

public class InflightProduct
{
    public string Id { get; set; } = "";
    public string Name { get; set; } = "";
    public string Description { get; set; } = "";
    public Dictionary<string, string> Metadata { get; set; } = new();
    
    public float[]? TextVector { get; set; }
    public float[]? ImageVector { get; set; }
    public bool HasImage { get; set; } = true;

    public bool IsComplete => TextVector != null && (!HasImage || ImageVector != null);
}

class Program
{
    // Configuration
    const string CsvPath = "/Users/kerpackie/RiderProjects/SemanticSearch/DatabaseSeeder/Data/articles.csv";
    const string ImagesRootPath = "/Users/kerpackie/RiderProjects/SemanticSearch/DatabaseSeeder/Data/images/"; 
    
    const string TextEmbedderUrl = "http://localhost:50051"; 
    const string ImageEmbedderUrl = "http://localhost:50053"; 
    const string MnemeUrl = "http://localhost:5074"; 
    const int MnemeBatchSize = 50;

    static ConcurrentDictionary<string, InflightProduct> _inflight = new();
    static Channel<Product> _uploadChannel = Channel.CreateBounded<Product>(1000);

    static async Task Main(string[] args)
    {
        Console.WriteLine("Starting Multi-Modal Ingestion Pipeline...");

        using var textChannel = GrpcChannel.ForAddress(TextEmbedderUrl);
        var textClient = new Embedder.Api.Embedder.EmbedderClient(textChannel);

        using var imageChannel = GrpcChannel.ForAddress(ImageEmbedderUrl);
        var imageClient = new ClipEmbedder.Api.ClipEmbedder.ClipEmbedderClient(imageChannel);

        using var mnemeChannel = GrpcChannel.ForAddress(MnemeUrl);
        var mnemeClient = new ProductSearch.ProductSearchClient(mnemeChannel);

        using var textCall = textClient.IndexTexts();
        using var imageCall = imageClient.IndexImages();

        var textTask = ProcessTextResponses(textCall.ResponseStream);
        var imageTask = ProcessImageResponses(imageCall.ResponseStream);
        
        var uploadTask = BatchAndUpload(mnemeClient);

        await ProcessCsv(textCall.RequestStream, imageCall.RequestStream);

        Console.WriteLine("\nFinished reading CSV. Closing streams...");
        await textCall.RequestStream.CompleteAsync();
        await imageCall.RequestStream.CompleteAsync();
        
        await Task.WhenAll(textTask, imageTask);
        
        _uploadChannel.Writer.Complete(); 
        await uploadTask;

        Console.WriteLine("\nIngestion Complete!");
    }

    static async Task ProcessCsv(
        IClientStreamWriter<Embedder.Api.IndexRequest> textStream, 
        IClientStreamWriter<ClipEmbedder.Api.IndexImageRequest> imageStream)
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

        await foreach (var row in records)
        {
            if (string.IsNullOrWhiteSpace(row.DetailDesc)) continue;

            string folderPrefix = row.ArticleId.Length >= 3 ? row.ArticleId.Substring(0, 3) : "000";
            string imagePath = Path.Combine(ImagesRootPath, folderPrefix, $"{row.ArticleId}.jpg");
            bool imageExists = File.Exists(imagePath);

            var product = new InflightProduct
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

            _inflight.TryAdd(row.ArticleId, product);

            var fullText = $"{row.ProdName}. {row.DetailDesc}. Color: {row.Color}. Type: {row.Type}. Pattern: {row.Pattern}. Group: {row.Group}.";
            await textStream.WriteAsync(new Embedder.Api.IndexRequest { DocumentId = row.ArticleId, Text = fullText });

            if (imageExists)
            {
                try 
                {
                    byte[] imageBytes = await File.ReadAllBytesAsync(imagePath);
                    await imageStream.WriteAsync(new ClipEmbedder.Api.IndexImageRequest 
                    { 
                        DocumentId = row.ArticleId, 
                        Image = ByteString.CopyFrom(imageBytes) 
                    });
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"\nError reading image {row.ArticleId}: {ex.Message}");
                    product.HasImage = false; 
                    await CheckAndDispatch(product); 
                }
            }

            count++;
            if (count % 1000 == 0) Console.Write($"\rDispatched {count} requests...");
        }
    }

    static async Task ProcessTextResponses(IAsyncStreamReader<Embedder.Api.IndexResponse> responseStream)
    {
        await foreach (var response in responseStream.ReadAllAsync())
        {
            if (!response.Success) 
            {
                Console.WriteLine($"\nText Embedding failed for {response.DocumentId}");
                _inflight.TryRemove(response.DocumentId, out _); 
                continue;
            }

            if (_inflight.TryGetValue(response.DocumentId, out var product))
            {
                product.TextVector = response.Embedding.Values.ToArray();
                await CheckAndDispatch(product);
            }
        }
    }

    static async Task ProcessImageResponses(IAsyncStreamReader<ClipEmbedder.Api.IndexResponse> responseStream)
    {
        await foreach (var response in responseStream.ReadAllAsync())
        {
            if (!response.Success) 
            {
                Console.WriteLine($"\nImage Embedding failed for {response.DocumentId}");
                if (_inflight.TryGetValue(response.DocumentId, out var failedProduct))
                {
                    failedProduct.HasImage = false; 
                    await CheckAndDispatch(failedProduct);
                }
                continue;
            }

            if (_inflight.TryGetValue(response.DocumentId, out var product))
            {
                product.ImageVector = response.Embedding.Values.ToArray();
                await CheckAndDispatch(product);
            }
        }
    }

    // UPDATED: Async method to handle channel writing safely
    static async Task CheckAndDispatch(InflightProduct product)
    {
        bool isReady = false;
        
        lock (product)
        {
            if (product.IsComplete)
            {
                // We mark it as ready and remove it from dictionary to prevent double processing
                // But we don't write to the channel inside the lock
                if (_inflight.TryRemove(product.Id, out _))
                {
                    isReady = true;
                }
            }
        }

        if (isReady)
        {
            var finalProduct = MapToProto(product);
            await _uploadChannel.Writer.WriteAsync(finalProduct);
        }
    }

    static async Task BatchAndUpload(ProductSearch.ProductSearchClient mnemeClient)
    {
        var batch = new List<Product>();
        int totalUploaded = 0;
        
        await foreach (var item in _uploadChannel.Reader.ReadAllAsync())
        {
            batch.Add(item);

            if (batch.Count >= MnemeBatchSize)
            {
                await UploadBatch(mnemeClient, batch);
                totalUploaded += batch.Count;
                Console.Write($"\rIndexed {totalUploaded} products (Text+Image)...");
                batch.Clear();
            }
        }

        if (batch.Count > 0) 
        {
            await UploadBatch(mnemeClient, batch);
            totalUploaded += batch.Count;
        }
    }

    static async Task UploadBatch(ProductSearch.ProductSearchClient client, List<Product> batch)
    {
        try
        {
            var req = new UpsertProductsRequest();
            req.Products.AddRange(batch);
            var res = await client.UpsertProductsAsync(req);
            
            if (!res.Success)
            {
                Console.WriteLine($"\n[ERROR] Batch rejected: {res.Message}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n[RPC ERROR] Failed to upload: {ex.Message}");
        }
    }

    static Product MapToProto(InflightProduct p)
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