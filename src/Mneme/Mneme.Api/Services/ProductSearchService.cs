using Google.Protobuf.Collections;
using Grpc.Core;
using Mneme.Api;
using Qdrant.Client;
using Qdrant.Client.Grpc;

namespace Mneme.Api.Services;

public class ProductSearchService : ProductSearch.ProductSearchBase
{
    private readonly QdrantClient _qdrantClient;
    private const string CollectionName = "hm_articles";
    
    // CONFIGURATION
    // Text: 768 for BERT/MPNet
    // Image: 512 for standard CLIP (ViT-B/32)
    private const ulong TextVectorSize = 768; 
    private const ulong ImageVectorSize = 512; 

    public ProductSearchService(QdrantClient qdrantClient)
    {
        _qdrantClient = qdrantClient;
    }

    private async Task EnsureCollectionExistsAsync()
    {
        try 
        {
            var collections = await _qdrantClient.ListCollectionsAsync();
            if (!collections.Contains(CollectionName))
            {
                Console.WriteLine($"Creating collection '{CollectionName}' with named vectors...");
                
                // Create collection with multiple named vectors
                await _qdrantClient.CreateCollectionAsync(CollectionName, new VectorParamsMap
                {
                    Map = 
                    {
                        ["text"] = new VectorParams { Size = TextVectorSize, Distance = Distance.Cosine },
                        ["image"] = new VectorParams { Size = ImageVectorSize, Distance = Distance.Cosine }
                    }
                });
                
                Console.WriteLine("Collection created successfully.");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Error] Failed to create collection: {ex.Message}");
            throw; 
        }
    }

    public override async Task<UpsertProductsResponse> UpsertProducts(UpsertProductsRequest request, ServerCallContext context)
    {
        try 
        {
            await EnsureCollectionExistsAsync();

            var points = new List<PointStruct>();

            foreach (var product in request.Products)
            {
                // 1. Build Metadata Payload
                var payload = new MapField<string, Value>
                {
                    { "name", new Value { StringValue = product.Name ?? "" } },
                    { "description", new Value { StringValue = product.Description ?? "" } }
                };

                foreach (var kvp in product.Metadata)
                {
                    payload[kvp.Key] = new Value { StringValue = kvp.Value ?? "" };
                }

                // 2. Build Named Vectors
                // We check which vectors are present and map them to the "text" and "image" names
                var vectors = new Dictionary<string, float[]>();
                
                if (product.TextVector != null && product.TextVector.Count > 0)
                {
                    vectors["text"] = product.TextVector.ToArray();
                }

                if (product.ImageVector != null && product.ImageVector.Count > 0)
                {
                    vectors["image"] = product.ImageVector.ToArray();
                }

                if (vectors.Count == 0) continue; // Skip empty products

                var point = new PointStruct
                {
                    Id = new PointId { Uuid = product.Id },
                    Payload = { payload },
                    Vectors = vectors // Implicitly converts Dictionary to PointsVectors
                };

                points.Add(point);
            }

            if (points.Count > 0)
            {
                await _qdrantClient.UpsertAsync(CollectionName, points);
                Console.WriteLine($"[Success] Upserted batch of {points.Count} products.");
            }

            return new UpsertProductsResponse 
            { 
                Success = true, 
                Count = points.Count, 
                Message = "Indexed successfully" 
            };
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[CRITICAL ERROR] UpsertProducts failed: {ex.Message}");
            return new UpsertProductsResponse { Success = false, Message = ex.Message };
        }
    }

    public override async Task<ProductSearchResponse> SearchProducts(ProductSearchRequest request, ServerCallContext context)
    {
        var response = new ProductSearchResponse();
        try
        {
            IReadOnlyList<ScoredPoint> searchResult;

            // Route search based on input
            if (request.ImageVector != null && request.ImageVector.Count > 0)
            {
                // Image Search
                searchResult = await _qdrantClient.SearchAsync(
                    CollectionName, 
                    request.ImageVector.ToArray(),
                    vectorName: "image", 
                    limit: (ulong)request.Limit,
                    payloadSelector: true
                );
            }
            else
            {
                // Text Search (Default)
                searchResult = await _qdrantClient.SearchAsync(
                    CollectionName, 
                    request.TextVector.ToArray(),
                    vectorName: "text", 
                    limit: (ulong)request.Limit,
                    payloadSelector: true
                );
            }
            
            Console.WriteLine($"[Mneme] Search returned {searchResult.Count} results");

            foreach (var point in searchResult)
            {
                var p = new Product { Id = point.Id.Uuid, Score = point.Score };
                
                Console.WriteLine($"[Mneme] Point {point.Id.Uuid} has {point.Payload.Count} payload keys: [{string.Join(", ", point.Payload.Keys)}]");
                
                if (point.Payload.TryGetValue("name", out var n)) p.Name = n.StringValue;
                if (point.Payload.TryGetValue("description", out var d)) p.Description = d.StringValue;
                
                foreach(var kvp in point.Payload) 
                {
                   if (kvp.Value.KindCase == Value.KindOneofCase.StringValue)
                       p.Metadata.TryAdd(kvp.Key, kvp.Value.StringValue);
                }
                
                Console.WriteLine($"[Mneme] Product {p.Id} metadata has {p.Metadata.Count} entries");

                response.Products.Add(p);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Search failed: {ex.Message}");
        }
        return response;
    }
}