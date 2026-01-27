using Google.Protobuf.Collections;
using Grpc.Core;
using Mneme.Api;
using Qdrant.Client;
using Qdrant.Client.Grpc;

namespace Mneme.Api.Services;

public class ProductSearchService : ProductSearch.ProductSearchBase
{
    private readonly QdrantClient _qdrantClient;
    private const string CollectionName = "hm_articles2";
    private const ulong VectorSize = 768; 

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
                Console.WriteLine($"Creating collection '{CollectionName}'...");
                await _qdrantClient.CreateCollectionAsync(CollectionName, new VectorParams
                {
                    Size = VectorSize,
                    Distance = Distance.Cosine
                });
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
                var payload = new MapField<string, Value>
                {
                    { "name", new Value { StringValue = product.Name ?? "" } },
                    { "description", new Value { StringValue = product.Description ?? "" } }
                };

                // Add all metadata fields
                foreach (var kvp in product.Metadata)
                {
                    payload[kvp.Key] = new Value { StringValue = kvp.Value ?? "" };
                }

                var point = new PointStruct
                {
                    Id = new PointId { Uuid = product.Id },
                    Vectors = product.TextVector.ToArray(),
                    Payload = { payload }
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
            Console.WriteLine($"[CRITICAL ERROR] UpsertProducts failed: {ex}");
            return new UpsertProductsResponse 
            { 
                Success = false, 
                Message = $"Server Error: {ex.Message}" 
            };
        }
    }

    public override async Task<ProductSearchResponse> SearchProducts(ProductSearchRequest request, ServerCallContext context)
    {
        var response = new ProductSearchResponse();
        try
        {
            var searchResult = await _qdrantClient.SearchAsync(
                CollectionName, 
                request.TextVector.ToArray(), 
                limit: (ulong)request.Limit
            );
            
            foreach (var point in searchResult)
            {
                var p = new Product { Id = point.Id.Uuid, Score = point.Score };
                if (point.Payload.TryGetValue("name", out var n)) p.Name = n.StringValue;
                if (point.Payload.TryGetValue("description", out var d)) p.Description = d.StringValue;
                
                // Return metadata
                foreach(var kvp in point.Payload) 
                {
                   if (kvp.Value.KindCase == Value.KindOneofCase.StringValue)
                       p.Metadata.TryAdd(kvp.Key, kvp.Value.StringValue);
                }

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