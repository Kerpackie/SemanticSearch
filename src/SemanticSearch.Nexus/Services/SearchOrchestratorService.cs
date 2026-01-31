using Grpc.Core;
using SemanticSearch.Nexus.Clients.Glyph;
using SemanticSearch.Nexus.Clients.Clip;
using SemanticSearch.Nexus.Clients.Mneme;
using SemanticSearch.Nexus.Clients.Arbiter;

namespace SemanticSearch.Nexus.Services;

/// <summary>
/// The main orchestration service that coordinates semantic search across multiple services.
/// Flow: Query -> NLP (GptApi) -> Embeddings (Glyph & Eidolon) -> Database (Mneme) -> Reranking (Arbiter)
/// </summary>
public class SearchOrchestratorService : SearchOrchestrator.SearchOrchestratorBase
{
    private readonly ILogger<SearchOrchestratorService> _logger;
    private readonly Embedder.EmbedderClient _glyphClient;
    private readonly ClipEmbedder.ClipEmbedderClient _eidolonClient;
    private readonly ProductSearch.ProductSearchClient _mnemeClient;
    private readonly Reranker.RerankerClient _arbiterClient;
    private readonly IHttpClientFactory _httpClientFactory;

    public SearchOrchestratorService(
        ILogger<SearchOrchestratorService> logger,
        Embedder.EmbedderClient glyphClient,
        ClipEmbedder.ClipEmbedderClient eidolonClient,
        ProductSearch.ProductSearchClient mnemeClient,
        Reranker.RerankerClient arbiterClient,
        IHttpClientFactory httpClientFactory)
    {
        _logger = logger;
        _glyphClient = glyphClient;
        _eidolonClient = eidolonClient;
        _mnemeClient = mnemeClient;
        _arbiterClient = arbiterClient;
        _httpClientFactory = httpClientFactory;
    }

    public override async Task<SearchResponse> Search(SearchRequest request, ServerCallContext context)
    {
        _logger.LogInformation("Processing search request: {Query}", request.Query);

        try
        {
            // Step 1: NLP Processing via GptApi (HTTP)
            var processedQuery = await ProcessQueryWithNlp(request.Query, context.CancellationToken);
            _logger.LogInformation("NLP processed query: {ProcessedQuery}", processedQuery);

            // Step 2: Generate text embeddings via Glyph
            var textEmbedding = await GetTextEmbedding(processedQuery, context.CancellationToken);
            _logger.LogInformation("Generated text embedding with {Dimensions} dimensions", textEmbedding.Length);

            // Step 3: Generate image embeddings via Eidolon (if image provided)
            float[] imageEmbedding = [];
            if (request.Image != null && !request.Image.IsEmpty)
            {
                imageEmbedding = await GetImageEmbedding(request.Image.ToByteArray(), context.CancellationToken);
                _logger.LogInformation("Generated image embedding with {Dimensions} dimensions", imageEmbedding.Length);
            }

            // Step 4: Search database via Mneme
            var products = await SearchProducts(textEmbedding, imageEmbedding, request.Limit, context.CancellationToken);
            _logger.LogInformation("Found {Count} products from database", products.Count);

            // Step 5: Rerank results via Arbiter (if enabled)
            var results = products.Select((p, i) => new SearchResult
            {
                Id = p.Id,
                Name = p.Name,
                Description = p.Description,
                Score = p.Score,
                Rank = i + 1,
                Metadata = { p.Metadata }
            }).ToList();

            if (request.EnableReranking && results.Count > 0)
            {
                results = await RerankResults(processedQuery, results, request.Limit, context.CancellationToken);
                _logger.LogInformation("Reranked {Count} results", results.Count);
            }

            return new SearchResponse
            {
                ProcessedQuery = processedQuery,
                TotalResults = results.Count,
                Results = { results }
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error processing search request");
            throw new RpcException(new Status(StatusCode.Internal, $"Search failed: {ex.Message}"));
        }
    }

    public override async Task<SearchResponse> SearchByText(TextSearchRequest request, ServerCallContext context)
    {
        return await Search(new SearchRequest
        {
            Query = request.Query,
            Limit = request.Limit,
            EnableReranking = request.EnableReranking
        }, context);
    }

    public override async Task<SearchResponse> SearchByImage(ImageSearchRequest request, ServerCallContext context)
    {
        _logger.LogInformation("Processing image-only search request");

        try
        {
            // Generate image embeddings via Eidolon
            var imageEmbedding = await GetImageEmbedding(request.Image.ToByteArray(), context.CancellationToken);
            _logger.LogInformation("Generated image embedding with {Dimensions} dimensions", imageEmbedding.Length);

            // Search database via Mneme (image-only)
            var products = await SearchProducts([], imageEmbedding, request.Limit, context.CancellationToken);
            _logger.LogInformation("Found {Count} products from database", products.Count);

            var results = products.Select((p, i) => new SearchResult
            {
                Id = p.Id,
                Name = p.Name,
                Description = p.Description,
                Score = p.Score,
                Rank = i + 1,
                Metadata = { p.Metadata }
            }).ToList();

            return new SearchResponse
            {
                ProcessedQuery = "[Image Search]",
                TotalResults = results.Count,
                Results = { results }
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error processing image search request");
            throw new RpcException(new Status(StatusCode.Internal, $"Image search failed: {ex.Message}"));
        }
    }

    private async Task<string> ProcessQueryWithNlp(string query, CancellationToken cancellationToken)
    {
        try
        {
            var httpClient = _httpClientFactory.CreateClient("GptApi");
            var response = await httpClient.PostAsJsonAsync("/api/process-query", new { Query = query }, cancellationToken);
            
            if (response.IsSuccessStatusCode)
            {
                var result = await response.Content.ReadFromJsonAsync<NlpResponse>(cancellationToken);
                return result?.ProcessedQuery ?? query;
            }
            
            _logger.LogWarning("NLP service returned {StatusCode}, using original query", response.StatusCode);
            return query;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "NLP service unavailable, using original query");
            return query;
        }
    }

    private async Task<float[]> GetTextEmbedding(string text, CancellationToken cancellationToken)
    {
        var request = new EmbedSingleRequest { Text = text };
        var response = await _glyphClient.EmbedSingleAsync(request, cancellationToken: cancellationToken);
        return response.Embedding?.Values.ToArray() ?? Array.Empty<float>();
    }

    private async Task<float[]> GetImageEmbedding(byte[] imageData, CancellationToken cancellationToken)
    {
        var request = new EmbedImageRequest { Image = Google.Protobuf.ByteString.CopyFrom(imageData) };
        var response = await _eidolonClient.EmbedImageAsync(request, cancellationToken: cancellationToken);
        return response.Embedding?.Values.ToArray() ?? Array.Empty<float>();
    }

    private async Task<List<Product>> SearchProducts(float[] textVector, float[] imageVector, int limit, CancellationToken cancellationToken)
    {
        var request = new ProductSearchRequest
        {
            Limit = limit > 0 ? limit : 10
        };
        request.TextVector.AddRange(textVector);
        request.ImageVector.AddRange(imageVector);

        var response = await _mnemeClient.SearchProductsAsync(request, cancellationToken: cancellationToken);
        return response.Products.ToList();
    }

    private async Task<List<SearchResult>> RerankResults(string query, List<SearchResult> results, int limit, CancellationToken cancellationToken)
    {
        var documents = results.Select(r => new Document
        {
            Id = r.Id,
            Text = $"{r.Name}. {r.Description}"
        }).ToList();

        var request = new RerankRequest
        {
            Query = query,
            TopK = limit > 0 ? limit : results.Count
        };
        request.Documents.AddRange(documents);

        var response = await _arbiterClient.RerankAsync(request, cancellationToken: cancellationToken);

        // Map back to SearchResult with updated scores and ranks
        var resultMap = results.ToDictionary(r => r.Id);
        return response.Results
            .Where(r => resultMap.ContainsKey(r.Id))
            .Select(r =>
            {
                var original = resultMap[r.Id];
                return new SearchResult
                {
                    Id = r.Id,
                    Name = original.Name,
                    Description = original.Description,
                    Score = r.Score,
                    Rank = r.Rank,
                    Metadata = { original.Metadata }
                };
            })
            .ToList();
    }

    private record NlpResponse(string ProcessedQuery);
}
