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
            var results = products.Select((p, i) =>
            {
                var result = new SearchResult
                {
                    Id = p.Id,
                    Name = p.Name,
                    Description = p.Description,
                    Score = p.Score,
                    Rank = i + 1
                };
                foreach (var kvp in p.Metadata)
                {
                    result.Metadata[kvp.Key] = kvp.Value;
                }
                _logger.LogDebug("Product {Id} has {MetadataCount} metadata entries", p.Id, result.Metadata.Count);
                return result;
            }).ToList();

            if (results.Count > 0)
            {
                var sampleResult = results[0];
                _logger.LogInformation("First result {Id} has {Count} metadata keys: [{Keys}]", 
                    sampleResult.Id, sampleResult.Metadata.Count, string.Join(", ", sampleResult.Metadata.Keys));
            }

            if (request.EnableReranking && results.Count > 0)
            {
                results = await RerankResults(processedQuery, results, request.Limit, request.CustomerId, context.CancellationToken);
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
            EnableReranking = request.EnableReranking,
            CustomerId = request.CustomerId
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

            var results = products.Select((p, i) =>
            {
                var result = new SearchResult
                {
                    Id = p.Id,
                    Name = p.Name,
                    Description = p.Description,
                    Score = p.Score,
                    Rank = i + 1
                };
                foreach (var kvp in p.Metadata)
                {
                    result.Metadata[kvp.Key] = kvp.Value;
                }
                return result;
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
            // Call the /suggest endpoint with { Request: string }
            var response = await httpClient.PostAsJsonAsync("/suggest", new { Request = query }, cancellationToken);
            
            if (response.IsSuccessStatusCode)
            {
                var result = await response.Content.ReadFromJsonAsync<OutfitSuggestionResponse>(cancellationToken);
                if (result?.ItemsToPurchase != null && result.ItemsToPurchase.Count > 0)
                {
                    // Build a search query from the suggested items
                    var itemNames = result.ItemsToPurchase.Select(i => $"{i.Color} {i.ItemName}").ToList();
                    var processedQuery = string.Join(" ", itemNames);
                    _logger.LogInformation("Outfit suggestion: {OutfitName} - searching for: {Items}", result.OutfitName, processedQuery);
                    return processedQuery;
                }
                return query;
            }
            
            _logger.LogWarning("GptApi /suggest returned {StatusCode}, using original query", response.StatusCode);
            return query;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "GptApi /suggest unavailable, using original query");
            return query;
        }
    }

    // Response models for GptApi /suggest endpoint
    private record OutfitSuggestionResponse(string OutfitName, string StyleDescription, List<SuggestedItem> ItemsToPurchase);
    private record SuggestedItem(string ItemName, string Color, string PriceRange, string Reasoning);

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

    private async Task<List<SearchResult>> RerankResults(string query, List<SearchResult> results, int limit, string? customerId, CancellationToken cancellationToken)
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

        // Get Arbiter reranking scores
        var arbiterResponse = await _arbiterClient.RerankAsync(request, cancellationToken: cancellationToken);
        var arbiterScores = arbiterResponse.Results.ToDictionary(r => r.Id, r => r.Score);

        // Get recommender scores if customer is logged in
        Dictionary<string, float>? recommenderScores = null;
        if (!string.IsNullOrEmpty(customerId))
        {
            recommenderScores = await GetRecommenderScores(customerId, results, cancellationToken);
        }

        // Combine scores and create final results
        var resultMap = results.ToDictionary(r => r.Id);
        var combinedResults = new List<(SearchResult result, float finalScore)>();

        foreach (var arbiterResult in arbiterResponse.Results)
        {
            if (!resultMap.ContainsKey(arbiterResult.Id)) continue;

            var original = resultMap[arbiterResult.Id];
            float arbiterScore = arbiterResult.Score;
            float finalScore;

            if (recommenderScores != null && recommenderScores.TryGetValue(arbiterResult.Id, out var recommenderScore))
            {
                // Combine arbiter and recommender scores (60% arbiter, 40% recommender)
                finalScore = (0.6f * arbiterScore) + (0.4f * recommenderScore);
                _logger.LogDebug("Product {Id}: Arbiter={ArbiterScore:F3}, Recommender={RecommenderScore:F3}, Final={FinalScore:F3}", 
                    arbiterResult.Id, arbiterScore, recommenderScore, finalScore);
            }
            else
            {
                // Use only arbiter score if no recommender score available
                finalScore = arbiterScore;
            }

            combinedResults.Add((original, finalScore));
        }

        // Sort by final score and assign ranks
        var rankedResults = combinedResults
            .OrderByDescending(r => r.finalScore)
            .Select((r, index) =>
            {
                var result = new SearchResult
                {
                    Id = r.result.Id,
                    Name = r.result.Name,
                    Description = r.result.Description,
                    Score = r.finalScore,
                    Rank = index + 1
                };
                foreach (var kvp in r.result.Metadata)
                {
                    result.Metadata[kvp.Key] = kvp.Value;
                }
                return result;
            })
            .ToList();

        return rankedResults;
    }

    private async Task<Dictionary<string, float>?> GetRecommenderScores(string customerId, List<SearchResult> results, CancellationToken cancellationToken)
    {
        try
        {
            var httpClient = _httpClientFactory.CreateClient("Recommender");
            
            _logger.LogInformation("GetRecommenderScores called with {Count} results", results.Count);
            
            // Log metadata for ALL results to diagnose the issue
            var resultsWithArticleId = 0;
            var resultsWithoutArticleId = 0;
            
            foreach (var searchResult in results)
            {
                if (searchResult.Metadata.ContainsKey("article_id"))
                {
                    resultsWithArticleId++;
                }
                else
                {
                    resultsWithoutArticleId++;
                    // Log first few results that are missing article_id
                    if (resultsWithoutArticleId <= 3)
                    {
                        _logger.LogWarning("Result {Id} missing article_id. Available keys: [{Keys}]", 
                            searchResult.Id, string.Join(", ", searchResult.Metadata.Keys));
                    }
                }
            }
            
            _logger.LogInformation("Metadata analysis: {WithArticleId} results have article_id, {WithoutArticleId} results missing article_id", 
                resultsWithArticleId, resultsWithoutArticleId);
            
            // Extract article IDs from metadata
            var articleIds = results
                .Where(r => r.Metadata.ContainsKey("article_id"))
                .Select(r => r.Metadata["article_id"])
                .Where(id => int.TryParse(id, out _))
                .Select(int.Parse)
                .ToList();

            if (articleIds.Count == 0)
            {
                _logger.LogWarning("No valid article IDs found in metadata for recommender scoring. All {TotalResults} results are missing article_id in metadata.", results.Count);
                _logger.LogWarning("This suggests the Qdrant vector database needs to be re-indexed with metadata. Run HmDataIngest to populate article_id metadata.");
                return null;
            }

            var requestBody = new
            {
                customer_id = customerId,
                article_ids = articleIds
            };

            var response = await httpClient.PostAsJsonAsync("/score", requestBody, cancellationToken);
            
            if (!response.IsSuccessStatusCode)
            {
                _logger.LogWarning("Recommender service returned {StatusCode}, falling back to arbiter-only scoring", response.StatusCode);
                return null;
            }

            var result = await response.Content.ReadFromJsonAsync<RecommenderScoreResponse>(cancellationToken);
            
            if (result?.Scores == null || result.Scores.Count == 0)
            {
                _logger.LogWarning("Recommender returned no scores");
                return null;
            }

            // Map scores back to result IDs using article_id metadata
            var scoreDict = new Dictionary<string, float>();
            foreach (var scoredArticle in result.Scores)
            {
                var matchingResult = results.FirstOrDefault(r => 
                    r.Metadata.TryGetValue("article_id", out var articleId) && 
                    articleId == scoredArticle.ArticleId.ToString());
                
                if (matchingResult != null)
                {
                    scoreDict[matchingResult.Id] = scoredArticle.Score;
                }
            }

            _logger.LogInformation("Got recommender scores for {Count} products", scoreDict.Count);
            return scoreDict;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to get recommender scores, falling back to arbiter-only scoring");
            return null;
        }
    }

    // Response models for Recommender API
    private record RecommenderScoreResponse(string CustomerId, List<ScoredArticle> Scores);
    private record ScoredArticle(int ArticleId, float Score);
}
