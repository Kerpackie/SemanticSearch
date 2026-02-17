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
            // Step 1: NLP Processing via GptApi (HTTP) - get full outfit response
            var outfitResponse = await GetOutfitSuggestion(request.Query, context.CancellationToken);
            
            // If we have outfit items, use per-item targeted search and reranking
            if (outfitResponse?.ItemsToPurchase != null && outfitResponse.ItemsToPurchase.Count > 0)
            {
                _logger.LogInformation("Outfit suggestion: {OutfitName} with {ItemCount} items", 
                    outfitResponse.OutfitName, outfitResponse.ItemsToPurchase.Count);
                
                return await SearchWithPerItemReranking(
                    outfitResponse, 
                    request, 
                    context.CancellationToken);
            }
            
            // Fallback: simple text search without outfit context
            _logger.LogInformation("No outfit items found, using simple text search");
            return await SearchSimple(request.Query, request, context.CancellationToken);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error processing search request");
            throw new RpcException(new Status(StatusCode.Internal, $"Search failed: {ex.Message}"));
        }
    }

    /// <summary>
    /// Performs per-item targeted search and reranking for outfit suggestions.
    /// Each item gets its own search, embedding, and rerank query for higher accuracy.
    /// </summary>
    private async Task<SearchResponse> SearchWithPerItemReranking(
        OutfitSuggestionResponse outfit,
        SearchRequest request,
        CancellationToken cancellationToken)
    {
        const int candidatesPerItem = 50;  // Retrieve 50 candidates per item
        const int topKPerItem = 10;        // Keep top 10 after reranking per item
        
        var allResults = new List<SearchResult>();
        var processedQueryParts = new List<string>();

        // Generate image embedding once if provided
        float[] imageEmbedding = [];
        if (request.Image != null && !request.Image.IsEmpty)
        {
            imageEmbedding = await GetImageEmbedding(request.Image.ToByteArray(), cancellationToken);
            _logger.LogInformation("Generated image embedding with {Dimensions} dimensions", imageEmbedding.Length);
        }

        // Process each outfit item independently
        foreach (var item in outfit.ItemsToPurchase)
        {
            _logger.LogInformation("=== Processing item: {Color} {ItemName} ===", item.Color, item.ItemName);
            
            // Build the search query for embedding (simple: "Color ItemName")
            var searchQuery = $"{item.Color} {item.ItemName}";
            processedQueryParts.Add(searchQuery);
            
            // Generate text embedding for this specific item
            var textEmbedding = await GetTextEmbedding(searchQuery, cancellationToken);
            _logger.LogInformation("Generated embedding for '{SearchQuery}'", searchQuery);
            
            // Search for candidates for this item
            var products = await SearchProducts(textEmbedding, imageEmbedding, candidatesPerItem, cancellationToken);
            _logger.LogInformation("Found {Count} candidates for {ItemName}", products.Count, item.ItemName);
            
            if (products.Count == 0) continue;

            // Convert to SearchResults
            var candidates = products.Select((p, i) =>
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

            if (request.EnableReranking)
            {
                // Build the TARGETED rerank query using the Golden Formula:
                // Query = StyleDescription + ItemName + Reasoning
                var targetedQuery = BuildTargetedRerankQuery(outfit.StyleDescription, item);
                _logger.LogInformation("Targeted rerank query: {Query}", targetedQuery);
                
                // Rerank only these candidates against the targeted query
                var rerankedResults = await RerankResultsForItem(
                    targetedQuery, 
                    candidates, 
                    topKPerItem, 
                    request.CustomerId, 
                    cancellationToken);
                
                _logger.LogInformation("Reranked to {Count} results for {ItemName}", rerankedResults.Count, item.ItemName);
                
                // Tag results with their source item for debugging/display
                foreach (var result in rerankedResults)
                {
                    result.Metadata["outfit_item"] = item.ItemName;
                    result.Metadata["outfit_item_color"] = item.Color;
                }
                
                allResults.AddRange(rerankedResults);
            }
            else
            {
                // No reranking - just take top K
                var topResults = candidates.Take(topKPerItem).ToList();
                foreach (var result in topResults)
                {
                    result.Metadata["outfit_item"] = item.ItemName;
                    result.Metadata["outfit_item_color"] = item.Color;
                }
                allResults.AddRange(topResults);
            }
        }

        // Deduplicate results (same product might match multiple items)
        var deduplicatedResults = allResults
            .GroupBy(r => r.Id)
            .Select(g => g.OrderByDescending(r => r.Score).First())
            .OrderByDescending(r => r.Score)
            .Select((r, i) =>
            {
                r.Rank = i + 1;
                return r;
            })
            .ToList();

        // Apply overall limit
        var finalLimit = request.Limit > 0 ? request.Limit : 50;
        var finalResults = deduplicatedResults.Take(finalLimit).ToList();

        _logger.LogInformation("=== PER-ITEM SEARCH COMPLETE ===");
        _logger.LogInformation("Total unique results: {Count}, returning top {Limit}", 
            deduplicatedResults.Count, finalResults.Count);

        if (finalResults.Count > 0)
        {
            _logger.LogInformation("Score range: {Max:F4} to {Min:F4}", 
                finalResults.Max(r => r.Score), finalResults.Min(r => r.Score));
        }

        var processedQuery = $"[Outfit: {outfit.OutfitName}] {string.Join(" | ", processedQueryParts)}";

        return new SearchResponse
        {
            ProcessedQuery = processedQuery,
            TotalResults = finalResults.Count,
            Results = { finalResults }
        };
    }

    /// <summary>
    /// Builds a targeted, natural language query for reranking a specific item.
    /// Formula: StyleDescription + ItemName + Color + Reasoning
    /// </summary>
    private static string BuildTargetedRerankQuery(string styleDescription, SuggestedItem item)
    {
        // Construct a natural language query that gives the Cross Encoder semantic context
        // Goal: Sound like a natural search query, not a keyword list
        // 
        // Example output:
        // "I'm looking for white canvas loafers. The overall style is relaxed and airy, 
        //  perfect for a sunny day outdoors. These shoes should provide a clean, casual 
        //  option that pairs well with chinos."
        
        var queryParts = new List<string>();

        // Start with what we're looking for (natural phrasing)
        queryParts.Add($"I'm looking for {item.Color.ToLowerInvariant()} {item.ItemName.ToLowerInvariant()}.");

        // Add style context if available (reworded to be more natural)
        if (!string.IsNullOrWhiteSpace(styleDescription))
        {
            // Clean up the style description - it's already descriptive
            queryParts.Add($"The overall style is {styleDescription.TrimEnd('.')}.");
        }

        // Add the reasoning - this is the semantic gold
        if (!string.IsNullOrWhiteSpace(item.Reasoning))
        {
            // The reasoning often starts with a verb like "Provides" or "Adds"
            // Make it flow better by prefixing if needed
            var reasoning = item.Reasoning.Trim();
            if (char.IsUpper(reasoning[0]) && !reasoning.StartsWith("I ") && !reasoning.StartsWith("The "))
            {
                // Likely starts with a verb like "Provides", "Adds", "Creates"
                // Add a subject to make it flow
                queryParts.Add($"This item {char.ToLowerInvariant(reasoning[0])}{reasoning[1..]}");
            }
            else
            {
                queryParts.Add(reasoning);
            }
        }

        return string.Join(" ", queryParts);
    }

    /// <summary>
    /// Fallback simple search for queries that don't produce outfit suggestions.
    /// </summary>
    private async Task<SearchResponse> SearchSimple(
        string query,
        SearchRequest request,
        CancellationToken cancellationToken)
    {
        // Generate text embeddings via Glyph
        var textEmbedding = await GetTextEmbedding(query, cancellationToken);
        _logger.LogInformation("Generated text embedding with {Dimensions} dimensions", textEmbedding.Length);

        // Generate image embeddings via Eidolon (if image provided)
        float[] imageEmbedding = [];
        if (request.Image != null && !request.Image.IsEmpty)
        {
            imageEmbedding = await GetImageEmbedding(request.Image.ToByteArray(), cancellationToken);
            _logger.LogInformation("Generated image embedding with {Dimensions} dimensions", imageEmbedding.Length);
        }

        // Search database via Mneme
        var products = await SearchProducts(textEmbedding, imageEmbedding, request.Limit, cancellationToken);
        _logger.LogInformation("Found {Count} products from database", products.Count);

        // Convert to SearchResults
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

        if (request.EnableReranking && results.Count > 0)
        {
            results = await RerankResults(query, results, request.Limit, request.CustomerId, cancellationToken);
            _logger.LogInformation("Reranked {Count} results", results.Count);
        }

        return new SearchResponse
        {
            ProcessedQuery = query,
            TotalResults = results.Count,
            Results = { results }
        };
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

    /// <summary>
    /// Gets the full outfit suggestion from GptApi, preserving all context including StyleDescription and Reasoning.
    /// </summary>
    private async Task<OutfitSuggestionResponse?> GetOutfitSuggestion(string query, CancellationToken cancellationToken)
    {
        try
        {
            var httpClient = _httpClientFactory.CreateClient("GptApi");
            var response = await httpClient.PostAsJsonAsync("/suggest", new { Request = query }, cancellationToken);
            
            if (response.IsSuccessStatusCode)
            {
                var result = await response.Content.ReadFromJsonAsync<OutfitSuggestionResponse>(cancellationToken);
                if (result?.ItemsToPurchase != null && result.ItemsToPurchase.Count > 0)
                {
                    _logger.LogInformation("Outfit suggestion received: {OutfitName} - {StyleDescription}", 
                        result.OutfitName, result.StyleDescription);
                    foreach (var item in result.ItemsToPurchase)
                    {
                        _logger.LogDebug("  Item: {Color} {ItemName} - {Reasoning}", 
                            item.Color, item.ItemName, item.Reasoning);
                    }
                    return result;
                }
            }
            
            _logger.LogWarning("GptApi /suggest returned {StatusCode} or empty items", response.StatusCode);
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "GptApi /suggest unavailable");
            return null;
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
        _logger.LogInformation("=== RERANKING STARTED ===");
        _logger.LogInformation("Query: {Query}, Results to rerank: {Count}, CustomerId: {CustomerId}", 
            query, results.Count, customerId ?? "NULL");
            
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

        _logger.LogInformation("Calling Arbiter with {DocCount} documents, TopK: {TopK}", 
            documents.Count, request.TopK);

        // Get Arbiter reranking scores
        RerankResponse? arbiterResponse = null;
        try
        {
            // Set a deadline of 2 minutes for the Arbiter call to allow for ML model processing
            var deadline = DateTime.UtcNow.AddMinutes(2);
            var callOptions = new CallOptions(deadline: deadline, cancellationToken: cancellationToken);
            
            arbiterResponse = await _arbiterClient.RerankAsync(request, callOptions);
            _logger.LogInformation("✅ Arbiter returned {Count} results", arbiterResponse.Results.Count);
            
            if (arbiterResponse.Results.Count > 0)
            {
                var avgArbiterScore = arbiterResponse.Results.Average(r => r.Score);
                var maxArbiterScore = arbiterResponse.Results.Max(r => r.Score);
                var minArbiterScore = arbiterResponse.Results.Min(r => r.Score);
                _logger.LogInformation("Arbiter scores - Avg: {Avg:F4}, Max: {Max:F4}, Min: {Min:F4}", 
                    avgArbiterScore, maxArbiterScore, minArbiterScore);
                    
                // Log first 3 arbiter results
                foreach (var r in arbiterResponse.Results.Take(3))
                {
                    _logger.LogInformation("  Arbiter result: {Id} -> Score: {Score:F4}", r.Id, r.Score);
                }
            }
        }
        catch (RpcException ex) when (ex.StatusCode == StatusCode.DeadlineExceeded)
        {
            _logger.LogWarning("⚠️ ARBITER CALL TIMED OUT after 30s! Falling back to original scores");
            return results;
        }
        catch (RpcException ex) when (ex.StatusCode == StatusCode.Cancelled)
        {
            _logger.LogWarning("⚠️ ARBITER CALL CANCELLED! Falling back to original scores");
            return results;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ ARBITER CALL FAILED! Falling back to original scores");
            // If Arbiter fails, return results with original scores
            return results;
        }

        // Get recommender scores if customer is logged in
        Dictionary<string, float>? recommenderScores = null;
        if (!string.IsNullOrEmpty(customerId))
        {
            _logger.LogInformation("Fetching recommender scores for customer: {CustomerId}", customerId);
            recommenderScores = await GetRecommenderScores(customerId, results, cancellationToken);
            if (recommenderScores != null && recommenderScores.Count > 0)
            {
                _logger.LogInformation("✅ Got {Count} recommender scores", recommenderScores.Count);
            }
            else
            {
                _logger.LogWarning("⚠️ No recommender scores returned");
            }
        }
        else
        {
            _logger.LogInformation("No customer ID provided - skipping recommender");
        }

        // Combine scores and create final results
        var resultMap = results.ToDictionary(r => r.Id);
        var combinedResults = new List<(SearchResult result, float finalScore)>();

        int combinedCount = 0;
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
                
                if (combinedCount < 3) // Log first 3
                {
                    _logger.LogInformation("Product {Id}: Arbiter={ArbiterScore:F4}, Recommender={RecommenderScore:F4}, Final={FinalScore:F4}", 
                        arbiterResult.Id, arbiterScore, recommenderScore, finalScore);
                }
                combinedCount++;
            }
            else
            {
                // Use only arbiter score if no recommender score available
                finalScore = arbiterScore;
                if (combinedCount < 3) // Log first 3
                {
                    _logger.LogInformation("Product {Id}: Arbiter={ArbiterScore:F4}, No Recommender, Final={FinalScore:F4}", 
                        arbiterResult.Id, arbiterScore, finalScore);
                }
                combinedCount++;
            }

            combinedResults.Add((original, finalScore));
        }

        _logger.LogInformation("Combined {Count} results", combinedResults.Count);

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

        if (rankedResults.Count > 0)
        {
            var finalAvg = rankedResults.Average(r => r.Score);
            var finalMax = rankedResults.Max(r => r.Score);
            var finalMin = rankedResults.Min(r => r.Score);
            _logger.LogInformation("=== RERANKING COMPLETE ===");
            _logger.LogInformation("Final scores - Avg: {Avg:F4}, Max: {Max:F4}, Min: {Min:F4}", 
                finalAvg, finalMax, finalMin);
            _logger.LogInformation("Top 3 results:");
            foreach (var r in rankedResults.Take(3))
            {
                _logger.LogInformation("  {Rank}. {Name} (Score: {Score:F4})", r.Rank, r.Name, r.Score);
            }
        }

        return rankedResults;
    }

    /// <summary>
    /// Reranks results for a single outfit item using a targeted query.
    /// This is optimized for per-item reranking with focused queries.
    /// </summary>
    private async Task<List<SearchResult>> RerankResultsForItem(
        string targetedQuery, 
        List<SearchResult> candidates, 
        int topK, 
        string? customerId, 
        CancellationToken cancellationToken)
    {
        _logger.LogInformation("=== ITEM RERANKING ===");
        _logger.LogInformation("Targeted query: {Query}", targetedQuery);
        _logger.LogInformation("Candidates: {Count}, TopK: {TopK}", candidates.Count, topK);
        
        // Build documents for Arbiter
        var documents = candidates.Select(r => new Document
        {
            Id = r.Id,
            Text = $"{r.Name}. {r.Description}"
        }).ToList();

        var request = new RerankRequest
        {
            Query = targetedQuery,
            TopK = topK > 0 ? topK : candidates.Count
        };
        request.Documents.AddRange(documents);

        RerankResponse? arbiterResponse = null;
        try
        {
            // Set a deadline of 2 minutes for the Arbiter call to allow for ML model processing
            var deadline = DateTime.UtcNow.AddMinutes(2);
            var callOptions = new CallOptions(deadline: deadline, cancellationToken: cancellationToken);
            
            arbiterResponse = await _arbiterClient.RerankAsync(request, callOptions);
            _logger.LogInformation("✅ Arbiter returned {Count} results for item", arbiterResponse.Results.Count);
            
            if (arbiterResponse.Results.Count > 0)
            {
                var avgScore = arbiterResponse.Results.Average(r => r.Score);
                var maxScore = arbiterResponse.Results.Max(r => r.Score);
                var minScore = arbiterResponse.Results.Min(r => r.Score);
                _logger.LogInformation("Item rerank scores - Avg: {Avg:F4}, Max: {Max:F4}, Min: {Min:F4}", 
                    avgScore, maxScore, minScore);
            }
        }
        catch (RpcException ex) when (ex.StatusCode == StatusCode.DeadlineExceeded)
        {
            _logger.LogWarning("⚠️ ARBITER CALL TIMED OUT for item after 30s! Returning top candidates by vector score");
            return candidates.Take(topK).ToList();
        }
        catch (RpcException ex) when (ex.StatusCode == StatusCode.Cancelled)
        {
            _logger.LogWarning("⚠️ ARBITER CALL CANCELLED for item! Returning top candidates by vector score");
            return candidates.Take(topK).ToList();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ ARBITER CALL FAILED for item! Returning top candidates by vector score");
            return candidates.Take(topK).ToList();
        }

        // Map arbiter results back to search results
        var candidateMap = candidates.ToDictionary(c => c.Id);
        var rerankedResults = new List<SearchResult>();

        // Get recommender scores if customer is logged in
        Dictionary<string, float>? recommenderScores = null;
        if (!string.IsNullOrEmpty(customerId))
        {
            recommenderScores = await GetRecommenderScores(customerId, candidates, cancellationToken);
        }

        foreach (var arbiterResult in arbiterResponse.Results)
        {
            if (!candidateMap.TryGetValue(arbiterResult.Id, out var original)) continue;

            float arbiterScore = arbiterResult.Score;
            float finalScore;

            if (recommenderScores != null && recommenderScores.TryGetValue(arbiterResult.Id, out var recommenderScore))
            {
                // Combine arbiter and recommender scores (60% arbiter, 40% recommender)
                finalScore = (0.6f * arbiterScore) + (0.4f * recommenderScore);
            }
            else
            {
                finalScore = arbiterScore;
            }

            var result = new SearchResult
            {
                Id = original.Id,
                Name = original.Name,
                Description = original.Description,
                Score = finalScore,
                Rank = rerankedResults.Count + 1
            };
            foreach (var kvp in original.Metadata)
            {
                result.Metadata[kvp.Key] = kvp.Value;
            }
            rerankedResults.Add(result);
        }

        // Sort by final score
        return rerankedResults
            .OrderByDescending(r => r.Score)
            .Select((r, i) =>
            {
                r.Rank = i + 1;
                return r;
            })
            .ToList();
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
