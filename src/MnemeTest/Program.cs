using Grpc.Net.Client;
using Mneme.Api;

Console.WriteLine("=== Mneme Metadata Test ===\n");

// Get port from command line or use default
var port = args.Length > 0 && int.TryParse(args[0], out var p) ? p : 5002;
var mnemeUrl = $"http://localhost:{port}";

Console.WriteLine($"Connecting to Mneme service at {mnemeUrl}");
Console.WriteLine("(Pass port as first argument if different, e.g.: dotnet run 7123)\n");

// Connect to Mneme service
var channel = GrpcChannel.ForAddress(mnemeUrl);
var client = new ProductSearch.ProductSearchClient(channel);

Console.WriteLine($"Connected to Mneme service at {mnemeUrl}");

// Create a simple search request with a dummy vector
var request = new ProductSearchRequest
{
    Limit = 5
};

// Add a dummy text vector (768 dimensions for BERT/MPNet)
var dummyVector = Enumerable.Repeat(0.1f, 768).ToArray();
request.TextVector.AddRange(dummyVector);

Console.WriteLine("\nSending search request with dummy vector...\n");

try
{
    var response = await client.SearchProductsAsync(request);
    
    Console.WriteLine($"Received {response.Products.Count} products\n");
    Console.WriteLine("=".PadRight(80, '='));
    
    foreach (var product in response.Products)
    {
        Console.WriteLine($"\nProduct ID: {product.Id}");
        Console.WriteLine($"Name: {product.Name}");
        var descPreview = product.Description?.Length > 50 
            ? product.Description.Substring(0, 50) + "..." 
            : product.Description ?? "(no description)";
        Console.WriteLine($"Description: {descPreview}");
        Console.WriteLine($"Score: {product.Score:F4}");
        Console.WriteLine($"\nMetadata ({product.Metadata.Count} entries):");
        
        if (product.Metadata.Count == 0)
        {
            Console.WriteLine("  ⚠️  NO METADATA FOUND!");
        }
        else
        {
            foreach (var kvp in product.Metadata)
            {
                Console.WriteLine($"  {kvp.Key}: {kvp.Value}");
            }
            
            if (product.Metadata.ContainsKey("article_id"))
            {
                Console.WriteLine($"  ✅ article_id present: {product.Metadata["article_id"]}");
            }
            else
            {
                Console.WriteLine("  ❌ article_id is MISSING!");
            }
        }
        
        Console.WriteLine("=".PadRight(80, '='));
    }
    
    // Summary
    var productsWithArticleId = response.Products.Count(p => p.Metadata.ContainsKey("article_id"));
    var productsWithoutArticleId = response.Products.Count - productsWithArticleId;
    
    Console.WriteLine($"\n📊 Summary:");
    Console.WriteLine($"   Total products: {response.Products.Count}");
    Console.WriteLine($"   ✅ With article_id: {productsWithArticleId}");
    Console.WriteLine($"   ❌ Without article_id: {productsWithoutArticleId}");
    
    if (productsWithArticleId == response.Products.Count)
    {
        Console.WriteLine("\n✅ SUCCESS: All products have article_id in metadata!");
    }
    else if (productsWithArticleId == 0)
    {
        Console.WriteLine("\n❌ FAILURE: No products have article_id in metadata!");
        Console.WriteLine("   This means the Qdrant database needs to be re-indexed.");
    }
    else
    {
        Console.WriteLine("\n⚠️  WARNING: Some products have article_id, but not all!");
    }
}
catch (Exception ex)
{
    Console.WriteLine($"❌ Error: {ex.Message}");
    Console.WriteLine($"\nMake sure:");
    Console.WriteLine($"  1. Mneme service (mneme-api) is running");
    Console.WriteLine($"  2. It's listening on {mnemeUrl}");
    Console.WriteLine($"  3. If using Aspire AppHost, check the dashboard for the actual port");
    Console.WriteLine($"     Then run: dotnet run <port>");
}

Console.WriteLine("\nTest complete.");
