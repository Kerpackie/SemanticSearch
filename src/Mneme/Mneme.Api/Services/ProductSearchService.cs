using Grpc.Core;
using Mneme.Api;

namespace Mneme.Api.Services;

public class ProductSearchService : ProductSearch.ProductSearchBase
{
    public override Task<ProductSearchResponse> SearchProducts(ProductSearchRequest request, ServerCallContext context)
    {
        // Example: return dummy data
        var products = new List<Product>
        {
            new Product { Id = "1", Name = "Sample", Description = "Sample product", Score = 0.99f }
        };

        var response = new ProductSearchResponse();
        response.Products.AddRange(products);

        return Task.FromResult(response);
    }
}