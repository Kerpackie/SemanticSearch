var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddOpenApi();
builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(policy =>
    {
        policy.WithOrigins("http://localhost:5173", "http://localhost:3000")
            .AllowAnyHeader()
            .AllowAnyMethod();
    });
});

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

app.UseCors();
app.UseHttpsRedirection();

// Stub product data for the clothing ecommerce platform
var products = new List<Product>
{
    new(1, "Classic White T-Shirt", "A timeless white cotton t-shirt perfect for any casual occasion.", 29.99m, "Tops", "https://picsum.photos/seed/tshirt1/400/500", ["White", "Black", "Gray"], ["S", "M", "L", "XL"], 4.5, 128),
    new(2, "Slim Fit Denim Jeans", "Modern slim fit jeans with comfortable stretch fabric.", 79.99m, "Bottoms", "https://picsum.photos/seed/jeans1/400/500", ["Blue", "Black", "Dark Blue"], ["28", "30", "32", "34", "36"], 4.7, 256),
    new(3, "Wool Blend Blazer", "Sophisticated blazer perfect for business or smart casual wear.", 149.99m, "Outerwear", "https://picsum.photos/seed/blazer1/400/500", ["Navy", "Charcoal", "Black"], ["S", "M", "L", "XL"], 4.8, 89),
    new(4, "Floral Summer Dress", "Light and breezy floral print dress for warm weather.", 59.99m, "Dresses", "https://picsum.photos/seed/dress1/400/500", ["Blue Floral", "Pink Floral", "Yellow Floral"], ["XS", "S", "M", "L"], 4.6, 167),
    new(5, "Leather Chelsea Boots", "Premium leather Chelsea boots with elastic side panels.", 189.99m, "Footwear", "https://picsum.photos/seed/boots1/400/500", ["Brown", "Black"], ["7", "8", "9", "10", "11", "12"], 4.9, 203),
    new(6, "Cashmere Sweater", "Luxuriously soft cashmere sweater for ultimate comfort.", 199.99m, "Tops", "https://picsum.photos/seed/sweater1/400/500", ["Cream", "Navy", "Burgundy"], ["S", "M", "L", "XL"], 4.8, 94),
    new(7, "Running Sneakers", "Lightweight performance sneakers with cushioned sole.", 129.99m, "Footwear", "https://picsum.photos/seed/sneakers1/400/500", ["White/Blue", "Black/Red", "Gray/Green"], ["7", "8", "9", "10", "11", "12"], 4.4, 312),
    new(8, "Linen Shirt", "Breathable linen shirt perfect for summer days.", 69.99m, "Tops", "https://picsum.photos/seed/shirt1/400/500", ["White", "Light Blue", "Beige"], ["S", "M", "L", "XL", "XXL"], 4.3, 178),
    new(9, "Pleated Midi Skirt", "Elegant pleated skirt with a flattering midi length.", 49.99m, "Bottoms", "https://picsum.photos/seed/skirt1/400/500", ["Black", "Navy", "Blush"], ["XS", "S", "M", "L"], 4.5, 145),
    new(10, "Puffer Jacket", "Warm and lightweight puffer jacket for cold weather.", 179.99m, "Outerwear", "https://picsum.photos/seed/puffer1/400/500", ["Black", "Navy", "Olive"], ["S", "M", "L", "XL"], 4.7, 221),
    new(11, "Cotton Chinos", "Versatile cotton chinos for a smart casual look.", 59.99m, "Bottoms", "https://picsum.photos/seed/chinos1/400/500", ["Khaki", "Navy", "Olive", "Gray"], ["28", "30", "32", "34", "36"], 4.4, 189),
    new(12, "Silk Blouse", "Elegant silk blouse with a relaxed fit.", 89.99m, "Tops", "https://picsum.photos/seed/blouse1/400/500", ["Ivory", "Black", "Dusty Rose"], ["XS", "S", "M", "L"], 4.6, 112)
};

var categories = new[] { "All", "Tops", "Bottoms", "Dresses", "Outerwear", "Footwear" };

// API Endpoints
app.MapGet("/api/products", (string? category, string? search) =>
{
    var result = products.AsEnumerable();
    
    if (!string.IsNullOrEmpty(category) && category != "All")
    {
        result = result.Where(p => p.Category.Equals(category, StringComparison.OrdinalIgnoreCase));
    }
    
    if (!string.IsNullOrEmpty(search))
    {
        result = result.Where(p => 
            p.Name.Contains(search, StringComparison.OrdinalIgnoreCase) ||
            p.Description.Contains(search, StringComparison.OrdinalIgnoreCase));
    }
    
    return result.ToList();
}).WithName("GetProducts");

app.MapGet("/api/products/{id:int}", (int id) =>
{
    var product = products.FirstOrDefault(p => p.Id == id);
    return product is null ? Results.NotFound() : Results.Ok(product);
}).WithName("GetProductById");

app.MapGet("/api/categories", () => categories).WithName("GetCategories");

app.MapPost("/api/search", (SearchRequest request) =>
{
    // This will eventually be replaced with semantic search
    var result = products.Where(p =>
        p.Name.Contains(request.Query, StringComparison.OrdinalIgnoreCase) ||
        p.Description.Contains(request.Query, StringComparison.OrdinalIgnoreCase)).ToList();
    
    return new SearchResponse(result, request.Query, result.Count);
}).WithName("SemanticSearch");

app.Run();

// Records for the API
record Product(
    int Id,
    string Name,
    string Description,
    decimal Price,
    string Category,
    string ImageUrl,
    string[] Colors,
    string[] Sizes,
    double Rating,
    int ReviewCount);

record SearchRequest(string Query);

record SearchResponse(List<Product> Products, string Query, int TotalResults);
