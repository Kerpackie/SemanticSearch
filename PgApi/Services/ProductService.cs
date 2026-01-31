using Grpc.Core;
using Npgsql;

namespace PgApi.Services;

public class ProductService(NpgsqlDataSource dataSource, ILogger<ProductService> logger) 
    : PgApi.ProductService.ProductServiceBase
{
    public override async Task<ProductResponse> GetProduct(GetProductRequest request, ServerCallContext context)
    {
        logger.LogInformation("Getting product with ID: {ArticleId}", request.ArticleId);
        
        await using var conn = await dataSource.OpenConnectionAsync(context.CancellationToken);
        await using var cmd = conn.CreateCommand();
        
        cmd.CommandText = """
            SELECT article_id, product_code, prod_name, product_type_no, product_type_name,
                   product_group_name, graphical_appearance_no, graphical_appearance_name,
                   colour_group_code, colour_group_name, perceived_colour_value_id,
                   perceived_colour_value_name, perceived_colour_master_id, perceived_colour_master_name,
                   department_no, department_name, index_code, index_name, index_group_no,
                   index_group_name, section_no, section_name, garment_group_no, garment_group_name,
                   detail_desc
            FROM articles
            WHERE article_id = @id
            """;
        cmd.Parameters.AddWithValue("id", request.ArticleId);

        await using var reader = await cmd.ExecuteReaderAsync(context.CancellationToken);
        
        if (await reader.ReadAsync(context.CancellationToken))
        {
            return new ProductResponse { Product = MapProduct(reader) };
        }

        throw new RpcException(new Status(StatusCode.NotFound, $"Product with ID {request.ArticleId} not found"));
    }

    public override async Task<ProductListResponse> GetProducts(GetProductsRequest request, ServerCallContext context)
    {
        logger.LogInformation("Getting {Count} products by IDs", request.ArticleIds.Count);
        
        if (request.ArticleIds.Count == 0)
        {
            return new ProductListResponse { TotalCount = 0 };
        }

        await using var conn = await dataSource.OpenConnectionAsync(context.CancellationToken);
        await using var cmd = conn.CreateCommand();
        
        cmd.CommandText = $"""
            SELECT article_id, product_code, prod_name, product_type_no, product_type_name,
                   product_group_name, graphical_appearance_no, graphical_appearance_name,
                   colour_group_code, colour_group_name, perceived_colour_value_id,
                   perceived_colour_value_name, perceived_colour_master_id, perceived_colour_master_name,
                   department_no, department_name, index_code, index_name, index_group_no,
                   index_group_name, section_no, section_name, garment_group_no, garment_group_name,
                   detail_desc
            FROM articles
            WHERE article_id = ANY(@ids)
            """;
        cmd.Parameters.AddWithValue("ids", request.ArticleIds.ToArray());

        var response = new ProductListResponse();
        await using var reader = await cmd.ExecuteReaderAsync(context.CancellationToken);
        
        while (await reader.ReadAsync(context.CancellationToken))
        {
            response.Products.Add(MapProduct(reader));
        }
        
        response.TotalCount = response.Products.Count;
        return response;
    }

    public override async Task<ProductListResponse> ListProducts(ListProductsRequest request, ServerCallContext context)
    {
        var page = request.Page > 0 ? request.Page : 1;
        var pageSize = request.PageSize > 0 ? Math.Min(request.PageSize, 100) : 20;
        var offset = (page - 1) * pageSize;

        logger.LogInformation("Listing products - Page: {Page}, PageSize: {PageSize}", page, pageSize);
        
        await using var conn = await dataSource.OpenConnectionAsync(context.CancellationToken);
        
        // Get total count
        await using var countCmd = conn.CreateCommand();
        countCmd.CommandText = "SELECT COUNT(*) FROM articles";
        var totalCount = Convert.ToInt32(await countCmd.ExecuteScalarAsync(context.CancellationToken));
        
        // Get products
        await using var cmd = conn.CreateCommand();
        cmd.CommandText = """
            SELECT article_id, product_code, prod_name, product_type_no, product_type_name,
                   product_group_name, graphical_appearance_no, graphical_appearance_name,
                   colour_group_code, colour_group_name, perceived_colour_value_id,
                   perceived_colour_value_name, perceived_colour_master_id, perceived_colour_master_name,
                   department_no, department_name, index_code, index_name, index_group_no,
                   index_group_name, section_no, section_name, garment_group_no, garment_group_name,
                   detail_desc
            FROM articles
            ORDER BY article_id
            LIMIT @limit OFFSET @offset
            """;
        cmd.Parameters.AddWithValue("limit", pageSize);
        cmd.Parameters.AddWithValue("offset", offset);

        var response = new ProductListResponse { TotalCount = totalCount };
        await using var reader = await cmd.ExecuteReaderAsync(context.CancellationToken);
        
        while (await reader.ReadAsync(context.CancellationToken))
        {
            response.Products.Add(MapProduct(reader));
        }
        
        return response;
    }

    public override async Task<ProductListResponse> SearchProducts(SearchProductsRequest request, ServerCallContext context)
    {
        var limit = request.Limit > 0 ? Math.Min(request.Limit, 100) : 20;
        
        logger.LogInformation("Searching products with query: {Query}", request.Query);
        
        await using var conn = await dataSource.OpenConnectionAsync(context.CancellationToken);
        await using var cmd = conn.CreateCommand();
        
        cmd.CommandText = """
            SELECT article_id, product_code, prod_name, product_type_no, product_type_name,
                   product_group_name, graphical_appearance_no, graphical_appearance_name,
                   colour_group_code, colour_group_name, perceived_colour_value_id,
                   perceived_colour_value_name, perceived_colour_master_id, perceived_colour_master_name,
                   department_no, department_name, index_code, index_name, index_group_no,
                   index_group_name, section_no, section_name, garment_group_no, garment_group_name,
                   detail_desc
            FROM articles
            WHERE prod_name ILIKE @query 
               OR detail_desc ILIKE @query
               OR product_type_name ILIKE @query
               OR product_group_name ILIKE @query
            LIMIT @limit
            """;
        cmd.Parameters.AddWithValue("query", $"%{request.Query}%");
        cmd.Parameters.AddWithValue("limit", limit);

        var response = new ProductListResponse();
        await using var reader = await cmd.ExecuteReaderAsync(context.CancellationToken);
        
        while (await reader.ReadAsync(context.CancellationToken))
        {
            response.Products.Add(MapProduct(reader));
        }
        
        response.TotalCount = response.Products.Count;
        return response;
    }

    private static Product MapProduct(NpgsqlDataReader reader)
    {
        return new Product
        {
            ArticleId = reader.GetString(0),
            ProductCode = reader.GetInt32(1),
            ProdName = reader.IsDBNull(2) ? "" : reader.GetString(2),
            ProductTypeNo = reader.GetInt32(3),
            ProductTypeName = reader.IsDBNull(4) ? "" : reader.GetString(4),
            ProductGroupName = reader.IsDBNull(5) ? "" : reader.GetString(5),
            GraphicalAppearanceNo = reader.GetInt32(6),
            GraphicalAppearanceName = reader.IsDBNull(7) ? "" : reader.GetString(7),
            ColourGroupCode = reader.GetInt32(8),
            ColourGroupName = reader.IsDBNull(9) ? "" : reader.GetString(9),
            PerceivedColourValueId = reader.GetInt32(10),
            PerceivedColourValueName = reader.IsDBNull(11) ? "" : reader.GetString(11),
            PerceivedColourMasterId = reader.GetInt32(12),
            PerceivedColourMasterName = reader.IsDBNull(13) ? "" : reader.GetString(13),
            DepartmentNo = reader.GetInt32(14),
            DepartmentName = reader.IsDBNull(15) ? "" : reader.GetString(15),
            IndexCode = reader.IsDBNull(16) ? "" : reader.GetString(16),
            IndexName = reader.IsDBNull(17) ? "" : reader.GetString(17),
            IndexGroupNo = reader.GetInt32(18),
            IndexGroupName = reader.IsDBNull(19) ? "" : reader.GetString(19),
            SectionNo = reader.GetInt32(20),
            SectionName = reader.IsDBNull(21) ? "" : reader.GetString(21),
            GarmentGroupNo = reader.GetInt32(22),
            GarmentGroupName = reader.IsDBNull(23) ? "" : reader.GetString(23),
            DetailDesc = reader.IsDBNull(24) ? "" : reader.GetString(24)
        };
    }
}
