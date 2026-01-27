namespace HmDataIngest;

public class ArticleCsv
{
    [CsvHelper.Configuration.Attributes.Name("article_id")]
    public string ArticleId { get; set; } = "";

    [CsvHelper.Configuration.Attributes.Name("prod_name")]
    public string ProdName { get; set; } = "";

    [CsvHelper.Configuration.Attributes.Name("detail_desc")]
    public string DetailDesc { get; set; } = "";

    [CsvHelper.Configuration.Attributes.Name("colour_group_name")]
    public string Color { get; set; } = "";

    [CsvHelper.Configuration.Attributes.Name("product_type_name")]
    public string Type { get; set; } = "";

    [CsvHelper.Configuration.Attributes.Name("graphical_appearance_name")]
    public string Pattern { get; set; } = "";

    [CsvHelper.Configuration.Attributes.Name("index_group_name")]
    public string Group { get; set; } = ""; // e.g., Ladieswear, Baby/Children
}