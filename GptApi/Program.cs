using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.AI;
using OpenAI;
using System.ClientModel; 
using System.ComponentModel;

var builder = WebApplication.CreateBuilder(args);

// Add service defaults & Aspire client integrations.
builder.AddServiceDefaults();

// Add services to the container.
builder.Services.AddProblemDetails();

// Learn more about configuring OpenAPI at https://aka.ms/aspnet/openapi
builder.Services.AddOpenApi();

// ==================================================================================
// 1. CONFIGURATION
// ==================================================================================
// Changed to Local for your LM Studio setup
var currentProvider = AiProvider.Local;

// Configuration Map
var config = currentProvider switch
{
    AiProvider.OpenAI => new { 
        Key = Environment.GetEnvironmentVariable("OPENAI_API_KEY") ?? "sk-demo", 
        Endpoint = new Uri("https://api.openai.com/v1"), 
        ModelId = "gpt-4o" 
    },
    AiProvider.DeepSeek => new { 
        Key = Environment.GetEnvironmentVariable("DEEPSEEK_API_KEY") ?? "sk-demo", 
        Endpoint = new Uri("https://api.deepseek.com"), 
        ModelId = "deepseek-chat" 
    },
    AiProvider.Local => new { 
        Key = "lm-studio", // LM Studio often ignores this, but it must not be empty
        Endpoint = new Uri("http://localhost:1234/v1"), // Standard LM Studio Port
        ModelId = "openai/gpt-oss-20b" // Your specific model identifier
    },
    _ => throw new NotImplementedException()
};

// ==================================================================================
// 2. DEPENDENCY INJECTION
// ==================================================================================
builder.Services.AddChatClient(serviceProvider =>
{
    // A: Create the low-level OpenAI Client
    var openAiClient = new OpenAIClient(
        new ApiKeyCredential(config.Key), 
        new OpenAIClientOptions { 
            Endpoint = config.Endpoint
            // Optional: Increase timeout for large local models if needed
            // NetworkTimeout = TimeSpan.FromMinutes(5) 
        }
    );

    // B: Wrap it in the Microsoft.Extensions.AI adapter
    return new OpenAIChatClient(openAiClient, config.ModelId);
});

var app = builder.Build();


// ==================================================================================
// 4. API ENDPOINTS
// ==================================================================================

app.MapPost("/suggest", async ([FromServices] IChatClient client, [FromBody] UserQuery query) =>
{
    var prompt = $@"
        You are an expert personal stylist.
        The user has potentially provided a scenario and/or potentially items they already own: '{query.Request}'.
        
        Analyze the scenario. Determine what items are missing to create a cohesive, stylish outfit.
        Return a structured list of ONLY the items they need to purchase. 
        Do not list items they already said they own.

        Respond ONLY with a valid JSON object matching this structure:
        {{{{
          """"OutfitName"""": """"string"""",
          """"StyleDescription"""": """"string"""",
          """"ItemsToPurchase"""": [
            {{{{ """"ItemName"""": """"string"""", """"Color"""": """"string"""", """"PriceRange"""": """"string"""", """"Reasoning"""": """"string"""" }}}}
          ]
        }}}}


        Respond with ONLY a JSON object. No reasoning, no code blocks.
    ";

    // Call AI with Structured Output
    // This sends the Schema for OutfitResponse to LM Studio.
    // Ensure your LM Studio version supports "Structured Output" or "JSON Mode" 
    // for this to work reliably.
    // Pass NULL for options to remove 'response_format' and 'strict' flags
    var response = await client.CompleteAsync(prompt, options: null);

    // Manually parse the text
    var result = System.Text.Json.JsonSerializer.Deserialize<OutfitResponse>(
        response.Message.Text, 
        new System.Text.Json.JsonSerializerOptions { PropertyNameCaseInsensitive = true }
    );

    return Results.Ok(result);
});

app.MapGet("/", () => "AI API Ready. POST to /suggest");

app.MapDefaultEndpoints();

app.Run();

public enum AiProvider { OpenAI, DeepSeek, Local }


// ==================================================================================
// 3. DATA CONTRACTS (Structured Output)
// ==================================================================================
public record FashionItem(
    [property: Description("The specific item to purchase (e.g. 'Beige Chinos', 'Leather Chelsea Boots').")] string ItemName,
    [property: Description("The color or pattern recommendation.")] string Color,
    [property: Description("Estimated price range for a quality version of this item.")] string PriceRange,
    [property: Description("Why this item completes the specific look.")] string Reasoning
);

public record OutfitResponse(
    [property: Description("A creative name for this outfit style.")] string OutfitName,
    [property: Description("A description of the overall vibe (e.g., 'Smart Casual', 'Black Tie').")] string StyleDescription,
    [property: Description("The list of specific items the user needs to BUY to complete the look.")] List<FashionItem> ItemsToPurchase
);

public record UserQuery(string Request);