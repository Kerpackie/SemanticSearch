using DatabaseSeeder;

var builder = Host.CreateApplicationBuilder(args);

builder.AddNpgsqlDataSource("postgres");
builder.Services.AddHostedService<Worker>();

var host = builder.Build();
host.Run();