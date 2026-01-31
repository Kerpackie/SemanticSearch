using DatabaseSeeder;

var builder = Host.CreateApplicationBuilder(args);

builder.AddNpgsqlDataSource("HM");
builder.Services.AddHostedService<Worker>();

var host = builder.Build();
host.Run();