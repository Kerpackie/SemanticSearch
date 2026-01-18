using System.Diagnostics;
using Npgsql;

namespace DatabaseSeeder;

public class Worker : BackgroundService
{
    private readonly NpgsqlDataSource _dataSource;
    private readonly ILogger<Worker> _logger;
    private readonly IHostApplicationLifetime _hostApplicationLifetime;

    public Worker(
        NpgsqlDataSource dataSource, 
        ILogger<Worker> logger, 
        IHostApplicationLifetime hostApplicationLifetime)
    {
        _dataSource = dataSource;
        _logger = logger;
        _hostApplicationLifetime = hostApplicationLifetime;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        var stopwatch = Stopwatch.StartNew();
        var csvPath = Path.Combine(AppContext.BaseDirectory, "Data");

        try
        {
            _logger.LogInformation("Waiting for database...");
            await WaitForDatabaseAsync(stoppingToken);

            // 1. Reset Schema (Fixes the "Extra Data" error by recreating the bad table)
            await EnsureSchemaAsync(stoppingToken);

            // 2. Import Data (Order is critical for Foreign Keys)
            await ImportTableAsync("customers", Path.Combine(csvPath, "customers.csv"), stoppingToken);
            await ImportTableAsync("articles", Path.Combine(csvPath, "articles.csv"), stoppingToken);
            await ImportTableAsync("transactions", Path.Combine(csvPath, "transactions_train.csv"), stoppingToken);

            _logger.LogInformation($"Seeding Complete! Total time: {stopwatch.Elapsed.TotalMinutes:F2} minutes.");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Critical error during database seeding.");
        }
        finally
        {
            _hostApplicationLifetime.StopApplication();
        }
    }

    private async Task EnsureSchemaAsync(CancellationToken token)
    {
        await using var conn = _dataSource.CreateConnection();
        await conn.OpenAsync(token);
        
        // CHECK: If articles has the wrong number of columns, we must drop it.
        // The previous run created it with 23 columns, but we need 25.
        // Dropping 'articles' will CASCADE delete 'transactions', forcing a clean rebuild.
        var cmdCheck = conn.CreateCommand();
        cmdCheck.CommandText = "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'articles';";
        var colCount = (long?)await cmdCheck.ExecuteScalarAsync(token) ?? 0;

        if (colCount > 0 && colCount != 25)
        {
            _logger.LogWarning($"Detected outdated schema (Columns: {colCount}). Rebuilding tables...");
            using var cmdDrop = conn.CreateCommand();
            cmdDrop.CommandText = "DROP TABLE IF EXISTS articles, transactions CASCADE;";
            await cmdDrop.ExecuteNonQueryAsync(token);
        }

        _logger.LogInformation("Ensuring schema exists...");
        
        const string createSql = @"
            CREATE TABLE IF NOT EXISTS customers (
                customer_id VARCHAR(64) PRIMARY KEY,
                FN DECIMAL(3,1),
                Active DECIMAL(3,1),
                club_member_status VARCHAR(50),
                fashion_news_frequency VARCHAR(50),
                age INT,
                postal_code VARCHAR(64)
            );

            CREATE TABLE IF NOT EXISTS articles (
                article_id VARCHAR(20) PRIMARY KEY,
                product_code INT,
                prod_name VARCHAR(255),
                product_type_no INT,
                product_type_name VARCHAR(255),
                product_group_name VARCHAR(255),
                graphical_appearance_no INT,
                graphical_appearance_name VARCHAR(255),
                colour_group_code INT,
                colour_group_name VARCHAR(50),
                perceived_colour_value_id INT,
                perceived_colour_value_name VARCHAR(50),
                
                -- THESE WERE MISSING IN THE PREVIOUS VERSION --
                perceived_colour_master_id INT,
                perceived_colour_master_name VARCHAR(50),
                
                department_no INT,
                department_name VARCHAR(255),
                index_code VARCHAR(10),
                index_name VARCHAR(50),
                index_group_no INT,
                index_group_name VARCHAR(50),
                section_no INT,
                section_name VARCHAR(255),
                garment_group_no INT,
                garment_group_name VARCHAR(255),
                detail_desc TEXT
            );

            CREATE TABLE IF NOT EXISTS transactions (
                t_dat DATE NOT NULL,
                customer_id VARCHAR(64) NOT NULL,
                article_id VARCHAR(20) NOT NULL,
                price DECIMAL(10, 5),
                sales_channel_id INT
            );
            
            CREATE INDEX IF NOT EXISTS idx_transactions_customer ON transactions(customer_id);
            CREATE INDEX IF NOT EXISTS idx_transactions_article ON transactions(article_id);
            CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(t_dat);
        ";

        await using var cmd = conn.CreateCommand();
        cmd.CommandText = createSql;
        await cmd.ExecuteNonQueryAsync(token);
    }

    private async Task WaitForDatabaseAsync(CancellationToken token)
    {
        const int MaxRetries = 10;
        for (int i = 0; i < MaxRetries; i++)
        {
            try
            {
                using var conn = _dataSource.CreateConnection();
                await conn.OpenAsync(token);
                return;
            }
            catch (NpgsqlException)
            {
                _logger.LogWarning($"Database not ready. Retrying {i + 1}/{MaxRetries}...");
                await Task.Delay(2000, token);
            }
        }
        throw new Exception("Database failed to start.");
    }

    private async Task ImportTableAsync(string tableName, string filePath, CancellationToken token)
    {
        if (!File.Exists(filePath))
        {
            _logger.LogWarning($"Skipping {tableName}: File not found at {filePath}");
            return;
        }

        await using var conn = _dataSource.CreateConnection();
        await conn.OpenAsync(token);

        // Optimization: Use LIMIT 1 to check for existence quickly
        using (var checkCmd = conn.CreateCommand())
        {
            checkCmd.CommandText = $"SELECT 1 FROM {tableName} LIMIT 1";
            var exists = await checkCmd.ExecuteScalarAsync(token);
            if (exists != null)
            {
                _logger.LogInformation($"Skipping {tableName}: Data already exists.");
                return;
            }
        }

        _logger.LogInformation($"Importing {tableName}...");
        
        // Note: The QUOTE '\"' is critical for the detail_desc column which contains commas
        var copyCommand = $"COPY {tableName} FROM STDIN (FORMAT CSV, HEADER, DELIMITER ',', QUOTE '\"')";
        
        using (var writer = await conn.BeginTextImportAsync(copyCommand, token))
        {
            var rowCount = 0;
            foreach (var line in File.ReadLines(filePath))
            {
                await writer.WriteLineAsync(line);
                rowCount++;

                // Logging every 100k rows prevents log spam but keeps you informed
                if (rowCount % 100000 == 0) _logger.LogInformation($"{tableName}: Imported {rowCount:N0} rows...");
            }
        }

        _logger.LogInformation($"Finished {tableName}.");
    }
}