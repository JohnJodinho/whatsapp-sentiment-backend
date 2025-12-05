# ==============================================================================
# AUTOMATED REDIS CONFIGURATION UPDATE SCRIPT
# ==============================================================================

# 1. Configuration Variables
$ResourceGroup = "rg-sentimentscope"
$RedisName     = "redis-whatsapp9454"
$RedisHost     = "$RedisName.redis.cache.windows.net"
$RedisPort     = "6380" # SSL Port

# List of all apps that need the update
$ContainerApps = @(
    "sentiment-api", 
    "worker-sentiment", 
    "worker-embeddings"
)

# 2. Fetch the Redis Access Key
Write-Host "Fetching access keys for $RedisName..." -ForegroundColor Cyan
try {
    $PrimaryKey = az redis list-keys --name $RedisName --resource-group $ResourceGroup --query primaryKey --output tsv
    
    if (-not $PrimaryKey) {
        Write-Error "Failed to retrieve Redis key. Make sure you are logged in via 'az login'."
        exit
    }
    Write-Host "Successfully retrieved Redis Key." -ForegroundColor Green
}
catch {
    Write-Error "An error occurred while fetching the key: $_"
    exit
}

# 3. Construct the Secure Connection String
# Note: 'rediss://' (double s) is required for SSL/Port 6380
$RedisUrl = "rediss://:$($PrimaryKey)@$($RedisHost):$($RedisPort)/0?ssl_cert_reqs=none"

# 4. Define the Environment Variables to Update
$EnvVars = @(
    "CELERY_BROKER_URL=$RedisUrl",
    "CELERY_RESULT_BACKEND=$RedisUrl",
    "REDIS_URL=$RedisUrl"
)

# 5. Loop through Apps and Update
foreach ($App in $ContainerApps) {
    Write-Host "------------------------------------------------------"
    Write-Host "Updating Environment Variables for: $App" -ForegroundColor Yellow
    
    try {
        az containerapp update `
            --name $App `
            --resource-group $ResourceGroup `
            --set-env-vars $EnvVars
            
        Write-Host "✅ Success: $App updated." -ForegroundColor Green
    }
    catch {
        Write-Error "❌ Failed to update $App. Error: $_"
    }
}

Write-Host "------------------------------------------------------"
Write-Host "All updates complete!" -ForegroundColor Cyan