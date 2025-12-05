# --- CONFIGURATION ---
$resourceGroup = "rg-sentimentscope" 
# List your 3 container names below
$containerApps = @("worker-sentiment", "worker-embeddings", "sentiment-api") 
$envFilePath = ".env"

# --- LOGIC ---

# 1. Read .env, ignore comments (#) and empty lines, and trim whitespace
$envVars = Get-Content $envFilePath | 
    Where-Object { $_ -match '=' -and $_ -notmatch '^\s*#' } | 
    ForEach-Object { $_.Trim() }

if ($envVars.Count -eq 0) {
    Write-Error "No variables found in $envFilePath. Please check the file."
    exit
}

# 2. Loop through each container and perform the update
foreach ($app in $containerApps) {
    Write-Host "`nUpdating $app..." -ForegroundColor Cyan
    
    # --set-env-vars performs a merge (add/update), preserving other existing vars
    az containerapp update `
        --name $app `
        --resource-group $resourceGroup `
        --set-env-vars $envVars

    if ($LASTEXITCODE -eq 0) {
        Write-Host "Successfully updated $app" -ForegroundColor Green
    } else {
        Write-Host "Failed to update $app" -ForegroundColor Red
    }
}