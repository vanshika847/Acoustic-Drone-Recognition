$ErrorActionPreference = "Stop"

$root = Get-Location
$output = Join-Path $root "Acoustic-Drone-Recognition_MANIFESTS.zip"

# Remove old ZIP if it exists
if (Test-Path $output) {
    Remove-Item $output -Force
}

# Important metadata / manifest files
$include = @(
    "datasets\metadata\*.csv",

    "datasets\processed\manifests\*.csv",

    "outputs\features\*_shard_manifest.csv",

    "outputs\diagnostics\*.csv",

    "outputs\error_analysis\*.csv",

    "evaluation_results\*.csv"
)

$files = @()

foreach ($pattern in $include) {
    $matches = Get-ChildItem `
        -Path (Join-Path $root $pattern) `
        -File `
        -ErrorAction SilentlyContinue

    if ($matches) {
        $files += $matches
    }
}

# Remove duplicates
$files = $files | Sort-Object FullName -Unique

if ($files.Count -eq 0) {
    Write-Host ""
    Write-Host "ERROR: No CSV files were found." -ForegroundColor Red
    Write-Host "Project root: $root"
    exit 1
}

Write-Host ""
Write-Host "Files that will be included:" -ForegroundColor Cyan
Write-Host "--------------------------------"

$totalBytes = 0

foreach ($file in $files) {
    $sizeMB = [math]::Round($file.Length / 1MB, 2)
    $totalBytes += $file.Length

    Write-Host ("{0,-80} {1,10} MB" -f `
        $file.FullName.Replace("$root\", ""), `
        $sizeMB)
}

Write-Host ""
Write-Host ("Total source size: {0:N2} MB" -f ($totalBytes / 1MB)) -ForegroundColor Yellow

# Create ZIP
Compress-Archive `
    -Path $files.FullName `
    -DestinationPath $output `
    -CompressionLevel Optimal `
    -Force

$zip = Get-Item $output

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "MANIFEST ZIP CREATED SUCCESSFULLY" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Path:"
Write-Host $zip.FullName
Write-Host ""
Write-Host ("ZIP size: {0:N2} MB" -f ($zip.Length / 1MB))
Write-Host ("Files:    {0}" -f $files.Count)
Write-Host ""
Write-Host "You can now upload this ZIP here."