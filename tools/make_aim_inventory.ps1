param(
    [string]$Root = ".",
    [string]$Output = "aim_processing_inventory.csv"
)

$rootPath = (Resolve-Path -LiteralPath $Root).Path

$files = Get-ChildItem -LiteralPath $rootPath -Recurse -File -Filter "*.S80" | Sort-Object FullName

$rows = foreach ($file in $files) {
    $relativePath = Resolve-Path -LiteralPath $file.FullName -Relative
    $sameBaseP80 = Join-Path -Path $file.DirectoryName -ChildPath ($file.BaseName + ".p80")
    $status = "raw_root"

    if ($file.FullName -like "*\processed good s80 files\*") {
        $status = "already_processed_folder"
    } elseif ($file.Name -match "(?i)backup|copy") {
        $status = "backup_or_copy"
    } elseif ($file.Length -lt 10000) {
        $status = "tiny_check_file"
    }

    [PSCustomObject]@{
        relative_path = $relativePath
        folder = $file.DirectoryName.Substring($rootPath.Length).TrimStart("\")
        file_name = $file.Name
        size_bytes = $file.Length
        size_mb = [math]::Round($file.Length / 1MB, 3)
        last_modified = $file.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
        has_matching_p80 = Test-Path -LiteralPath $sameBaseP80
        status_hint = $status
    }
}

$rows | Export-Csv -LiteralPath (Join-Path -Path $rootPath -ChildPath $Output) -NoTypeInformation
Write-Host "Wrote $($rows.Count) S80 file records to $Output"
