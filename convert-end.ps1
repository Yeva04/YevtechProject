# convert-line-endings.ps1
Get-ChildItem -Include *.py,*.csv -Recurse | ForEach-Object {
    $content = Get-Content $_.FullName -Raw
    if ($content -ne $null) {
        $content = $content -replace "`r`n", "`n"
        Set-Content $_.FullName -Value $content -NoNewline
        Write-Host "Converted line endings to LF for: $($_.Name)"
    }
}