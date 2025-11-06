# PowerShell helper to run the Streamlit app with .env loaded
if (Test-Path .env) {
    Write-Host "Loading .env"
    Get-Content .env | ForEach-Object {
        if ($_ -match "^\s*#") { return }
        $parts = $_ -split '='
        if ($parts.Count -ge 2) {
            $name = $parts[0].Trim()
            $value = ($parts[1..($parts.Count-1)] -join '=').Trim()
            Set-Item -Path Env:\$name -Value $value
        }
    }
}

Write-Host "Starting Streamlit..."
streamlit run src/app.py
