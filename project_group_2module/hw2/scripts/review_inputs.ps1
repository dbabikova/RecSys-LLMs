param([string]$UserAlias = 'U030')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$payloadPath = Join-Path $projectRoot 'data/hw2/common_inputs/payloads.jsonl'
$featurePath = Join-Path $projectRoot 'data/hw2/common_inputs/feature_records.jsonl'
$payloads = @(Get-Content -LiteralPath $payloadPath -Encoding UTF8 | ForEach-Object { $_ | ConvertFrom-Json })
$features = @(Get-Content -LiteralPath $featurePath -Encoding UTF8 | ForEach-Object { $_ | ConvertFrom-Json })
if ($payloads.Count -ne 100 -or $features.Count -ne 2000) { throw 'Wrong input coverage' }
foreach ($payload in $payloads) {
    $rows = @($features | Where-Object { $_.user_alias -ceq $payload.user_alias })
    if ($rows.Count -ne 20 -or $payload.candidates.Count -ne 20) { throw 'Wrong candidate coverage' }
    foreach ($candidate in $payload.candidates) {
        $row = @($rows | Where-Object { $_.candidate_alias -ceq $candidate.asset_alias })
        if ($row.Count -ne 1) { throw 'Missing or duplicate feature row' }
        $row = $row[0]
        if ($row.numeric.source_cbf_rank -ne $candidate.source_cbf_rank -or
            $row.numeric.source_cbf_score -ne $candidate.source_cbf_score -or
            $row.numeric.past_buy_count -ne $candidate.past_buy_count -or
            $row.numeric.last_buy_age_days -ne $candidate.last_buy_age_days -or
            $row.text.candidate_name -cne $candidate.name) { throw 'Candidate information differs' }
        $leftHistory = ConvertTo-Json -InputObject @($row.history_context) -Depth 32 -Compress
        $rightHistory = ConvertTo-Json -InputObject @($payload.history) -Depth 32 -Compress
        if ($leftHistory -cne $rightHistory) { throw 'History information differs' }
    }
}
'PASS: 100 payloads, 2000 feature records; candidate values and retained history match'
$example = @($payloads | Where-Object { $_.user_alias -ceq $UserAlias })
if ($example.Count -ne 1) { throw 'Unknown review alias' }
$example = $example[0]
$example.history_summary | Format-List | Out-Host
$example.profile | ConvertTo-Json -Depth 8 | Out-Host
$example.candidates | Select-Object -First 3 asset_alias,name,source_cbf_rank,past_buy_count,last_buy_age_days |
    Format-Table -AutoSize | Out-Host
