# Independent check of the pre-trained weights.
#
# Re-parses model-weights.bin from scratch (no trainer code involved) to confirm
# the layout matches model-weights.json, then re-scores ratings directly from the
# binary. The shipped blob is refit on the full dataset, so the numbers below are
# IN-SAMPLE on purpose - the honest generalisation figure is `testRmse` in the
# header, which the trainer measured on a held-out 10% split.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File verify-weights.ps1
#   powershell -ExecutionPolicy Bypass -File verify-weights.ps1 -Samples 10

param(
    [string]$Header = (Join-Path $PSScriptRoot '..\model-weights.json'),
    [string]$Binary = (Join-Path $PSScriptRoot '..\model-weights.bin'),
    [string]$Ratings = (Join-Path $PSScriptRoot '..\u.data'),
    [int]$Samples = 6
)

$ErrorActionPreference = 'Stop'

$meta = Get-Content -LiteralPath $Header -Raw | ConvertFrom-Json
$latentDim = [int]$meta.latentDim
$numUsers = [int]$meta.numUsers
$numMovies = [int]$meta.numMovies

$bytes = [System.IO.File]::ReadAllBytes($Binary)
$floatCount = $bytes.Length / 4

$uRows = $numUsers + 1
$mRows = $numMovies + 1
$expected = $uRows * $latentDim + $mRows * $latentDim + $uRows + $mRows

'--- layout ---'
"users x movies : $numUsers x $numMovies"
"latent dim     : $latentDim"
"blob bytes     : $($bytes.Length)"
"float count    : $floatCount"
"expected floats: $expected"
"header testRmse: $($meta.testRmse)  (held-out, measured by the trainer)"

if ($floatCount -ne $expected) {
    throw "float count $floatCount does not match the header layout ($expected)"
}
if ([int]$meta.floatCount -ne $expected) {
    throw "header floatCount $($meta.floatCount) does not match the layout ($expected)"
}

# Element offsets into the flat float32 array.
$uOff = 0
$mOff = $uRows * $latentDim
$buOff = $mOff + $mRows * $latentDim
$bvOff = $buOff + $uRows

function Get-F32([int]$i) { [BitConverter]::ToSingle($bytes, $i * 4) }

function Get-Score([int]$u, [int]$m) {
    if ($u -lt 0 -or $u -gt $numUsers) { throw "user id $u out of range" }
    if ($m -lt 0 -or $m -gt $numMovies) { throw "movie id $m out of range" }
    $score = [double](Get-F32 ($buOff + $u)) + [double](Get-F32 ($bvOff + $m))
    $up = $uOff + $u * $latentDim
    $mp = $mOff + $m * $latentDim
    for ($f = 0; $f -lt $latentDim; $f++) {
        $score += [double](Get-F32 ($up + $f)) * [double](Get-F32 ($mp + $f))
    }
    return $score
}

# --- score the dataset straight from the blob ---
$rows = Get-Content -LiteralPath $Ratings | Where-Object { $_.Trim() -ne '' }
$sse = 0.0
$absSum = 0.0
$sum = 0.0
$min = [double]::MaxValue
$max = [double]::MinValue
$outside = 0
$sampleRows = @()

for ($i = 0; $i -lt $rows.Count; $i++) {
    $p = $rows[$i] -split '\s+'
    $u = [int]$p[0]
    $m = [int]$p[1]
    $target = [double]::Parse($p[2], [Globalization.CultureInfo]::InvariantCulture)

    $score = Get-Score $u $m
    $err = $score - $target
    $sse += $err * $err
    $absSum += [Math]::Abs($err)
    $sum += $score
    if ($score -lt $min) { $min = $score }
    if ($score -gt $max) { $max = $score }
    if ($score -lt 1.0 -or $score -gt 5.0) { $outside++ }
    if ($sampleRows.Count -lt $Samples) { $sampleRows += [pscustomobject]@{ user = $u; movie = $m; actual = $target; predicted = $score } }
}

$n = [double]$rows.Count
'--- in-sample on u.data (not held-out) ---'
"ratings scored : $($rows.Count)"
"RMSE           : {0:F4}" -f [Math]::Sqrt($sse / $n)
"MAE            : {0:F4}" -f ($absSum / $n)
"mean prediction: {0:F4}" -f ($sum / $n)
"pred range     : {0:F3} .. {1:F3}" -f $min, $max
"outside 1..5   : $outside of $($rows.Count)"
''
'--- sample predictions ---'
$sampleRows | Format-Table -AutoSize
