$ErrorActionPreference = "Stop"
$ci = [System.Globalization.CultureInfo]::InvariantCulture
if ($args.Count -ge 1 -and -not [string]::IsNullOrWhiteSpace($args[0])) { $week2 = $args[0] } else { $week2 = Split-Path -Parent $MyInvocation.MyCommand.Path }

function popcount([int]$x) {
    $c = 0
    while ($x -ne 0) { $c += ($x -band 1); $x = $x -shr 1 }
    return $c
}

# ---------- СЃС‚СЂСѓРєС‚СѓСЂРЅС‹Рµ РїСЂРѕРІРµСЂРєРё ----------
Write-Host "=== STRUCTURE ==="
foreach ($name in "u.recommend","u.recommend_2") {
    $lines = [System.IO.File]::ReadAllLines((Join-Path $week2 $name)) | Where-Object { $_ -ne "" }
    $keys = New-Object System.Collections.Generic.HashSet[int]
    $badField = 0; $self = 0; $dupe = 0; $outOfRange = 0
    foreach ($l in $lines) {
        $f = $l.Split('|')
        if ($f.Length -ne 6) { $badField++; continue }
        $k = [int]$f[0]
        [void]$keys.Add($k)
        $seen = New-Object System.Collections.Generic.HashSet[int]
        for ($r = 1; $r -lt 6; $r++) {
            $rid = [int]$f[$r]
            if ($rid -lt 1 -or $rid -gt 1682) { $outOfRange++; continue }
            if ($rid -eq $k) { $self++ }
            if (-not $seen.Add($rid)) { $dupe++ }
        }
    }
    $cnt = $lines.Count
    $missing = @(); for ($m = 1; $m -le 1682; $m++) { if (-not $keys.Contains($m)) { $missing += $m } }
    Write-Host ("{0}: lines={1} fields_ok(6)={2} missing_keys={3} (sample {4}) badFields={5} self={6} dupes={7} outOfRange={8}" -f `
        $name, $cnt, (($lines | Where-Object { ($_ -split '\|').Length -eq 6 }).Count), $missing.Count, (($missing | Select-Object -First 5) -join ','), $badField, $self, $dupe, $outOfRange)
}

# libraries of file recs
$recMap1 = @{}; $recMap2 = @{}
foreach ($l in [System.IO.File]::ReadAllLines((Join-Path $week2 "u.recommend"))) {
    if ($l -eq "") { continue }
    $f = $l.Split('|'); $recMap1[[int]$f[0]] = ($f[1..5] | ForEach-Object { [int]$_ })
}
foreach ($l in [System.IO.File]::ReadAllLines((Join-Path $week2 "u.recommend_2"))) {
    if ($l -eq "") { continue }
    $f = $l.Split('|'); $recMap2[[int]$f[0]] = ($f[1..5] | ForEach-Object { [int]$_ })
}

# ---------- V1: РїРµСЂРµСЃС‡С‘С‚ РёР· СЃС‹СЂРѕРіРѕ u.item (Р¶Р°РЅСЂС‹) ----------
Write-Host "`n=== VARIANT 1 (u.recommend) ==="
$titles = @{}
$mask = @{}
foreach ($line in [System.IO.File]::ReadAllLines((Join-Path $week2 "u.item"))) {
    if ($line -eq "") { continue }
    $f = $line.Split('|')
    $id = [int]$f[0]
    $titles[$id] = $f[1]
    $m = 0
    for ($k = 0; $k -lt 18; $k++) { if ([int]$f[$k+6] -eq 1) { $m = $m -bor (1 -shl $k) } }
    $mask[$id] = $m
}
$norm = @{}
foreach ($id in $mask.Keys) { $norm[$id] = [math]::Sqrt([double](popcount $mask[$id])) }
$ids = ($mask.Keys | Sort-Object)

$setMism1 = 0; $orderDiff1 = 0; $mismSamples1 = @()
foreach ($i in $ids) {
    $bestId = New-Object 'int[]' 5
    $bestS  = New-Object 'double[]' 5
    for ($k = 0; $k -lt 5; $k++) { $bestId[$k] = [int]::MaxValue; $bestS[$k] = -1.0 }
    $ni = $norm[$i]; $mi = $mask[$i]
    foreach ($j in $ids) {
        if ($j -eq $i) { continue }
        $sim = 0.0
        if ($ni -gt 0 -and $norm[$j] -gt 0) { $sim = (popcount ($mi -band $mask[$j])) / ($ni * $norm[$j]) }
        if ($sim -gt $bestS[4] -or ($sim -eq $bestS[4] -and $j -lt $bestId[4])) {
            $idx = 4
            while ($idx -gt 0) {
                if ($sim -lt $bestS[$idx-1] -or ($sim -eq $bestS[$idx-1] -and $j -gt $bestId[$idx-1])) { break }
                $bestS[$idx] = $bestS[$idx-1]; $bestId[$idx] = $bestId[$idx-1]
                $idx--
            }
            $bestS[$idx] = $sim; $bestId[$idx] = $j
        }
    }
    $top = $bestId
    $file = $recMap1[$i]
    $sameSet = ([string]::Join(',', ($top | Sort-Object)) -eq [string]::Join(',', ($file | Sort-Object)))
    if (-not $sameSet) {
        $setMism1++
        if ($mismSamples1.Count -lt 3) { $mismSamples1 += ("movie {0} ({1}): file=[{2}] vs recompute=[{3}]" -f $i, $titles[$i], ($file -join ','), ($top -join ',')) }
    } elseif ([string]::Join(',', $top) -ne [string]::Join(',', $file)) { $orderDiff1++ }
}
Write-Host ("set mismatches = {0}, order-only diffs = {1}, checked = {2}" -f $setMism1, $orderDiff1, $ids.Count)
if ($mismSamples1.Count -gt 0) { $mismSamples1 }

# ---------- V2: РїРµСЂРµСЃС‡С‘С‚ РёР· СЃС‹СЂРѕРіРѕ u.data (adjusted cosine) ----------
Write-Host "`n=== VARIANT 2 (u.recommend_2) ==="
$uSum = @{}; $uCnt = @{}
foreach ($line in [System.IO.File]::ReadAllLines((Join-Path $week2 "u.data"))) {
    if ($line -eq "") { continue }
    $f = $line.Split("`t")
    $uSum[[int]$f[0]] = ([double]$uSum[[int]$f[0]]) + [double]($f[2].Replace(',','.'))
    $uCnt[[int]$f[0]] = ([int]$uCnt[[int]$f[0]]) + 1
}
$uMean = @{}
foreach ($u in $uSum.Keys) { $uMean[$u] = $uSum[$u] / $uCnt[$u] }

$itemVec = @{}; $userItems = @{}
foreach ($line in [System.IO.File]::ReadAllLines((Join-Path $week2 "u.data"))) {
    if ($line -eq "") { continue }
    $f = $line.Split("`t")
    $u = [int]$f[0]; $it = [int]$f[1]
    $c = [double]($f[2].Replace(',','.')) - $uMean[$u]
    if (-not $itemVec.ContainsKey($it)) { $itemVec[$it] = @{} }
    $itemVec[$it][$u] = $c
    if (-not $userItems.ContainsKey($u)) { $userItems[$u] = New-Object System.Collections.Generic.List[object] }
    $userItems[$u].Add(@{ itemId = $it; c = $c })
}
$norm2 = @{}
foreach ($it in $itemVec.Keys) { $s=0.0; foreach ($c in $itemVec[$it].Values) { $s += $c*$c }; $norm2[$it] = [math]::Sqrt($s) }

$ids2 = ($itemVec.Keys | Sort-Object)
$stamp = @{}; $stampVal = 0; $dot = @{}
$setMism2 = 0; $orderDiff2 = 0; $mismSamples2 = @()
foreach ($i in $ids2) {
    $stampVal++
    $touched = @()
    foreach ($u in $itemVec[$i].Keys) {
        $ci = $itemVec[$i][$u]
        foreach ($e in $userItems[$u]) {
            $j = $e.itemId
            if ($j -eq $i) { continue }
            if ($stamp[$j] -ne $stampVal) { $stamp[$j] = $stampVal; $dot[$j] = 0.0; $touched += $j }
            $dot[$j] += $ci * $e.c
        }
    }
$bestId = New-Object 'int[]' 5
    $bestS  = New-Object 'double[]' 5
    for ($k = 0; $k -lt 5; $k++) { $bestId[$k] = [int]::MaxValue; $bestS[$k] = -1.0 }
    $nI = $norm2[$i]
    foreach ($j in $touched) {
        $sim = 0.0
        if ($nI -gt 0 -and $norm2[$j] -gt 0) { $sim = $dot[$j] / ($nI * $norm2[$j]) }
        if ($sim -gt $bestS[4] -or ($sim -eq $bestS[4] -and $j -lt $bestId[4])) {
            $idx = 4
            while ($idx -gt 0) {
                if ($sim -lt $bestS[$idx-1] -or ($sim -eq $bestS[$idx-1] -and $j -gt $bestId[$idx-1])) { break }
                $bestS[$idx] = $bestS[$idx-1]; $bestId[$idx] = $bestId[$idx-1]
                $idx--
            }
            $bestS[$idx] = $sim; $bestId[$idx] = $j
        }
    }
    $top = $bestId
    $file = $recMap2[$i]
    $sameSet = ([string]::Join(',', ($top | Sort-Object)) -eq [string]::Join(',', ($file | Sort-Object)))
    if (-not $sameSet) {
        $setMism2++
        if ($mismSamples2.Count -lt 3) { $mismSamples2 += ("movie {0} ({1}): file=[{2}] vs recompute=[{3}]" -f $i, $titles[$i], ($file -join ','), ($top -join ',')) }
    } elseif ([string]::Join(',', $top) -ne [string]::Join(',', $file)) { $orderDiff2++ }
}
Write-Host ("set mismatches = {0}, order-only diffs = {1}, checked = {2}" -f $setMism2, $orderDiff2, $ids2.Count)
if ($mismSamples2.Count -gt 0) { $mismSamples2 }
