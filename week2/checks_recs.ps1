$ErrorActionPreference = "Stop"
$ci = [System.Globalization.CultureInfo]::InvariantCulture
if ($args.Count -ge 1 -and -not [string]::IsNullOrWhiteSpace($args[0])) { $week2 = $args[0] } else { $week2 = Split-Path -Parent $MyInvocation.MyCommand.Path }
$N = 18

function MedianOf([double[]]$arr) {
    if ($arr.Length -eq 0) { return 0.0 }
    $sorted = $arr | Sort-Object
    $m = [math]::Floor($sorted.Count / 2)
    if ($sorted.Count % 2 -eq 1) { return [double]$sorted[$m] }
    return ([double]$sorted[$m-1] + [double]$sorted[$m]) / 2
}

# --- load u.item_vector -> vectors + genre counts; u.item -> titles; u.data -> rating counts + per-user rated sets ---
$vec = @{}; $title = @{}; $gcnt = @{}
foreach ($line in [System.IO.File]::ReadAllLines((Join-Path $week2 "u.item_vector"))) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    $f = $line.Split('|')
    if ($f.Length -lt 20) { continue }
    $id = [int]$f[0]; $title[$id] = $f[1]
    $v = New-Object 'double[]' $N; $c = 0
    for ($k = 0; $k -lt $N; $k++) { $v[$k] = [double]$f[$k+2]; if ($v[$k] -gt 0) { $c++ } }
    $vec[$id] = $v; $gcnt[$id] = $c
}
foreach ($line in [System.IO.File]::ReadAllLines((Join-Path $week2 "u.item"))) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    $f = $line.Split('|')
    if ($f.Length -lt 2) { continue }
    $title[[int]$f[0]] = $f[1]
}
$ratingCnt = @{}; $userRated = @{}
foreach ($line in [System.IO.File]::ReadAllLines((Join-Path $week2 "u.data"))) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    $f = $line.Split("`t")
    if ($f.Length -lt 3) { continue }
    $u = [int]$f[0]; $it = [int]$f[1]
    $ratingCnt[$it] = ([int]$ratingCnt[$it]) + 1
    if (-not $userRated.ContainsKey($u)) { $userRated[$u] = @{} }
    $userRated[$u][$it] = $true
}

$catalogCounts = @(foreach ($id in $vec.Keys) { [double]$ratingCnt[$id] })
$catalogMedian = MedianOf $catalogCounts
Write-Host ""
Write-Host ("catalog: movies={0} median_ratings={1}" -f $catalogCounts.Count, $catalogMedian.ToString('F1',$ci))

function CollectTop5Popularity($lists) {
    # $lists: collection of arrays of 5 movie ids
    $all = New-Object System.Collections.Generic.List[double]
    $seen = New-Object 'System.Collections.Generic.HashSet[int]'
    foreach ($list in $lists) {
        foreach ($r in $list) {
            $all.Add([double]$ratingCnt[$r])
            [void]$seen.Add($r)
        }
    }
    return @{ mean = ($all | Measure-Object -Average).Average; median = MedianOf ($all.ToArray()); unique = $seen.Count }
}

# --- (c,d): u.recommend ---
$rec1 = @()
foreach ($line in [System.IO.File]::ReadAllLines((Join-Path $week2 "u.recommend"))) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    $f = $line.Split('|')
    if ($f.Length -lt 6) { continue }
    $rec1 += ,($f[1..5] | ForEach-Object { [int]$_ })
}
Write-Host ""
$r1 = CollectTop5Popularity $rec1
Write-Host ("u.recommend:   top5_ratingcount mean={0:F1} median={1:F1}  unique_movies={2}" -f $r1.mean, $r1.median, $r1.unique)

# --- (c,d): u.recommend_2 ---
$rec2 = @()
foreach ($line in [System.IO.File]::ReadAllLines((Join-Path $week2 "u.recommend_2"))) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    $f = $line.Split('|')
    if ($f.Length -lt 6) { continue }
    $rec2 += ,($f[1..5] | ForEach-Object { [int]$_ })
}
$r2 = CollectTop5Popularity $rec2
Write-Host ("u.recommend_2: top5_ratingcount mean={0:F1} median={1:F1}  unique_movies={2}" -f $r2.mean, $r2.median, $r2.unique)

# --- (a,c,d,e): profile-based for all 943 users from u.data_profile ---
function Cosine2([double[]]$a, [double[]]$b) {
    $dot = 0.0; $na = 0.0; $nb = 0.0
    for ($k = 0; $k -lt $a.Length; $k++) { $dot += $a[$k]*$b[$k]; $na += $a[$k]*$a[$k]; $nb += $b[$k]*$b[$k] }
    if ($na -le 0 -or $nb -le 0) { return 0.0 }
    return $dot / ([math]::Sqrt($na) * [math]::Sqrt($nb))
}
$userCount = 0; $violations = 0
$profPop = New-Object System.Collections.Generic.List[double]
$profSeen = New-Object 'System.Collections.Generic.HashSet[int]'
$dotSeen = New-Object 'System.Collections.Generic.HashSet[int]'
$dotPop = New-Object System.Collections.Generic.List[double]
$profGenreSum = 0.0; $dotGenreSum = 0.0
foreach ($line in [System.IO.File]::ReadAllLines((Join-Path $week2 "u.data_profile"))) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    $f = $line.Split('|')
    if ($f.Length -lt 19) { continue }
    $u = [int]$f[0]
    $prof = New-Object 'double[]' $N
    for ($k = 0; $k -lt $N; $k++) { $prof[$k] = [double]::Parse($f[$k+1], $ci) }
    $rated = $userRated[$u]
    $userCount++

    # parallel: manual top-5 for cosine and for dot product (no normalization)
    $bcid = New-Object 'int[]' 5; $bcs = New-Object 'double[]' 5; for ($k=0;$k -lt 5;$k++){ $bcid[$k]=[int]::MaxValue; $bcs[$k]=-1.0 }
    $bdid = New-Object 'int[]' 5; $bds = New-Object 'double[]' 5; for ($k=0;$k -lt 5;$k++){ $bdid[$k]=[int]::MaxValue; $bds[$k]=-1.0 }
    foreach ($id in $vec.Keys) {
        if ($rated.ContainsKey($id)) { continue }
        $v = $vec[$id]
        $dot = 0.0; $na = 0.0; $nb = 0.0
        for ($k = 0; $k -lt $N; $k++) { $dot += $prof[$k]*$v[$k]; $na += $prof[$k]*$prof[$k]; $nb += $v[$k]*$v[$k] }
        $sim = ($dot / ([math]::Sqrt($na) * [math]::Sqrt($nb)))
        if ($sim -gt $bcs[4] -or ($sim -eq $bcs[4] -and $id -lt $bcid[4])) { $p=4; while($p -gt 0){ if($sim -lt $bcs[$p-1]){ break }; if($sim -eq $bcs[$p-1] -and $id -gt $bcid[$p-1]){ break }; $bcs[$p]=$bcs[$p-1]; $bcid[$p]=$bcid[$p-1]; $p-- }; $bcs[$p]=$sim; $bcid[$p]=$id }
        if ($dot -gt $bds[4] -or ($dot -eq $bds[4] -and $id -lt $bdid[4])) { $p=4; while($p -gt 0){ if($dot -lt $bds[$p-1]){ break }; if($dot -eq $bds[$p-1] -and $id -gt $bdid[$p-1]){ break }; $bds[$p]=$bds[$p-1]; $bdid[$p]=$bdid[$p-1]; $p-- }; $bds[$p]=$dot; $bdid[$p]=$id }
    }
    if ($userCount -le 3) { Write-Host ("  [example u{0}] rated={1} profileTop5={2} | dotTop5={3}" -f $u, $rated.Count, (($bcid|ForEach-Object{$_}) -join ','), (($bdid|ForEach-Object{$_}) -join ',')) }
    foreach ($k in 0..4) {
        if ($rated.ContainsKey($bcid[$k])) { $violations++ }   # (a)
        $profPop.Add([double]$ratingCnt[$bcid[$k]]); [void]$profSeen.Add($bcid[$k])
        $profGenreSum += $gcnt[$bcid[$k]]
        $dotPop.Add([double]$ratingCnt[$bdid[$k]]); [void]$dotSeen.Add($bdid[$k])
        $dotGenreSum += $gcnt[$bdid[$k]]
    }
}
Write-Host ""
Write-Host ("profile-based: users={0} overlap_violations(a)={1}" -f $userCount, $violations)
Write-Host ("profile-based top5 (cosine): mean={0:F1} median={1:F1} unique={2}" -f ($profPop | Measure-Object -Average).Average, (MedianOf ($profPop.ToArray())), $profSeen.Count)
Write-Host ("profile-based top5 (dot)   : mean={0:F1} median={1:F1} unique={2}" -f ($dotPop | Measure-Object -Average).Average, (MedianOf ($dotPop.ToArray())), $dotSeen.Count)
Write-Host ""
Write-Host ("(e) avg genres in top-5: cosine-top5={0:F2}  dot-top5={1:F2}" -f ($profGenreSum/($userCount*5)), ($dotGenreSum/($userCount*5)))

# --- (b) cosine of each watched movie with its set profile (3 example sets) ---
$sets = @(@(50,172,181), @(1,95,71), @(288,88,56))
Write-Host ""
Write-Host "(b) cos(watched movie, set profile):"
foreach ($s in $sets) {
    $prof = New-Object 'double[]' $N
    foreach ($id in $s) { for ($k=0;$k -lt $N;$k++){ $prof[$k] += $vec[$id][$k] } }
    for ($k=0;$k -lt $N;$k++){ $prof[$k] = $prof[$k]/$s.Count }
    $line = "  set (" + (($s | ForEach-Object { $title[$_] }) -join ' | ') + "): "
    $line += (($s | ForEach-Object { "{0}:{1:F4}" -f $title[$_], (Cosine2 $prof $vec[$_]) }) -join '  ')
    Write-Host $line
}