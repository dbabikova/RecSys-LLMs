$ErrorActionPreference = "Stop"
$ci = [System.Globalization.CultureInfo]::InvariantCulture

if ($args.Count -ge 1 -and -not [string]::IsNullOrWhiteSpace($args[0])) { $week2 = $args[0] } else { $week2 = Split-Path -Parent $MyInvocation.MyCommand.Path }
$N = 18

# наборы фильмов (id): Star Wars/ESB/RotJ; Toy Story/Aladdin/Lion King; Scream/Sleepless/Pulp Fiction
$sets = @(
    @{ name = "Star Wars (1977) / Return of the Jedi / Empire Strikes Back"; ids = @(50, 172, 181) },
    @{ name = "Toy Story / Aladdin / Lion King"; ids = @(1, 95, 71) },
    @{ name = "Scream / Sleepless in Seattle / Pulp Fiction"; ids = @(288, 88, 56) }
)
if ($args.Count -ge 2) {
    $sets = @()
    for ($i = 1; $i -lt $args.Count; $i++) {
        $parts = ($args[$i] -split ':')
        $name = $parts[0]
        $ids = ($parts[1] -split ',') | ForEach-Object { [int]$_ }
        $sets += @{ name = $name; ids = $ids }
    }
}

# --- читаем u.item_vector (id|title|18 генов), u.item (title), u.data (число оценок) ---
$vec = @{}   # itemId -> double[18]
$title = @{}
foreach ($line in [System.IO.File]::ReadAllLines((Join-Path $week2 "u.item_vector"))) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    $f = $line.Split('|')
    if ($f.Length -lt ($N + 2)) { continue }
    $id = [int]$f[0]
    $title[$id] = $f[1]
    $v = New-Object 'double[]' $N
    for ($k = 0; $k -lt $N; $k++) { $v[$k] = [double]$f[$k+2] }
    $vec[$id] = $v
}
$ratingCnt = @{}
foreach ($line in [System.IO.File]::ReadAllLines((Join-Path $week2 "u.data"))) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    $f = $line.Split("`t")
    if ($f.Length -lt 3) { continue }
    $it = [int]$f[1]
    $ratingCnt[$it] = ([int]$ratingCnt[$it]) + 1
}

function cosine2([double[]]$a, [double[]]$b) {
    $dot = 0.0; $na = 0.0; $nb = 0.0
    for ($k = 0; $k -lt $a.Length; $k++) { $dot += $a[$k]*$b[$k]; $na += $a[$k]*$a[$k]; $nb += $b[$k]*$b[$k] }
    if ($na -le 0 -or $nb -le 0) { return 0.0 }
    return $dot / ([math]::Sqrt($na) * [math]::Sqrt($nb))
}

# u.recommend (v1) для сверки c первым фильмом набора
$recV1 = @{}
foreach ($line in [System.IO.File]::ReadAllLines((Join-Path $week2 "u.recommend"))) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    $f = $line.Split('|')
    if ($f.Length -lt 6) { continue }
    $recV1[[int]$f[0]] = ($f[1..5] | ForEach-Object { [int]$_ })
}

foreach ($s in $sets) {
    $ids = $s.ids
    $idsStr = ($ids | ForEach-Object { $title[$_] }) -join ' | '
    Write-Host ""
    Write-Host ("===== SET: {0} =====" -f $idsStr)

    # профиль = среднее векторов
    $prof = New-Object 'double[]' $N
    foreach ($id in $ids) { for ($k = 0; $k -lt $N; $k++) { $prof[$k] += $vec[$id][$k] } }
    for ($k = 0; $k -lt $N; $k++) { $prof[$k] = $prof[$k] / $ids.Count }
    Write-Host ("profile = [{0}]" -f (($prof | ForEach-Object { $_.ToString('F4', $ci) }) -join ', '))

    # top-5 по косинусу, просмотренные исключены
    $bestId = New-Object 'int[]' 5
    $bestS  = New-Object 'double[]' 5
    for ($k = 0; $k -lt 5; $k++) { $bestId[$k] = [int]::MaxValue; $bestS[$k] = -1.0 }
    foreach ($id in $vec.Keys) {
        if ($ids -contains $id) { continue }
        $sim = cosine2 $prof $vec[$id]
        if ($sim -gt $bestS[4] -or ($sim -eq $bestS[4] -and $id -lt $bestId[4])) {
            $idx = 4
            while ($idx -gt 0) {
                if ($sim -lt $bestS[$idx-1] -or ($sim -eq $bestS[$idx-1] -and $id -gt $bestId[$idx-1])) { break }
                $bestS[$idx] = $bestS[$idx-1]; $bestId[$idx] = $bestId[$idx-1]
                $idx--
            }
            $bestS[$idx] = $sim; $bestId[$idx] = $id
        }
    }
    Write-Host "Top-5 (profile-based, watched excluded):"
    for ($k = 0; $k -lt 5; $k++) {
        $mId = $bestId[$k]
        Write-Host ("  #{0} {1}  score={2:F4}  ratings_in_u.data={3}" -f ($k+1), $title[$mId], $bestS[$k], $ratingCnt[$mId])
    }

    # сверка c u.recommend (первый фильм набора)
    $first = $ids[0]
    Write-Host ("Compare with u.recommend for first movie ({0}):" -f $title[$first])
    $fRec = $recV1[$first]
    for ($k = 0; $k -lt 5; $k++) {
        $mId = $fRec[$k]
        Write-Host ("  u.recommend #{0} {1}  ratings={2}" -f ($k+1), $title[$mId], $ratingCnt[$mId])
    }
    $alreadyAny = ($bestId | Where-Object { $fRec -contains $_ }).Count
    Write-Host ("  overlap with profile Top-5: {0} of 5" -f $alreadyAny)
}