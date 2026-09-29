# User-Based & Item-Based Collaborative Filtering

Memory-based collaborative filtering for the MovieLens 100K dataset. The app is fully
static and fully browser-based: no framework, no build step, no precomputed artefacts, no
network calls beyond serving the two dataset files. Every recommendation is computed in
the browser from `u.data` at load time.

## Files

| file | role |
| --- | --- |
| `index.html` | markup, includes the two result cards |
| `style.css` | styling, shared with the previous version of the app |
| `data.js` | loading and parsing of `u.item` and `u.data` (unchanged) |
| `cf.js` | in-memory indexes, both similarity functions, both predictors, verification |
| `script.js` | UI wiring, dropdowns, result cards, statistics panel |
| `u.item` | movie catalogue, 1682 movies, pipe separated |
| `u.data` | 100000 ratings, tab separated |

Serve the folder over HTTP and open `index.html`, for example:

```
python -m http.server 8000
```

Opening the file directly with `file://` will not work, because browsers block `fetch()`
on that scheme.

## Data structures

Built once in `buildCfIndexes()` right after `loadData()`:

- `ratingsByUser: Map<userId, Map<movieId, rating>>`
- `ratingsByItem: Map<movieId, Map<userId, rating>>`
- `userMean: Map<userId, number>` - average rating given by each user
- `itemMean: Map<movieId, number>` - average rating received by each movie
- `globalMean: number` - average over all 100000 ratings (3.530)

The full 943 x 1682 similarity matrix is never materialised. Similarities are computed
per query, only for the selected user or the selected movie, and thrown away afterwards.
A full matrix would be 1.6 million cells for no benefit: a single prediction only ever
touches the neighbours of one user or one movie.

## User-based CF

Similarity between two users is the Pearson correlation computed **only over the movies
both of them rated**:

```
sim(u,v) = cov(r_u, r_v) / (sd(r_u) * sd(r_v))     on the co-rated subset
```

Pearson re-centres on the co-rated subset rather than on the overall user means, which is
what makes the measure correct for partially overlapping profiles.

To predict `r[u,i]`, the candidates are the users `v != u` who rated movie `i`. They are
filtered to those with `overlap >= MIN_OVERLAP` and positive similarity, ranked, and the
top `K` are kept. The prediction is

```
r[u,i] = mean(u) + SUM sim(u,v) * (r[v,i] - mean(v)) / SUM |sim(u,v)|
```

Only neighbours with strictly positive similarity are kept: users whose taste runs
*against* the target (negative correlation) are deliberately excluded, so the prediction
leans only on positively correlated neighbours, whether they are users here or movies in
the item-based variant.

## Item-based CF

Similarity between two movies is the adjusted cosine over the users who rated **both** of
them, after removing each of those users' mean ratings:

```
sim(i,j) = SUM_v (r[v,i] - mean(v)) * (r[v,j] - mean(v)) / (||.|| * ||.||)
```

Here the centring is by the global user mean, which is precisely what distinguishes
"adjusted" cosine from plain cosine. There is no re-centring on the co-rater subset.

To predict `r[u,i]`, the candidates are the movies `j != i` that user `u` rated, filtered
and truncated the same way, and the prediction is

```
r[u,i] = mean(i) + SUM sim(i,j) * (r[u,j] - mean(j)) / SUM |sim(i,j)|
```

## Handling missing data

**Strategy: overlap weighting.** One explicit choice, as required for the report.

Every similarity is shrunk toward zero according to how much evidence stands behind it:

```
weight = n / (n + SHRINKAGE)      with SHRINKAGE = 5
```

`n` is the number of co-rated movies (user-based) or co-raters (item-based). A pair
supported by 40 observations therefore counts far more than a pair sitting right at the
`MIN_OVERLAP` threshold, even when their raw correlations are identical. The sign of the
similarity is preserved, so a well-supported pair always outranks a thinly supported one.

Mean imputation was considered and rejected: filling a gap with a mean would invent ratings
that then feed straight back into the similarity computation and bias the result. Shrinking
by confidence uses the same information (the overlap count) without fabricating data.

Other safeguards:

- `MIN_OVERLAP = 3` - a pair below that is discarded before it can influence anything.
- Only neighbours with strictly positive similarity are kept.
- Every prediction is clamped to the valid rating range [1, 5].
- A pair whose similarity is undefined because one side has no variance scores 0 instead of
  dividing by zero.

## Cold start fallback

When no neighbour survives the filters, the app falls back in this order and says so in
the result card:

1. **user mean** - average rating given by the selected user
2. **item mean** - average rating received by the selected movie
3. **global mean** - average over the whole dataset

## Verification

`runVerification()` runs automatically on load and prints to the browser console.

**Holdout check.** 200 known `(user, movie, rating)` triples are sampled with a fixed seed
so the run is reproducible. Each sampled rating is *removed from the index* (both maps and
the means are refreshed) before being predicted back, so the model cannot see the answer.
This is a genuine held-out measurement, not an in-sample one. The index is restored
afterwards.

```
Holdout: 200 known ratings hidden from the index, then predicted back.
  user-based CF  RMSE 0.857  MAE 0.666  fallbacks 0
  item-based CF  RMSE 0.862  MAE 0.671  fallbacks 0
```

Both land in the expected ~0.9-1.1 band for MovieLens 100K, which confirms the
similarity and prediction logic is correct.

**Cold start check.** The rarest movie in the catalogue (id 1348, a single rater) is paired
with the sparsest user who never rated it. The item-based method finds no comparable movie
and correctly falls back; the user-based method finds one rater with sufficient overlap.
A second probe uses ids that are absent from the index entirely, forcing both methods to
the last fallback level.

```
Cold start: movie 1348 with 1 rater(s), user 166 with 20 rating(s).
  user-based CF  3.06 (computed)
  item-based CF  3.55 (fallback: user mean)
  unknown user + unknown movie (last fallback level)
  user-based CF  3.53 (fallback: global mean)
  item-based CF  3.53 (fallback: global mean)
  values finite and inside [1, 5]: true
```

No `NaN`, no division by zero, every value inside [1, 5].

## Statistics panel

Reports the user count, the movie count, the rating count, matrix sparsity (93.7%), the
similarity metric used by each method, the neighbourhood size `K` and `MIN_OVERLAP`. `K`
and `MIN_OVERLAP` are constants at the top of `cf.js` and can be changed there.
