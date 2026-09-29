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
| `script.js` | UI wiring, search boxes, dropdowns, result cards |
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

When no neighbour survives the filters, the app falls back in this order:

1. **user mean** - average rating given by the selected user
2. **item mean** - average rating received by the selected movie
3. **global mean** - average over the whole dataset

The result card does not label which of the three produced the number. A fallback is a
per-prediction implementation detail rather than a property of the method, and the
cold-start probe in the console already reports it explicitly for the cases that matter.

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

### Where accuracy is reported

Each result card shows the predicted rating, a verdict band, the meter, the neighbour count
and the similarity metric. It deliberately carries no observed rating, no absolute error and
no fallback label, because MovieLens 100K is 93.7% empty: out of the
1 586 126 possible `(user, movie)` cells only 100 000 hold a rating, so roughly 94% of all
pairs have no ground truth to compare against. A per-prediction error field would print
"not rated by this user" and "n/a" for almost every pair the user clicks, which reads as a
broken metric rather than an absent measurement.

Accuracy is therefore reported once, over the whole model, by the holdout check above
rather than per click. It runs on 200 known ratings that are hidden from the index before
being predicted back, which is the statistically meaningful way to compare the two
methods: a single pair with one rater says nothing about accuracy, while RMSE over 200
held-out ratings does. The headline figures are printed to the console, and the same
user-based and item-based RMSE values are repeated in the status line under the dropdowns
so they are visible without opening developer tools.

## Interface

The page is deliberately sparse: two text boxes, one button and two result cards. The
dataset figures (943 users, 1682 movies, 100 000 ratings, 93.7% sparsity) and the tuning
constants (`K`, `MIN_OVERLAP`, the two similarity metrics) are not shown as on-screen
chips. They are properties of the model rather than of the current prediction, they never
change while the page is open, and the two similarity metrics are already named on the
result cards. Everything remains visible in the console and in this readme; `K` and
`MIN_OVERLAP` are the constants at the top of `cf.js`.

**Combobox.** There is no separate search field and no `<select>`. Each control is a
single text box with its list attached underneath, so the query and the results it matches
share one window. Typing filters on every keystroke with a case-insensitive substring test
against the visible label: `god` narrows 1682 movies to 6, `toy story` to 1, `19` to the
20 user ids containing those digits. Focusing or clicking the box with empty text opens the
full list. An empty result set shows a single `Nothing found` entry that cannot be picked.

Picking an entry, by click or with `Enter`, writes its label back into the same box and
closes the list. `ArrowDown` and `ArrowUp` move the highlight and wrap around, `Enter` takes
the highlighted entry or the first match if nothing is highlighted yet, and `Escape` closes
the list. `aria-expanded` tracks the list and `aria-activedescendant` tracks the highlight,
so the control is announced correctly as a combobox.

The picked id is stored separately from the visible text. Editing the text after a pick
clears the stored id, so `predictRating()` can never combine a label the user can see with
an id they cannot; it reports that a choice is missing instead. That is also why the
option labels are plain (`User 196`, `Godfather, The`) and the year is appended only in the
context line below the controls, where it cannot be mistaken for part of the query.
