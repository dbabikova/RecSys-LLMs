# Pre-training the recommender offline

`model-weights.bin` / `model-weights.json` are produced here, outside the browser, so the
page can load a ready model instead of training from scratch on every visit. Training in
the browser is still available as a fallback and via the "Retrain in browser" button.

## Files

- `Trainer.cs` - self-contained matrix factorization trainer (C#, no external packages).
  Mini-batch Adam over `dot(user, movie) + userBias + movieBias`.
- `verify-weights.ps1` - independent checker. Re-parses the binary from scratch and
  re-scores `u.data` without using any trainer code.

## Model layout

Weights are stored as a flat little-endian `float32` array, in this order:

| block | shape | floats |
| --- | --- | --- |
| user factors | `(numUsers+1) x latentDim` | 3776 |
| movie factors | `(numMovies+1) x latentDim` | 6732 |
| user bias | `(numUsers+1) x 1` | 944 |
| movie bias | `(numMovies+1) x 1` | 1683 |

Row `i` of a factor block belongs to id `i`; the row is all zeros for the padding id `0`.

## Usage

```powershell
# compile
Add-Type -TypeDefinition (Get-Content -Raw .\Trainer.cs) -OutputAssembly .\pretrain.exe -OutputType ConsoleApplication

# 1. measure generalisation on a held-out split (80/10/10, shuffled once with a fixed seed)
.\pretrain.exe ..\u.item ..\u.data .\out 4 4 0.005 42 0 validate 256

# 2. refit on all 100000 ratings with the winning hyper-parameters, reusing the
#    test RMSE that step 1 reported
.\pretrain.exe ..\u.item ..\u.data .. 4 4 0.005 42 0 final 256 0.9244

# 3. check the result
powershell -ExecutionPolicy Bypass -File .\verify-weights.ps1
```

Argument order: `<u.item> <u.data> <outDir> [latentDim] [epochs] [lr] [seed] [l2] [mode] [batch] [referenceTestRmse]`.

`mode=validate` holds out 10% of the data and picks the best epoch by validation RMSE.
`mode=final` trains on everything, so it has no hold-out to score and instead records the
test RMSE you pass in, together with a note explaining where that number came from.

## Configuration used for the shipped weights

`latentDim=4`, `epochs=4`, `lr=0.005`, `batch=256`, `seed=42`, `l2=0`.
Held-out test RMSE `0.9244`; in-sample RMSE on the full dataset `0.8370`.

Chosen from a sweep over `latentDim` 3-8 and `lr` 0.003-0.007, confirmed across several
seeds. The dimension is deliberately small: it keeps the blob at 52 KB, which matters more
than the marginal accuracy gain from a wider latent space.

## If the dataset changes

`u.item` / `u.data` are static, but if they are ever replaced, rerun both steps above and
commit the regenerated `model-weights.bin` / `model-weights.json`. The app also guards
against a stale blob at load time: it compares `numUsers`, `numMovies`, `totalRatings` and
the expected float count against the dataset, and falls back to browser training if
anything does not line up.
