using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;

// Offline pre-training for the MovieLens matrix-factorization model used by
// week3/script.js. It produces the exact parameter set that createModel() builds
// in the browser, so the page can rebuild the same tf.Layers model and call
// model.setWeights() instead of training.
//
// Parameter layout (identical order to the blob):
//   userEmbedding  : (numUsers+1)  x latentDim
//   movieEmbedding : (numMovies+1) x latentDim
//   userBias       : (numUsers+1)  x 1   (init = global mean rating)
//   movieBias      : (numMovies+1) x 1   (init = 0)
//
// Modes
//   validate : 80/10/10 split. The 10% validation set picks the epoch, the
//              untouched 10% test set is scored exactly once at the end. This
//              is the mode to use when judging a configuration.
//   final    : fits every rating and runs for a fixed number of epochs. Use
//              after 'validate' has chosen the epoch count.
//
// usage:
//   pretrain <u.item> <u.data> <outDir> [latentDim] [epochs] [lr] [seed] [l2] [mode] [batch]
public static class MFPreTrain
{
    const double MEAN_RATING = 3.5;

    public static int Main(string[] args)
    {
        if (args.Length < 3)
        {
            Console.Error.WriteLine("usage: pretrain <u.item> <u.data> <outDir> " +
                "[latentDim] [epochs] [lr] [seed] [l2] [mode] [batch]");
            return 2;
        }

        string itemPath = args[0];
        string ratingPath = args[1];
        string outDir = args[2];
        int latentDim = ArgInt(args, 3, 4);
        int epochs = ArgInt(args, 4, 3);
        double lr = ArgDouble(args, 5, 0.008);
        int seed = ArgInt(args, 6, 42);
        double l2 = ArgDouble(args, 7, 0.0);
        string mode = args.Length > 8 ? args[8] : "validate";
        int batch = ArgInt(args, 9, 256);
        // In 'final' mode there is no hold-out left to score, so the caller
        // passes in the test RMSE that mode=validate measured for the same
        // hyper-parameters. It is recorded with a note, never silently reused.
        double refTestRmse = ArgDouble(args, 10, -1.0);

        // ---- u.item contributes the largest movie id to the embedding size
        int maxMovieId = 0;
        foreach (string line in File.ReadLines(itemPath))
        {
            if (line.Length == 0) continue;
            int bar = line.IndexOf('|');
            if (bar <= 0) continue;
            int id;
            if (int.TryParse(line.Substring(0, bar), NumberStyles.Integer, CultureInfo.InvariantCulture, out id)
                && id > maxMovieId) maxMovieId = id;
        }

        // ---- u.data is the training set
        var users = new List<int>();
        var movies = new List<int>();
        var values = new List<float>();
        int maxUserId = 0;
        foreach (string line in File.ReadLines(ratingPath))
        {
            if (line.Length == 0) continue;
            string[] parts = line.Split('\t');
            if (parts.Length < 3) continue;
            int u, m;
            float r;
            if (!int.TryParse(parts[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out u)) continue;
            if (!int.TryParse(parts[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out m)) continue;
            if (!float.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture, out r)) continue;
            users.Add(u);
            movies.Add(m);
            values.Add(r);
            if (u > maxUserId) maxUserId = u;
            if (m > maxMovieId) maxMovieId = m;
        }

        int n = users.Count;
        int numUsers = maxUserId;
        int numMovies = maxMovieId;
        if (n == 0) { Console.Error.WriteLine("no ratings parsed"); return 1; }

        int uRows = numUsers + 1;
        int mRows = numMovies + 1;
        bool finalMode = mode == "final";
        Console.WriteLine("mode={0} ratings={1} numUsers={2} numMovies={3} latent={4} epochs={5} lr={6} l2={7} batch={8}",
            mode, n, numUsers, numMovies, latentDim, epochs, lr, l2, batch);

        var P = new float[uRows * latentDim];
        var Q = new float[mRows * latentDim];
        var bu = new float[uRows];
        var bv = new float[mRows];

        var rnd = new Random(seed);
        for (int i = 0; i < uRows; i++) bu[i] = (float)MEAN_RATING;
        for (int i = 0; i < P.Length; i++) P[i] = (float)((rnd.NextDouble() * 2.0 - 1.0) * 0.1);
        for (int i = 0; i < Q.Length; i++) Q[i] = (float)((rnd.NextDouble() * 2.0 - 1.0) * 0.1);

        // The split is drawn once and never mixed again. Re-shuffling the whole
        // order each epoch would let hold-out rows back into training and
        // report an optimistic RMSE.
        var all = new int[n];
        for (int i = 0; i < n; i++) all[i] = i;
        Shuffle(all, rnd);

        int[] trainIdx, valIdx, testIdx;
        if (finalMode)
        {
            trainIdx = all;
            valIdx = new int[0];
            testIdx = new int[0];
        }
        else
        {
            int nVal = (int)(n * 0.1);
            int nTest = (int)(n * 0.1);
            valIdx = Slice(all, 0, nVal);
            testIdx = Slice(all, nVal, nTest);
            trainIdx = Slice(all, nVal + nTest, n - nVal - nTest);
        }

        var mP = new float[P.Length]; var vP = new float[P.Length];
        var mQ = new float[Q.Length]; var vQ = new float[Q.Length];
        var mBu = new float[uRows]; var vBu = new float[uRows];
        var mBv = new float[mRows]; var vBv = new float[mRows];

        var gP = new float[P.Length];
        var gQ = new float[Q.Length];
        var gBu = new float[uRows];
        var gBv = new float[mRows];
        var uTouched = new List<int>();
        var mTouched = new List<int>();
        var uMark = new int[uRows];
        var mMark = new int[mRows];
        int mark = 0;

        var bestP = new float[P.Length];
        var bestQ = new float[Q.Length];
        var bestBu = new float[uRows];
        var bestBv = new float[mRows];
        double bestVal = double.MaxValue;
        int bestEpoch = 0;

        const double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
        int step = 0;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            Shuffle(trainIdx, rnd);
            double trainSse = 0.0;
            int trainCount = 0;

            for (int start = 0; start < trainIdx.Length; start += batch)
            {
                int end = Math.Min(start + batch, trainIdx.Length);
                int bs = end - start;
                mark++;
                uTouched.Clear();
                mTouched.Clear();

                for (int k = start; k < end; k++)
                {
                    int idx = trainIdx[k];
                    int u = users[idx], m = movies[idx];
                    int up = u * latentDim, mp = m * latentDim;

                    double dot = bu[u] + bv[m];
                    for (int f = 0; f < latentDim; f++) dot += (double)P[up + f] * Q[mp + f];

                    double err = dot - values[idx];
                    trainSse += err * err;
                    trainCount++;

                    float df = (float)(2.0 * err / bs);

                    if (uMark[u] != mark) { uMark[u] = mark; uTouched.Add(u); }
                    if (mMark[m] != mark) { mMark[m] = mark; mTouched.Add(m); }

                    gBu[u] += df;
                    gBv[m] += df;
                    for (int f = 0; f < latentDim; f++)
                    {
                        gP[up + f] += df * Q[mp + f];
                        gQ[mp + f] += df * P[up + f];
                    }
                }

                step++;
                double bc1 = 1.0 - Math.Pow(beta1, step);
                double bc2 = 1.0 - Math.Pow(beta2, step);

                for (int t = 0; t < uTouched.Count; t++)
                {
                    int u = uTouched[t];
                    int up = u * latentDim;
                    for (int f = 0; f < latentDim; f++)
                    {
                        // L2 on the factors only. Regularising the biases as well
                        // is harmful: l2 * 3.5 dwarfs the 2*err/batch signal, so
                        // Adam would shrink every bias toward 0 on each step.
                        float g = gP[up + f] + (float)l2 * P[up + f];
                        gP[up + f] = 0f;
                        int i = up + f;
                        mP[i] = (float)(beta1 * mP[i] + (1 - beta1) * g);
                        vP[i] = (float)(beta2 * vP[i] + (1 - beta2) * g * g);
                        P[i] -= (float)(lr * (mP[i] / bc1) / (Math.Sqrt(vP[i] / bc2) + eps));
                    }
                    float gbu = gBu[u];
                    gBu[u] = 0f;
                    mBu[u] = (float)(beta1 * mBu[u] + (1 - beta1) * gbu);
                    vBu[u] = (float)(beta2 * vBu[u] + (1 - beta2) * gbu * gbu);
                    bu[u] -= (float)(lr * (mBu[u] / bc1) / (Math.Sqrt(vBu[u] / bc2) + eps));
                }

                for (int t = 0; t < mTouched.Count; t++)
                {
                    int m = mTouched[t];
                    int mp = m * latentDim;
                    for (int f = 0; f < latentDim; f++)
                    {
                        float g = gQ[mp + f] + (float)l2 * Q[mp + f];
                        gQ[mp + f] = 0f;
                        int i = mp + f;
                        mQ[i] = (float)(beta1 * mQ[i] + (1 - beta1) * g);
                        vQ[i] = (float)(beta2 * vQ[i] + (1 - beta2) * g * g);
                        Q[i] -= (float)(lr * (mQ[i] / bc1) / (Math.Sqrt(vQ[i] / bc2) + eps));
                    }
                    float gbv = gBv[m];
                    gBv[m] = 0f;
                    mBv[m] = (float)(beta1 * mBv[m] + (1 - beta1) * gbv);
                    vBv[m] = (float)(beta2 * vBv[m] + (1 - beta2) * gbv * gbv);
                    bv[m] -= (float)(lr * (mBv[m] / bc1) / (Math.Sqrt(vBv[m] / bc2) + eps));
                }
            }

            if (!finalMode)
            {
                double val = Evaluate(valIdx, users, movies, values, P, Q, bu, bv, latentDim);
                if (val < bestVal)
                {
                    bestVal = val;
                    bestEpoch = epoch + 1;
                    Array.Copy(P, bestP, P.Length);
                    Array.Copy(Q, bestQ, Q.Length);
                    Array.Copy(bu, bestBu, bu.Length);
                    Array.Copy(bv, bestBv, bv.Length);
                }
                Console.WriteLine("epoch {0,3}  train RMSE {1:F4}  val RMSE {2:F4}  best {3:F4} @ {4}",
                    epoch + 1, Math.Sqrt(trainSse / Math.Max(1, trainCount)), val, bestVal, bestEpoch);
            }
            else
            {
                Console.WriteLine("epoch {0,3}  train RMSE {1:F4}", epoch + 1,
                    Math.Sqrt(trainSse / Math.Max(1, trainCount)));
            }
        }

        double testRmse = -1;
        string rmseNote = null;
        if (finalMode)
        {
            bestEpoch = epochs;
            testRmse = refTestRmse;
            rmseNote = "measured by mode=validate on a held-out 10% test split with the same " +
                       "hyper-parameters; the shipped weights were then refit on all " + n + " ratings";
        }
        else
        {
            Array.Copy(bestP, P, P.Length);
            Array.Copy(bestQ, Q, Q.Length);
            Array.Copy(bestBu, bu, bu.Length);
            Array.Copy(bestBv, bv, bv.Length);
            // the test set is scored exactly once
            testRmse = Evaluate(testIdx, users, movies, values, P, Q, bu, bv, latentDim);
            Console.WriteLine("BEST epoch {0}  val RMSE {1:F4}  test RMSE {2:F4}", bestEpoch, bestVal, testRmse);
        }

        Directory.CreateDirectory(outDir);

        string binPath = Path.Combine(outDir, "model-weights.bin");
        long totalFloats = P.Length + Q.Length + bu.Length + bv.Length;
        using (var fs = new FileStream(binPath, FileMode.Create, FileAccess.Write))
        using (var bw = new BinaryWriter(fs))
        {
            WriteFloats(bw, P);
            WriteFloats(bw, Q);
            WriteFloats(bw, bu);
            WriteFloats(bw, bv);
        }

        // The scored set is dumped so an independent implementation can re-check
        // the saved blob without re-running the training.
        string holdoutName = finalMode ? null : "holdout.tsv";
        if (holdoutName != null)
        {
            var sb = new StringBuilder();
            foreach (int i in testIdx)
            {
                sb.Append(users[i]).Append('\t').Append(movies[i]).Append('\t')
                  .Append(values[i].ToString("R", CultureInfo.InvariantCulture)).Append('\n');
            }
            File.WriteAllText(Path.Combine(outDir, holdoutName), sb.ToString(), new UTF8Encoding(false));
        }

        var jb = new StringBuilder();
        jb.Append("{\n");
        jb.AppendFormat(CultureInfo.InvariantCulture,
            "  \"numUsers\": {0},\n  \"numMovies\": {1},\n  \"latentDim\": {2},\n",
            numUsers, numMovies, latentDim);
        jb.AppendFormat(CultureInfo.InvariantCulture,
            "  \"totalRatings\": {0},\n  \"trainRatings\": {1},\n", n, trainIdx.Length);
        jb.AppendFormat(CultureInfo.InvariantCulture,
            "  \"epochs\": {0},\n  \"learningRate\": {1},\n  \"l2\": {2},\n  \"batchSize\": {3},\n  \"seed\": {4},\n",
            bestEpoch, lr, l2, batch, seed);
        jb.AppendFormat(CultureInfo.InvariantCulture,
            "  \"validationRmse\": {0:F6},\n  \"testRmse\": {1:F6},\n",
            finalMode ? -1.0 : bestVal, testRmse);
        if (rmseNote != null)
        {
            jb.Append("  \"rmseNote\": \"").Append(rmseNote.Replace("\"", "'")).Append("\",\n");
        }
        jb.AppendFormat(CultureInfo.InvariantCulture, "  \"floatCount\": {0},\n", totalFloats);
        jb.Append("  \"file\": \"model-weights.bin\"\n}\n");
        File.WriteAllText(Path.Combine(outDir, "model-weights.json"), jb.ToString(), new UTF8Encoding(false));

        Console.WriteLine("wrote {0} ({1} bytes)", binPath, totalFloats * 4);
        return 0;
    }

    static int[] Slice(int[] src, int start, int count)
    {
        var r = new int[count];
        Array.Copy(src, start, r, 0, count);
        return r;
    }

    static int ArgInt(string[] a, int i, int def)
    {
        return i < a.Length ? int.Parse(a[i], CultureInfo.InvariantCulture) : def;
    }

    static double ArgDouble(string[] a, int i, double def)
    {
        return i < a.Length ? double.Parse(a[i], CultureInfo.InvariantCulture) : def;
    }

    static void Shuffle(int[] order, Random rnd)
    {
        for (int i = order.Length - 1; i > 0; i--)
        {
            int j = rnd.Next(i + 1);
            int t = order[i]; order[i] = order[j]; order[j] = t;
        }
    }

    static void WriteFloats(BinaryWriter bw, float[] data)
    {
        for (int i = 0; i < data.Length; i++) bw.Write(data[i]);
    }

    static double Evaluate(int[] idx, List<int> users, List<int> movies,
        List<float> values, float[] P, float[] Q, float[] bu, float[] bv, int latentDim)
    {
        double sse = 0.0;
        for (int k = 0; k < idx.Length; k++)
        {
            int i = idx[k];
            int u = users[i], m = movies[i];
            int up = u * latentDim, mp = m * latentDim;
            double dot = bu[u] + bv[m];
            for (int f = 0; f < latentDim; f++) dot += (double)P[up + f] * Q[mp + f];
            double err = dot - values[i];
            sse += err * err;
        }
        return idx.Length == 0 ? 0 : Math.Sqrt(sse / idx.Length);
    }
}
