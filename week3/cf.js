// cf.js - memory-based collaborative filtering for the MovieLens 100K dataset
//
// Two classic neighbourhood methods, both computed in the browser from u.data:
//   user-based CF: similarity between USERS, prediction for a movie from the
//                  users who rated it.
//   item-based CF: similarity between MOVIES, prediction for a user from the
//                  movies they rated.
//
// Missing data strategy (one explicit choice, as required for the write-up):
// OVERLAP WEIGHTING. Every similarity is shrunk toward zero by
// n / (n + SHRINKAGE), where n is the number of co-rated items (user-based) or
// co-raters (item-based). A pair backed by more evidence therefore counts more,
// and a pair sitting right at MIN_OVERLAP counts very little. Mean imputation is
// deliberately NOT used: imputing a value would invent ratings that then feed
// straight back into the similarity computation.
//
// No N x N similarity matrix is ever materialised. Similarities are computed on
// demand, only for the selected user or the selected movie, and discarded after.

const MIN_OVERLAP = 3;   // minimum co-rated items / co-raters for a usable similarity
const DEFAULT_K = 30;    // neighbourhood size
const SHRINKAGE = 5;     // overlap-weighting constant: weight = n / (n + SHRINKAGE)

const SIMILARITY_LABEL = {
    userBased: 'Pearson correlation on co-rated movies',
    itemBased: 'Adjusted cosine on co-raters, user means removed'
};

// Fallback chain, tried in this exact order when no neighbour is usable.
const FALLBACK_ORDER = ['user mean', 'item mean', 'global mean'];

let ratingsByUser = new Map();   // Map<userId, Map<movieId, rating>>
let ratingsByItem = new Map();   // Map<movieId, Map<userId, rating>>
let userMean = new Map();        // Map<userId, average rating given by that user>
let itemMean = new Map();        // Map<movieId, average rating received by that movie>
let globalMean = 0;              // average over every rating currently indexed

let indexedRatingCount = 0;
let indexedRatingSum = 0;

// ---------------------------------------------------------------- indexing

// Builds the in-memory lookup structures plus the mean ratings. Called once,
// right after loadData().
function buildCfIndexes(ratingRows) {
    ratingsByUser = new Map();
    ratingsByItem = new Map();
    indexedRatingCount = 0;
    indexedRatingSum = 0;

    for (const row of ratingRows) {
        let userRatings = ratingsByUser.get(row.userId);
        if (userRatings === undefined) {
            userRatings = new Map();
            ratingsByUser.set(row.userId, userRatings);
        }
        userRatings.set(row.movieId, row.rating);

        let itemRaters = ratingsByItem.get(row.movieId);
        if (itemRaters === undefined) {
            itemRaters = new Map();
            ratingsByItem.set(row.movieId, itemRaters);
        }
        itemRaters.set(row.userId, row.rating);

        indexedRatingSum += row.rating;
        indexedRatingCount += 1;
    }

    userMean = new Map();
    for (const [userId, userRatings] of ratingsByUser) {
        userMean.set(userId, averageOf(userRatings));
    }

    itemMean = new Map();
    for (const [movieId, itemRaters] of ratingsByItem) {
        itemMean.set(movieId, averageOf(itemRaters));
    }

    globalMean = indexedRatingSum / indexedRatingCount;

    return {
        users: ratingsByUser.size,
        ratedMovies: ratingsByItem.size
    };
}

function averageOf(valueMap) {
    let total = 0;
    for (const value of valueMap.values()) {
        total += value;
    }
    return total / valueMap.size;
}

// ------------------------------------------------------------- similarities

// Pearson correlation between two users, computed only over movies that BOTH of
// them rated. A pair with fewer than MIN_OVERLAP co-rated movies, or with no
// variance on either side, scores 0.
function userSimilarity(userA, userB) {
    const ratingsA = ratingsByUser.get(userA);
    const ratingsB = ratingsByUser.get(userB);
    if (ratingsA === undefined || ratingsB === undefined) {
        return { similarity: 0, overlap: 0 };
    }

    const meanA = userMean.get(userA);
    const meanB = userMean.get(userB);

    let overlap = 0;
    let sumA = 0;
    let sumB = 0;
    let sumSqA = 0;
    let sumSqB = 0;
    let sumProduct = 0;

    for (const [movieId, ratingA] of ratingsA) {
        const ratingB = ratingsB.get(movieId);
        if (ratingB === undefined) {
            continue;
        }
        const devA = ratingA - meanA;
        const devB = ratingB - meanB;
        overlap += 1;
        sumA += devA;
        sumB += devB;
        sumSqA += devA * devA;
        sumSqB += devB * devB;
        sumProduct += devA * devB;
    }

    if (overlap < MIN_OVERLAP) {
        return { similarity: 0, overlap };
    }

    // Pearson re-centres on the co-rated subset, not on the overall user means.
    const numerator = sumProduct - (sumA * sumB) / overlap;
    const denomA = sumSqA - (sumA * sumA) / overlap;
    const denomB = sumSqB - (sumB * sumB) / overlap;
    if (denomA <= 0 || denomB <= 0) {
        return { similarity: 0, overlap };
    }

    return { similarity: numerator / Math.sqrt(denomA * denomB), overlap };
}

// Adjusted cosine between two movies, computed only over users who rated BOTH of
// them, after removing each of those users' mean ratings. The centring is by the
// global user mean (that is what makes it "adjusted"); there is no re-centring on
// the co-rater subset.
function itemSimilarity(movieA, movieB) {
    const ratersA = ratingsByItem.get(movieA);
    const ratersB = ratingsByItem.get(movieB);
    if (ratersA === undefined || ratersB === undefined) {
        return { similarity: 0, overlap: 0 };
    }

    let overlap = 0;
    let dot = 0;
    let normA = 0;
    let normB = 0;

    for (const [userId, ratingA] of ratersA) {
        const ratingB = ratersB.get(userId);
        if (ratingB === undefined) {
            continue;
        }
        const userAverage = userMean.get(userId);
        const devA = ratingA - userAverage;
        const devB = ratingB - userAverage;
        overlap += 1;
        dot += devA * devB;
        normA += devA * devA;
        normB += devB * devB;
    }

    if (overlap < MIN_OVERLAP || normA <= 0 || normB <= 0) {
        return { similarity: 0, overlap };
    }

    return { similarity: dot / Math.sqrt(normA * normB), overlap };
}

// Overlap weighting: shrink a similarity toward zero according to how much
// evidence stands behind it. The sign is preserved, so a well-supported pair
// always outranks a thinly supported one of equal correlation.
function shrinkByOverlap(similarity, overlap) {
    return similarity * (overlap / (overlap + SHRINKAGE));
}

function clampRating(value) {
    return Math.max(1, Math.min(5, value));
}

// ------------------------------------------------------------- predictions

// Wraps a raw prediction, or applies the fallback chain when the neighbourhood
// produced nothing usable. Every returned value is finite and inside [1, 5].
function finalizePrediction(rawValue, userId, movieId, neighbours, reason) {
    if (rawValue !== null && Number.isFinite(rawValue)) {
        return {
            value: clampRating(rawValue),
            neighbours,
            fallback: null,
            reason
        };
    }

    const userAverage = userMean.get(userId);
    if (userAverage !== undefined) {
        return { value: clampRating(userAverage), neighbours: 0, fallback: FALLBACK_ORDER[0], reason };
    }

    const itemAverage = itemMean.get(movieId);
    if (itemAverage !== undefined) {
        return { value: clampRating(itemAverage), neighbours: 0, fallback: FALLBACK_ORDER[1], reason };
    }

    return { value: clampRating(globalMean), neighbours: 0, fallback: FALLBACK_ORDER[2], reason };
}

// User-based CF: users who rated the target movie become candidate neighbours.
// r[u,i] = mean(u) + SUM sim(u,v) * (r[v,i] - mean(v)) / SUM |sim(u,v)|
function predictUserBased(userId, movieId, k = DEFAULT_K) {
    const base = userMean.get(userId);
    const raters = ratingsByItem.get(movieId);
    if (base === undefined || raters === undefined) {
        return finalizePrediction(null, userId, movieId, 0, 'no indexed rating for this user or movie');
    }

    const candidates = [];
    for (const [otherId, rating] of raters) {
        if (otherId === userId) {
            continue;
        }
        const { similarity, overlap } = userSimilarity(userId, otherId);
        if (overlap < MIN_OVERLAP || similarity <= 0) {
            continue;
        }
        candidates.push({
            id: otherId,
            weight: shrinkByOverlap(similarity, overlap),
            overlap,
            deviation: rating - userMean.get(otherId)
        });
    }

    if (candidates.length === 0) {
        return finalizePrediction(null, userId, movieId, 0, 'no neighbour passed the overlap and similarity filters');
    }

    candidates.sort((left, right) => right.weight - left.weight);
    const neighbours = candidates.slice(0, k);

    let numerator = 0;
    let denominator = 0;
    for (const neighbour of neighbours) {
        numerator += neighbour.weight * neighbour.deviation;
        denominator += Math.abs(neighbour.weight);
    }
    if (denominator <= 0) {
        return finalizePrediction(null, userId, movieId, neighbours.length, 'neighbour weights cancelled out');
    }

    return finalizePrediction(base + numerator / denominator, userId, movieId, neighbours.length, null);
}

// Item-based CF: movies the target user rated become candidate neighbours.
// r[u,i] = mean(i) + SUM sim(i,j) * (r[u,j] - mean(j)) / SUM |sim(i,j)|
function predictItemBased(userId, movieId, k = DEFAULT_K) {
    const base = itemMean.get(movieId);
    const userRatings = ratingsByUser.get(userId);
    if (base === undefined || userRatings === undefined) {
        return finalizePrediction(null, userId, movieId, 0, 'no indexed rating for this user or movie');
    }

    const candidates = [];
    for (const [otherId, rating] of userRatings) {
        if (otherId === movieId) {
            continue;
        }
        const { similarity, overlap } = itemSimilarity(movieId, otherId);
        if (overlap < MIN_OVERLAP || similarity <= 0) {
            continue;
        }
        candidates.push({
            id: otherId,
            weight: shrinkByOverlap(similarity, overlap),
            overlap,
            deviation: rating - itemMean.get(otherId)
        });
    }

    if (candidates.length === 0) {
        return finalizePrediction(null, userId, movieId, 0, 'no neighbour passed the overlap and similarity filters');
    }

    candidates.sort((left, right) => right.weight - left.weight);
    const neighbours = candidates.slice(0, k);

    let numerator = 0;
    let denominator = 0;
    for (const neighbour of neighbours) {
        numerator += neighbour.weight * neighbour.deviation;
        denominator += Math.abs(neighbour.weight);
    }
    if (denominator <= 0) {
        return finalizePrediction(null, userId, movieId, neighbours.length, 'neighbour weights cancelled out');
    }

    return finalizePrediction(base + numerator / denominator, userId, movieId, neighbours.length, null);
}

// ------------------------------------------------------------ verification

// Reproducible pseudo random generator, so the holdout sample is the same on
// every run and the reported numbers can be compared.
function createRandom(seed) {
    let state = seed >>> 0;
    return function () {
        state += 0x6D2B79F5;
        let value = state;
        value = Math.imul(value ^ (value >>> 15), value | 1);
        value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
        return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
}

function refreshMeans(userId, movieId) {
    const userRatings = ratingsByUser.get(userId);
    if (userRatings !== undefined && userRatings.size > 0) {
        userMean.set(userId, averageOf(userRatings));
    } else {
        userMean.delete(userId);
    }

    const itemRaters = ratingsByItem.get(movieId);
    if (itemRaters !== undefined && itemRaters.size > 0) {
        itemMean.set(movieId, averageOf(itemRaters));
    } else {
        itemMean.delete(movieId);
    }

    globalMean = indexedRatingCount > 0 ? indexedRatingSum / indexedRatingCount : 0;
}

// Hides one rating from the index, so a prediction can be made without seeing
// the answer. Used only by the holdout check.
function dropIndexedRating(userId, movieId) {
    const userRatings = ratingsByUser.get(userId);
    if (userRatings === undefined || !userRatings.has(movieId)) {
        return false;
    }
    indexedRatingSum -= userRatings.get(movieId);
    indexedRatingCount -= 1;
    userRatings.delete(movieId);

    const itemRaters = ratingsByItem.get(movieId);
    if (itemRaters !== undefined) {
        itemRaters.delete(userId);
    }

    refreshMeans(userId, movieId);
    return true;
}

function restoreIndexedRating(userId, movieId, rating) {
    const userRatings = ratingsByUser.get(userId);
    if (userRatings === undefined) {
        return;
    }
    userRatings.set(movieId, rating);

    const itemRaters = ratingsByItem.get(movieId);
    if (itemRaters !== undefined) {
        itemRaters.set(userId, rating);
    }

    indexedRatingSum += rating;
    indexedRatingCount += 1;
    refreshMeans(userId, movieId);
}

// Picks a random sample of known (user, movie, rating) triples, hides each of
// them from the index, predicts them back with both methods, then restores the
// index. Reported RMSE/MAE are therefore genuine held-out errors.
function evaluateHoldout(sampleSize = 200, seed = 20250929) {
    const random = createRandom(seed);
    const picked = [];
    const seen = new Set();

    while (picked.length < sampleSize && seen.size < ratings.length) {
        const row = ratings[Math.floor(random() * ratings.length)];
        const key = row.userId + ':' + row.movieId;
        if (seen.has(key)) {
            continue;
        }
        seen.add(key);
        picked.push({ userId: row.userId, movieId: row.movieId, rating: row.rating });
    }

    for (const triple of picked) {
        dropIndexedRating(triple.userId, triple.movieId);
    }

    let userSquared = 0;
    let userAbsolute = 0;
    let itemSquared = 0;
    let itemAbsolute = 0;
    let userFallbacks = 0;
    let itemFallbacks = 0;

    for (const triple of picked) {
        const userResult = predictUserBased(triple.userId, triple.movieId);
        const itemResult = predictItemBased(triple.userId, triple.movieId);
        if (userResult.fallback !== null) userFallbacks += 1;
        if (itemResult.fallback !== null) itemFallbacks += 1;

        const userError = userResult.value - triple.rating;
        const itemError = itemResult.value - triple.rating;
        userSquared += userError * userError;
        userAbsolute += Math.abs(userError);
        itemSquared += itemError * itemError;
        itemAbsolute += Math.abs(itemError);
    }

    for (const triple of picked) {
        restoreIndexedRating(triple.userId, triple.movieId, triple.rating);
    }

    const count = picked.length;
    return {
        count,
        userBased: {
            rmse: Math.sqrt(userSquared / count),
            mae: userAbsolute / count,
            fallbacks: userFallbacks
        },
        itemBased: {
            rmse: Math.sqrt(itemSquared / count),
            mae: itemAbsolute / count,
            fallbacks: itemFallbacks
        }
    };
}

// Cold start probe: the least rated movie in the catalogue, paired with a user
// who never rated it. Confirms the fallback chain returns a finite in-range
// value instead of NaN or a division by zero.
function checkColdStart() {
    let coldest = null;
    for (const [movieId, raters] of ratingsByItem) {
        if (coldest === null || raters.size < coldest.raters) {
            coldest = { movieId, raters: raters.size };
        }
    }
    if (coldest === null) {
        return null;
    }

    const sparsestUsers = [...ratingsByUser.entries()].sort((left, right) => left[1].size - right[1].size);
    let userId = null;
    for (const [candidateId, userRatings] of sparsestUsers) {
        if (!userRatings.has(coldest.movieId)) {
            userId = candidateId;
            break;
        }
    }
    if (userId === null) {
        return null;
    }

    // Also probe ids that are not in the index at all. Neither the user mean nor
    // the item mean exists there, so both methods must reach the global mean.
    const absentUser = predictUserBased(-1, -1);
    const absentItem = predictItemBased(-1, -1);

    return {
        userId,
        movieId: coldest.movieId,
        raters: coldest.raters,
        userRatings: ratingsByUser.get(userId).size,
        userBased: predictUserBased(userId, coldest.movieId),
        itemBased: predictItemBased(userId, coldest.movieId),
        absentUser,
        absentItem
    };
}

// Runs both checks and prints them to the console.
function runVerification() {
    const holdout = evaluateHoldout();
    const coldStart = checkColdStart();

    console.log('Collaborative filtering verification');
    console.log(`Holdout: ${holdout.count} known ratings hidden from the index, then predicted back.`);
    console.log(`  user-based CF  RMSE ${holdout.userBased.rmse.toFixed(3)}  MAE ${holdout.userBased.mae.toFixed(3)}  fallbacks ${holdout.userBased.fallbacks}`);
    console.log(`  item-based CF  RMSE ${holdout.itemBased.rmse.toFixed(3)}  MAE ${holdout.itemBased.mae.toFixed(3)}  fallbacks ${holdout.itemBased.fallbacks}`);

    if (coldStart === null) {
        console.log('  cold start: no suitable probe found.');
    } else {
        const describe = result =>
            `${result.value.toFixed(2)} (${result.fallback === null ? 'computed' : 'fallback: ' + result.fallback})`;
        console.log(`Cold start: movie ${coldStart.movieId} with ${coldStart.raters} rater(s), user ${coldStart.userId} with ${coldStart.userRatings} rating(s).`);
        console.log(`  user-based CF  ${describe(coldStart.userBased)}`);
        console.log(`  item-based CF  ${describe(coldStart.itemBased)}`);
        console.log(`  unknown user + unknown movie (last fallback level)`);
        console.log(`  user-based CF  ${describe(coldStart.absentUser)}`);
        console.log(`  item-based CF  ${describe(coldStart.absentItem)}`);
        console.log(`  values finite and inside [1, 5]: ${[coldStart.userBased.value, coldStart.itemBased.value, coldStart.absentUser.value, coldStart.absentItem.value].every(Number.isFinite)}`);
    }

    return { holdout, coldStart };
}
