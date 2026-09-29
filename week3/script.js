// script.js - user interface for the memory-based collaborative filtering demo
// The algorithms live in cf.js, the parsing in data.js.

window.onload = function () {
    loadData()
        .then(function () {
            setProgress(35);
            const indexSummary = buildCfIndexes(ratings);
            setProgress(80);
            populateUserDropdown();
            populateMovieDropdown();
            setProgress(100);
            document.getElementById('predict-btn').disabled = false;
            renderStats(indexSummary);
            resetResultCards();

            const verification = runVerification();
            updateStatus(
                'Ready: ' + indexSummary.users + ' users and ' + indexSummary.ratedMovies +
                ' rated movies indexed. Holdout check: user-based RMSE ' +
                verification.holdout.userBased.rmse.toFixed(3) +
                ', item-based RMSE ' + verification.holdout.itemBased.rmse.toFixed(3) +
                ' (details in the console).'
            );
        })
        .catch(function (error) {
            showError(error.message);
        });
};

function populateUserDropdown() {
    const select = document.getElementById('user-select');
    const fragment = document.createDocumentFragment();

    for (const userId of [...ratingsByUser.keys()].sort((left, right) => left - right)) {
        const option = document.createElement('option');
        option.value = userId;
        option.textContent = 'User ' + userId + ' (' + ratingsByUser.get(userId).size + ' ratings)';
        fragment.appendChild(option);
    }

    select.replaceChildren(fragment);
    select.disabled = false;
}

function populateMovieDropdown() {
    const select = document.getElementById('movie-select');
    const fragment = document.createDocumentFragment();

    for (const movieId of [...ratingsByItem.keys()].sort((left, right) => left - right)) {
        const movie = movies.find(entry => entry.id === movieId);
        const option = document.createElement('option');
        option.value = movieId;
        option.textContent = (movie ? movie.title : 'Movie ' + movieId) +
            ' (' + ratingsByItem.get(movieId).size + ' ratings)';
        fragment.appendChild(option);
    }

    select.replaceChildren(fragment);
    select.disabled = false;
}

function predictRating() {
    const userId = parseInt(document.getElementById('user-select').value, 10);
    const movieId = parseInt(document.getElementById('movie-select').value, 10);

    if (Number.isNaN(userId) || Number.isNaN(movieId)) {
        updateStatus('Choose a user and a movie first.');
        return;
    }

    const movie = movies.find(entry => entry.id === movieId);
    const title = movie ? movie.title + (movie.year ? ' (' + movie.year + ')' : '') : 'Movie ' + movieId;
    document.getElementById('prediction-context').textContent = 'User ' + userId + ' and ' + title;

    renderResultCard('user', predictUserBased(userId, movieId));
    renderResultCard('item', predictItemBased(userId, movieId));

    updateStatus('Prediction ready for user ' + userId + ' and ' + title + '.');
}

function renderResultCard(prefix, result) {
    document.getElementById(prefix + '-headline').textContent =
        (prefix === 'user' ? 'User-Based CF' : 'Item-Based CF') + ': ' + result.value.toFixed(2) + ' predicted';
    document.getElementById(prefix + '-score').textContent = result.value.toFixed(2);
    document.getElementById(prefix + '-verdict').textContent = verdictFor(result.value);

    const meter = document.getElementById(prefix + '-meter');
    meter.style.width = ((result.value - 1) / 4 * 100) + '%';
    meter.className = 'meter-fill ' + bandFor(result.value);

    const similarity = prefix === 'user' ? SIMILARITY_LABEL.userBased : SIMILARITY_LABEL.itemBased;
    document.getElementById(prefix + '-neighbours').textContent = result.neighbours + ' of top ' + DEFAULT_K;

    const pathCell = document.getElementById(prefix + '-path');
    if (result.fallback === null) {
        pathCell.textContent = 'computed from the neighbourhood';
        pathCell.className = 'ok';
    } else {
        pathCell.textContent = 'cold start fallback: ' + result.fallback;
        pathCell.className = 'warn';
    }

    const detail = document.getElementById(prefix + '-metric');
    detail.textContent = similarity;
}

function resetResultCards() {
    for (const prefix of ['user', 'item']) {
        const label = prefix === 'user' ? 'User-Based CF' : 'Item-Based CF';
        document.getElementById(prefix + '-headline').textContent = label + ': -- predicted';
        document.getElementById(prefix + '-score').textContent = '--';
        document.getElementById(prefix + '-verdict').textContent = 'Pick a user and a movie';
        const meter = document.getElementById(prefix + '-meter');
        meter.style.width = '0%';
        meter.className = 'meter-fill';
        document.getElementById(prefix + '-neighbours').textContent = '-';
        document.getElementById(prefix + '-path').textContent = '-';
        document.getElementById(prefix + '-metric').textContent = prefix === 'user'
            ? SIMILARITY_LABEL.userBased
            : SIMILARITY_LABEL.itemBased;
    }
    document.getElementById('prediction-context').textContent = 'No user and movie selected yet';
}

function renderStats(indexSummary) {
    const chips = [
        indexSummary.users + ' users',
        movies.length + ' movies in the catalogue',
        indexSummary.ratedMovies + ' movies with ratings',
        ratings.length.toLocaleString('en-US') + ' ratings',
        'sparsity ' + sparsityPercent().toFixed(1) + '%',
        'user metric: Pearson correlation',
        'item metric: adjusted cosine',
        'K = ' + DEFAULT_K + ' neighbours',
        'MIN_OVERLAP = ' + MIN_OVERLAP
    ];

    document.getElementById('stats').innerHTML = chips
        .map(function (chip) { return '<span class="chip">' + chip + '</span>'; })
        .join('');
}

function bandFor(value) {
    if (value >= 4) return 'high';
    if (value >= 3) return 'medium';
    return 'low';
}

function verdictFor(value) {
    if (value >= 4) return 'Would probably love it';
    if (value >= 3) return 'Would probably like it';
    return 'Would probably not enjoy it';
}

function updateStatus(message) {
    const element = document.getElementById('status');
    element.className = 'status';
    element.textContent = message;
}

function showError(message) {
    const element = document.getElementById('status');
    element.className = 'status error';
    element.textContent = message;
    // The dataset never arrived, so there is nothing to choose from. The
    // dropdowns are re-disabled here so the failure state is explicit even if
    // an error were raised after they had already been populated.
    document.getElementById('user-select').disabled = true;
    document.getElementById('movie-select').disabled = true;
    document.getElementById('predict-btn').disabled = true;
}

function setProgress(percent) {
    document.getElementById('progress-bar').style.width = percent + '%';
}
