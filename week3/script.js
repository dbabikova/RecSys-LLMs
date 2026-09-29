// script.js - user interface for the memory-based collaborative filtering demo
// The algorithms live in cf.js, the parsing in data.js.

// Full option lists, kept separately from the <select> so that typing in the
// search box can rebuild the list from scratch at any time.
const userChoices = [];
const movieChoices = [];

window.onload = function () {
    wireSearchBoxes();

    loadData()
        .then(function () {
            setProgress(35);
            const indexSummary = buildCfIndexes(ratings);
            setProgress(80);
            populateUserDropdown();
            populateMovieDropdown();
            setProgress(100);
            document.getElementById('predict-btn').disabled = false;
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

// The search box narrows the list below it on every keystroke, so the options
// always match exactly what has been typed. Matching is a case-insensitive
// substring test against the visible label, which lets "god" find Godfather
// and "96" find User 196.
function wireSearchBoxes() {
    const pairs = [
        { search: 'user-search', select: 'user-select' },
        { search: 'movie-search', select: 'movie-select' }
    ];

    for (const pair of pairs) {
        const box = document.getElementById(pair.search);
        box.addEventListener('input', function () {
            renderOptions(pair.select, choiceListFor(pair.select), this.value);
        });
    }
}

function choiceListFor(selectId) {
    return selectId === 'user-select' ? userChoices : movieChoices;
}

function populateUserDropdown() {
    userChoices.length = 0;

    for (const userId of [...ratingsByUser.keys()].sort((left, right) => left - right)) {
        userChoices.push({ value: String(userId), label: 'User ' + userId });
    }

    document.getElementById('user-search').disabled = false;
    renderOptions('user-select', userChoices, document.getElementById('user-search').value);
}

function populateMovieDropdown() {
    movieChoices.length = 0;

    for (const movieId of [...ratingsByItem.keys()].sort((left, right) => left - right)) {
        const movie = movies.find(entry => entry.id === movieId);
        movieChoices.push({
            value: String(movieId),
            label: movie ? movie.title : 'Movie ' + movieId
        });
    }

    document.getElementById('movie-search').disabled = false;
    renderOptions('movie-select', movieChoices, document.getElementById('movie-search').value);
}

// Rebuilds a <select> from the entries that match the query. A previously
// chosen entry is kept whenever the new query still contains it, so narrowing
// the list does not silently move the selection.
function renderOptions(selectId, entries, query) {
    const select = document.getElementById(selectId);
    const previous = select.value;
    const needle = query.trim().toLowerCase();
    const matches = needle === ''
        ? entries
        : entries.filter(entry => entry.label.toLowerCase().includes(needle));

    const fragment = document.createDocumentFragment();
    for (const entry of matches) {
        const option = document.createElement('option');
        option.value = entry.value;
        option.textContent = entry.label;
        fragment.appendChild(option);
    }

    if (matches.length === 0) {
        const empty = document.createElement('option');
        empty.value = '';
        empty.textContent = 'Nothing found';
        fragment.appendChild(empty);
    }

    select.replaceChildren(fragment);

    const stillThere = matches.some(entry => entry.value === previous);
    select.value = stillThere ? previous : '';
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
        document.getElementById(prefix + '-metric').textContent = prefix === 'user'
            ? SIMILARITY_LABEL.userBased
            : SIMILARITY_LABEL.itemBased;
    }
    document.getElementById('prediction-context').textContent = 'No user and movie selected yet';
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
    // search boxes and dropdowns are re-disabled here so the failure state is
    // explicit even if an error were raised after they had been populated.
    document.getElementById('user-search').disabled = true;
    document.getElementById('user-select').disabled = true;
    document.getElementById('movie-search').disabled = true;
    document.getElementById('movie-select').disabled = true;
    document.getElementById('predict-btn').disabled = true;
}

function setProgress(percent) {
    document.getElementById('progress-bar').style.width = percent + '%';
}
