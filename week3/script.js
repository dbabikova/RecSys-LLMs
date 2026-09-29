// script.js - user interface for the memory-based collaborative filtering demo
// The algorithms live in cf.js, the parsing in data.js.

let userCombobox = null;
let movieCombobox = null;

window.onload = function () {
    userCombobox = createCombobox('user');
    movieCombobox = createCombobox('movie');

    loadData()
        .then(function () {
            setProgress(35);
            const indexSummary = buildCfIndexes(ratings);
            setProgress(80);
            populateUserCombobox();
            populateMovieCombobox();
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

// ------------------------------------------------------------- combobox

// One text box per control: typing filters the list attached to that same box,
// and picking an entry writes its label back into it. The picked value is kept
// in `selected`, separate from the visible text, so the text can be edited
// without the stored value silently drifting away from what the user sees.
function createCombobox(prefix) {
    const input = document.getElementById(prefix + '-input');
    const list = document.getElementById(prefix + '-list');
    const box = input.parentElement;

    let choices = [];
    let selected = '';
    let activeIndex = -1;
    let isOpen = false;

    function labelFor(value) {
        for (const entry of choices) {
            if (entry.value === value) {
                return entry.label;
            }
        }
        return '';
    }

    function close() {
        isOpen = false;
        activeIndex = -1;
        list.replaceChildren();
        list.hidden = true;
        input.setAttribute('aria-expanded', 'false');
        input.removeAttribute('aria-activedescendant');
    }

    // Rebuilds the attached list from the current text. Matching is a
    // case-insensitive substring test, so "god" finds Godfather and "96"
    // finds User 196.
    function openWith(query) {
        const needle = query.trim().toLowerCase();
        const matches = needle === ''
            ? choices
            : choices.filter(entry => entry.label.toLowerCase().includes(needle));

        const fragment = document.createDocumentFragment();
        for (const entry of matches) {
            const item = document.createElement('li');
            item.id = prefix + '-option-' + entry.value;
            item.className = 'combobox-option';
            item.setAttribute('role', 'option');
            item.setAttribute('aria-selected', 'false');
            item.dataset.value = entry.value;
            item.textContent = entry.label;
            fragment.appendChild(item);
        }

        if (matches.length === 0) {
            const empty = document.createElement('li');
            empty.className = 'combobox-option combobox-empty';
            empty.setAttribute('role', 'option');
            empty.setAttribute('aria-disabled', 'true');
            empty.textContent = 'Nothing found';
            fragment.appendChild(empty);
        }

        list.replaceChildren(fragment);
        list.hidden = false;
        isOpen = true;
        activeIndex = -1;
        input.setAttribute('aria-expanded', 'true');
    }

    function selectableItems() {
        return list.querySelectorAll('.combobox-option[data-value]');
    }

    function highlight(index) {
        for (const item of list.querySelectorAll('.combobox-option')) {
            item.classList.remove('is-active');
            item.setAttribute('aria-selected', 'false');
        }

        const items = selectableItems();
        if (index < 0 || index >= items.length) {
            activeIndex = -1;
            input.removeAttribute('aria-activedescendant');
            return;
        }

        const item = items[index];
        activeIndex = index;
        item.classList.add('is-active');
        item.setAttribute('aria-selected', 'true');
        input.setAttribute('aria-activedescendant', item.id);
        item.scrollIntoView({ block: 'nearest' });
    }

    function choose(value) {
        selected = value;
        input.value = labelFor(value);
        close();
    }

    function move(delta) {
        if (!isOpen) {
            openWith(input.value);
        }

        const items = selectableItems();
        if (items.length === 0) {
            return;
        }

        let next;
        if (activeIndex < 0) {
            next = delta > 0 ? 0 : items.length - 1;
        } else {
            next = activeIndex + delta;
            if (next < 0) next = items.length - 1;
            if (next >= items.length) next = 0;
        }
        highlight(next);
    }

    // Editing the text drops the previous pick. The visible label and the
    // stored value must never disagree, otherwise predictRating() would use a
    // pair the user can no longer see.
    input.addEventListener('input', function () {
        selected = '';
        openWith(this.value);
    });

    input.addEventListener('focus', function () { openWith(this.value); });
    input.addEventListener('click', function () { openWith(this.value); });

    input.addEventListener('keydown', function (event) {
        if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
            event.preventDefault();
            move(event.key === 'ArrowDown' ? 1 : -1);
            return;
        }

        if (event.key === 'Enter') {
            if (isOpen && selectableItems().length > 0) {
                event.preventDefault();
                const items = selectableItems();
                choose(items[activeIndex < 0 ? 0 : activeIndex].dataset.value);
            }
            return;
        }

        if (event.key === 'Escape' && isOpen) {
            event.preventDefault();
            close();
        }
    });

    // Keeps focus on the text box so a click on an entry is not swallowed by a
    // blur that closes the list first.
    list.addEventListener('mousedown', function (event) { event.preventDefault(); });

    list.addEventListener('click', function (event) {
        const item = event.target.closest('.combobox-option');
        if (!item || item.dataset.value === undefined) {
            return;
        }
        choose(item.dataset.value);
    });

    document.addEventListener('mousedown', function (event) {
        if (isOpen && !box.contains(event.target)) {
            close();
        }
    });

    return {
        setChoices(next) { choices = next; },
        enable() { input.disabled = false; },
        disable() { input.disabled = true; close(); },
        value() { return selected; }
    };
}

// ------------------------------------------------------------- populating

function populateUserCombobox() {
    const choices = [];

    for (const userId of [...ratingsByUser.keys()].sort((left, right) => left - right)) {
        choices.push({ value: String(userId), label: 'User ' + userId });
    }

    userCombobox.setChoices(choices);
    userCombobox.enable();
}

function populateMovieCombobox() {
    const choices = [];

    for (const movieId of [...ratingsByItem.keys()].sort((left, right) => left - right)) {
        const movie = movies.find(entry => entry.id === movieId);
        choices.push({
            value: String(movieId),
            label: movie ? movie.title : 'Movie ' + movieId
        });
    }

    movieCombobox.setChoices(choices);
    movieCombobox.enable();
}

// ------------------------------------------------------------- prediction

function predictRating() {
    const userId = parseInt(userCombobox.value(), 10);
    const movieId = parseInt(movieCombobox.value(), 10);

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
    // The dataset never arrived, so there is nothing to choose from. Both
    // comboboxes are re-disabled here so the failure state is explicit even if
    // an error were raised after they had already been populated.
    userCombobox.disable();
    movieCombobox.disable();
    document.getElementById('predict-btn').disabled = true;
}

function setProgress(percent) {
    document.getElementById('progress-bar').style.width = percent + '%';
}
