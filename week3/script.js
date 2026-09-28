// script.js - Matrix Factorization model (TensorFlow.js): build, train, predict

const LATENT_DIM = 16;   // length of the learned user/movie vectors
const EPOCHS = 15;
const BATCH_SIZE = 64;
const MEAN_RATING = 3.5; // mean of the 1-5 MovieLens scale, used to prime the bias

let model;
let isTraining = false;
let validationRmse = null;
let backend = '';

window.onload = async function () {
    if (typeof tf === 'undefined') {
        showError('TensorFlow.js did not load. Check the network connection to the jsDelivr CDN.');
        return;
    }

    try {
        backend = await selectBackend();
        updateStatus('Loading MovieLens 100K data...');

        await loadData();

        populateUserDropdown();
        populateMovieDropdown();

        updateStatus('Data loaded. Training the matrix factorization model...');
        await trainModel();
    } catch (error) {
        console.error('Initialization error:', error);
        showError(error.message);
    }
};

// WebGL runs the training on the GPU and is several times faster than the CPU
// backend; silently keep whatever backend is available if it is not.
async function selectBackend() {
    if (tf.env().getBool('HAS_WEBGL')) {
        try {
            await tf.setBackend('webgl');
        } catch (error) {
            console.warn('WebGL backend unavailable, keeping the default one.', error);
        }
    }
    return tf.getBackend();
}

// Observed rating for a pair, or undefined when the pair is missing.
// A linear scan beats building a 100k-entry Map on start-up: it runs once per click.
function observedRating(userId, movieId) {
    const row = ratings.find(r => r.userId === userId && r.movieId === movieId);
    return row ? row.rating : undefined;
}

function populateUserDropdown() {
    const userSelect = document.getElementById('user-select');
    const counts = new Map();

    for (const row of ratings) {
        counts.set(row.userId, (counts.get(row.userId) || 0) + 1);
    }

    const ids = Array.from(counts.keys()).sort((a, b) => a - b);
    userSelect.innerHTML = '';
    for (const id of ids) {
        userSelect.appendChild(new Option(`User ${id} (${counts.get(id)} ratings)`, id));
    }
    userSelect.disabled = false;
}

function populateMovieDropdown() {
    const movieSelect = document.getElementById('movie-select');
    const ratedMovies = new Set(ratings.map(row => row.movieId));

    movieSelect.innerHTML = '';
    for (const movie of movies) {
        const label = movie.year ? `${movie.title} (${movie.year})` : movie.title;
        const suffix = ratedMovies.has(movie.id) ? '' : ' - unrated';
        movieSelect.appendChild(new Option(label + suffix, movie.id));
    }
    movieSelect.disabled = false;
}

// r_hat(user, movie) = <user vector, movie vector> + userBias + movieBias
function createModel(numUsers, numMovies, latentDim = LATENT_DIM) {
    const userInput = tf.input({ shape: [1], dtype: 'int32', name: 'userInput' });
    const movieInput = tf.input({ shape: [1], dtype: 'int32', name: 'movieInput' });

    // Latent factors. inputDim is maxId + 1 because id 0 is the padding row.
    const userVector = tf.layers.flatten({
        name: 'userVector'
    }).apply(tf.layers.embedding({
        inputDim: numUsers + 1,
        outputDim: latentDim,
        name: 'userEmbedding'
    }).apply(userInput));

    const movieVector = tf.layers.flatten({
        name: 'movieVector'
    }).apply(tf.layers.embedding({
        inputDim: numMovies + 1,
        outputDim: latentDim,
        name: 'movieEmbedding'
    }).apply(movieInput));

    // Bias terms. embeddingsInitializer (not initializer) is the option an
    // Embedding layer reads; the user bias starts at the global mean so the
    // model opens at "everyone rates 3.5" and only learns the deviations.
    // Dropping the two tf.layers.add() calls below leaves plain MF.
    const userBias = tf.layers.flatten().apply(tf.layers.embedding({
        inputDim: numUsers + 1,
        outputDim: 1,
        embeddingsInitializer: tf.initializers.constant({ value: MEAN_RATING }),
        name: 'userBias'
    }).apply(userInput));

    const movieBias = tf.layers.flatten().apply(tf.layers.embedding({
        inputDim: numMovies + 1,
        outputDim: 1,
        embeddingsInitializer: tf.initializers.zeros(),
        name: 'movieBias'
    }).apply(movieInput));

    const dotProduct = tf.layers.dot({ axes: 1, name: 'dotProduct' })
        .apply([userVector, movieVector]);

    const withBias = tf.layers.add({ name: 'withBias' }).apply([dotProduct, userBias]);
    const prediction = tf.layers.add({ name: 'prediction' }).apply([withBias, movieBias]);

    return tf.model({ inputs: [userInput, movieInput], outputs: prediction });
}

async function trainModel() {
    isTraining = true;
    setPredictEnabled(false);

    try {
        model = createModel(numUsers, numMovies, LATENT_DIM);

        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError'
        });

        const userIds = ratings.map(row => row.userId);
        const movieIds = ratings.map(row => row.movieId);
        const values = ratings.map(row => row.rating);

        const userTensor = tf.tensor2d(userIds, [userIds.length, 1], 'int32');
        const movieTensor = tf.tensor2d(movieIds, [movieIds.length, 1], 'int32');
        const ratingTensor = tf.tensor2d(values, [values.length, 1], 'float32');

        const history = await model.fit([userTensor, movieTensor], ratingTensor, {
            epochs: EPOCHS,
            batchSize: BATCH_SIZE,
            validationSplit: 0.1,
            shuffle: true,
            verbose: 0, // the default 1 prints a line per batch: ~23k console writes
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    const progress = Math.round(((epoch + 1) / EPOCHS) * 100);
                    setProgress(progress);
                    const val = logs.val_loss === undefined ? '' : `, val loss ${logs.val_loss.toFixed(4)}`;
                    updateStatus(`Epoch ${epoch + 1}/${EPOCHS} - loss ${logs.loss.toFixed(4)}${val}`);
                }
            }
        });

        tf.dispose([userTensor, movieTensor, ratingTensor]);

        const finalLoss = history.history.val_loss
            ? history.history.val_loss[history.history.val_loss.length - 1]
            : history.history.loss[history.history.loss.length - 1];
        validationRmse = Math.sqrt(finalLoss);

        setProgress(100);
        updateStatus('Model ready - select a user and a movie to get a predicted rating.');
        setPredictEnabled(true);
        isTraining = false;

        renderStats();
    } catch (error) {
        console.error('Training error:', error);
        isTraining = false;
        showError('Training failed: ' + error.message);
    }
}

async function predictRating() {
    if (isTraining || !model) {
        renderResult('The model is still training. Please wait...', 'medium');
        return;
    }

    const userId = parseInt(document.getElementById('user-select').value, 10);
    const movieId = parseInt(document.getElementById('movie-select').value, 10);

    if (!userId || !movieId) {
        renderResult('Please select both a user and a movie.', 'medium');
        return;
    }

    const userTensor = tf.tensor2d([[userId]], [1, 1], 'int32');
    const movieTensor = tf.tensor2d([[movieId]], [1, 1], 'int32');

    try {
        const output = model.predict([userTensor, movieTensor]);
        const predicted = Math.min(5, Math.max(1, (await output.data())[0]));
        tf.dispose([userTensor, movieTensor, output]);

        const movie = movies.find(m => m.id === movieId);
        const movieTitle = movie
            ? (movie.year ? `${movie.title} (${movie.year})` : movie.title)
            : `movie ${movieId}`;

        const observed = observedRating(userId, movieId);
        const observedLine = observed === undefined
            ? '<div class="meta">This user has not rated this movie - the rating is a genuine estimate.</div>'
            : '<div class="meta">Observed rating in the dataset: <strong>' + observed.toFixed(2) +
              '</strong> (error ' + Math.abs(predicted - observed).toFixed(2) + ')</div>';

        const tone = predicted >= 4 ? 'high' : (predicted >= 3 ? 'medium' : 'low');

        renderResult(
            '<div class="result-headline">User ' + userId + ' would rate</div>' +
            '<div class="result-title">' + escapeHtml(movieTitle) + '</div>' +
            '<div class="score">' + predicted.toFixed(2) + ' <span>/ 5</span></div>' +
            '<div class="meter"><div class="meter-fill ' + tone + '" style="width:' +
                (predicted / 5 * 100).toFixed(1) + '%"></div></div>' +
            '<div class="verdict">' + verdictFor(predicted) + '</div>' +
            observedLine,
            tone
        );
    } catch (error) {
        console.error('Prediction error:', error);
        tf.dispose([userTensor, movieTensor]);
        renderResult('Prediction failed: ' + error.message, 'low');
    }
}

function verdictFor(value) {
    if (value >= 4.5) return 'Loved it';
    if (value >= 3.5) return 'Really liked it';
    if (value >= 2.5) return 'It was okay';
    if (value >= 1.5) return 'Did not like it';
    return 'Hated it';
}

function renderStats() {
    const parts = [
        ratings.length.toLocaleString('en-US') + ' ratings',
        numUsers + ' users',
        numMovies.toLocaleString('en-US') + ' movies',
        LATENT_DIM + ' latent factors',
        'validation RMSE ' + validationRmse.toFixed(3),
        (backend || tf.getBackend()) + ' backend'
    ];
    document.getElementById('stats').innerHTML =
        parts.map(text => '<span class="chip">' + text + '</span>').join('');
}

// UI helpers
function setPredictEnabled(enabled) {
    document.getElementById('predict-btn').disabled = !enabled;
}

function setProgress(percent) {
    document.getElementById('progress-bar').style.width = percent + '%';
}

function updateStatus(message, isError = false) {
    const status = document.getElementById('status');
    status.textContent = message;
    status.className = 'status' + (isError ? ' error' : '');
}

function showError(message) {
    updateStatus(message, true);
    setProgress(0);
}

function renderResult(html, tone = '') {
    const result = document.getElementById('result');
    result.innerHTML = html;
    result.className = 'result' + (tone ? ' ' + tone : '');
}

function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, ch => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[ch]));
}
