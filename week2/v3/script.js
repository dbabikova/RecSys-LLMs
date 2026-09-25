// Initialize the application when the window loads
window.onload = async function() {
    try {
        // Display loading message
        const resultElement = document.getElementById('result');
        resultElement.textContent = "Loading data...";
        resultElement.className = 'loading';

        // Load data
        await loadData();

        // Populate dropdowns and update status
        populateMoviesDropdown('movie-select-1');
        populateMoviesDropdown('movie-select-2');
        populateMoviesDropdown('movie-select-3');
        resultElement.textContent = "Data loaded. Please select three movies.";
        resultElement.className = 'success';
    } catch (error) {
        console.error('Initialization error:', error);
        // Error message already set in data.js
    }
};

// Populate a movies dropdown with sorted movie titles
function populateMoviesDropdown(selectId) {
    const selectElement = document.getElementById(selectId);

    // Clear existing options except the first placeholder
    while (selectElement.options.length > 1) {
        selectElement.remove(1);
    }

    // Sort movies alphabetically by title
    const sortedMovies = [...movies].sort((a, b) => a.title.localeCompare(b.title));

    // Add movies to dropdown
    sortedMovies.forEach(movie => {
        const option = document.createElement('option');
        option.value = movie.id;
        option.textContent = movie.title;
        selectElement.appendChild(option);
    });
}

// Cosine similarity between two vectors
function cosine(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let k = 0; k < a.length; k++) {
        dot += a[k] * b[k];
        na += a[k] * a[k];
        nb += b[k] * b[k];
    }
    if (na <= 0 || nb <= 0) return 0;
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// Main recommendation function: profile = average of 3 genre vectors
function getRecommendations() {
    const resultElement = document.getElementById('result');

    const selectedIds = [
        document.getElementById('movie-select-1').value,
        document.getElementById('movie-select-2').value,
        document.getElementById('movie-select-3').value
    ].map(value => parseInt(value));

    if (selectedIds.some(id => isNaN(id))) {
        resultElement.textContent = "Please select three different movies first.";
        resultElement.className = 'error';
        return;
    }

    const uniqueIds = [...new Set(selectedIds)];
    if (uniqueIds.length !== 3) {
        resultElement.textContent = "Please select three different movies.";
        resultElement.className = 'error';
        return;
    }

    // Build profile as the average of the 3 genre vectors
    const N = 18;
    const profile = new Float64Array(N);
    for (const id of selectedIds) {
        const v = vectors.get(id);
        if (!v) {
            resultElement.textContent = "Error: no vector data for one of the selected movies.";
            resultElement.className = 'error';
            return;
        }
        for (let k = 0; k < N; k++) profile[k] += v[k] / 3;
    }

    // Rank all movies by cosine with the profile, excluding the watched ones
    const excluded = new Set(selectedIds);
    const scored = [];
    movies.forEach(movie => {
        if (excluded.has(movie.id)) return;
        const v = vectors.get(movie.id);
        if (!v) return;
        scored.push({ id: movie.id, sim: cosine(profile, v) });
    });

    scored.sort((a, b) => (b.sim - a.sim) || (a.id - b.id));
    const top5 = scored.slice(0, 5);

    if (top5.length === 0) {
        resultElement.textContent = "No recommendations found for the selected movies.";
        resultElement.className = 'error';
        return;
    }

    const titleById = new Map(movies.map(movie => [movie.id, movie.title]));
    const recTitles = top5.map(rec => titleById.get(rec.id));
    resultElement.textContent = recTitles.join(', ');
    resultElement.className = 'success';
}