// Initialize the application when the window loads
window.onload = async function() {
    try {
        // Display loading message
        const resultElement = document.getElementById('result');
        resultElement.textContent = "Loading movie data...";
        resultElement.className = 'loading';
        
        // Load data
        await loadData();
        
        // Populate dropdown and update status
        populateMoviesDropdown();
        resultElement.textContent = "Data loaded. Please select a movie.";
        resultElement.className = 'success';
    } catch (error) {
        console.error('Initialization error:', error);
        // Error message already set in data.js
    }
};

// Populate the movies dropdown with sorted movie titles
function populateMoviesDropdown() {
    const selectElement = document.getElementById('movie-select');
    
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

// Main recommendation function
function getRecommendations() {
    const resultElement = document.getElementById('result');
    const selectElement = document.getElementById('movie-select');

    // Get user input
    const selectedMovieId = parseInt(selectElement.value);
    if (isNaN(selectedMovieId)) {
        resultElement.textContent = "Please select a movie first.";
        resultElement.className = 'error';
        return;
    }

    // Find the liked movie
    const likedMovie = movies.find(movie => movie.id === selectedMovieId);
    if (!likedMovie) {
        resultElement.textContent = "Error: Selected movie not found in database.";
        resultElement.className = 'error';
        return;
    }

    // Look up top-5 recommendations from the u.recommend dataset
    const recIds = recommendations.get(selectedMovieId);
    if (!recIds || recIds.length === 0) {
        resultElement.textContent = `No recommendations found for "${likedMovie.title}".`;
        resultElement.className = 'error';
        return;
    }

    // Resolve recommendation ids to titles
    const titleById = new Map(movies.map(movie => [movie.id, movie.title]));
    const recTitles = recIds.map(id => titleById.get(id)).filter(title => title);

    if (recTitles.length === 0) {
        resultElement.textContent = `No recommendations found for "${likedMovie.title}".`;
        resultElement.className = 'error';
        return;
    }

    resultElement.textContent = recTitles.join(', ');
    resultElement.className = 'success';
}
