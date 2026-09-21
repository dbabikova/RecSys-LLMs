// Global variables for storing movie and recommendation data
let movies = [];
let recommendations = new Map(); // movieId -> array of 5 recommended movie ids

// Primary function to load data from the parent week2/ directory
async function loadData() {
    try {
        // Load and parse movie data
        const moviesResponse = await fetch('../u.item');
        if (!moviesResponse.ok) {
            throw new Error(`Failed to load movie data: ${moviesResponse.status}`);
        }
        const moviesText = await moviesResponse.text();
        parseItemData(moviesText);

        // Load and parse variant 2 recommendation data (u.recommend_2)
        const recommendResponse = await fetch('../u.recommend_2');
        if (!recommendResponse.ok) {
            throw new Error(`Failed to load recommendation data: ${recommendResponse.status}`);
        }
        const recommendText = await recommendResponse.text();
        parseRecommendData(recommendText);
    } catch (error) {
        console.error('Error loading data:', error);
        const resultElement = document.getElementById('result');
        if (resultElement) {
            resultElement.textContent = `Error: ${error.message}. Please make sure ../u.item and ../u.recommend_2 files are in the correct location.`;
            resultElement.className = 'error';
        }
        throw error; // Re-throw to allow script.js to handle the error
    }
}

// Parse movie data from u.item format (only id and title are needed)
function parseItemData(text) {
    const lines = text.split('\n');

    for (const line of lines) {
        if (line.trim() === '') continue;

        const fields = line.split('|');
        if (fields.length < 2) continue; // Skip invalid lines

        const id = parseInt(fields[0]);
        const title = fields[1];

        movies.push({ id, title });
    }
}

// Parse recommendation data from u.recommend_2 format (movieId|rec1|rec2|rec3|rec4|rec5)
function parseRecommendData(text) {
    const lines = text.split('\n');

    for (const line of lines) {
        if (line.trim() === '') continue;

        const fields = line.split('|');
        if (fields.length < 6) continue; // Skip invalid lines

        const movieId = parseInt(fields[0]);
        const recIds = fields.slice(1, 6).map(id => parseInt(id));

        recommendations.set(movieId, recIds);
    }
}