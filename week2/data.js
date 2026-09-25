// Global variables for storing movie, rating and recommendation data
let movies = [];
let ratings = [];
let recommendations = new Map(); // movieId -> array of 5 recommended movie ids

// Genre names as defined in the u.item file (19 fields, unknown flag first)
const genreNames = [
    "Unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
];

// Primary function to load data from files
async function loadData() {
    try {
        // Load and parse movie data
        const moviesResponse = await fetch('u.item');
        if (!moviesResponse.ok) {
            throw new Error(`Failed to load movie data: ${moviesResponse.status}`);
        }
        // u.item is encoded in latin-1 (iso-8859-1); decode bytes explicitly
        const moviesBuffer = await moviesResponse.arrayBuffer();
        const moviesText = new TextDecoder('iso-8859-1').decode(moviesBuffer);
        parseItemData(moviesText);

        // Load and parse rating data
        const ratingsResponse = await fetch('u.data');
        if (!ratingsResponse.ok) {
            throw new Error(`Failed to load rating data: ${ratingsResponse.status}`);
        }
        const ratingsText = await ratingsResponse.text();
        parseRatingData(ratingsText);

        // Load and parse recommendation data
        const recommendResponse = await fetch('u.recommend');
        if (!recommendResponse.ok) {
            throw new Error(`Failed to load recommendation data: ${recommendResponse.status}`);
        }
        const recommendText = await recommendResponse.text();
        parseRecommendData(recommendText);
    } catch (error) {
        console.error('Error loading data:', error);
        const resultElement = document.getElementById('result');
        if (resultElement) {
            resultElement.textContent = `Error: ${error.message}. Please make sure u.item, u.data and u.recommend files are in the correct location.`;
            resultElement.className = 'error';
        }
        throw error; // Re-throw to allow script.js to handle the error
    }
}

// Parse movie data from u.item format
function parseItemData(text) {
    const lines = text.split('\n');
    
    for (const line of lines) {
        if (line.trim() === '') continue;
        
        const fields = line.split('|');
        if (fields.length < 5) continue; // Skip invalid lines
        
        const id = parseInt(fields[0]);
        const title = fields[1];
        
        // Extract genres (last 19 fields)
        const genreValues = fields.slice(5, 24).map(value => parseInt(value));
        const genres = genreNames.filter((_, index) => genreValues[index] === 1);
        
        movies.push({ id, title, genres });
    }
}

// Parse rating data from u.data format
function parseRatingData(text) {
    const lines = text.split('\n');
    
    for (const line of lines) {
        if (line.trim() === '') continue;
        
        const fields = line.split('\t');
        if (fields.length < 4) continue; // Skip invalid lines
        
        const userId = parseInt(fields[0]);
        const itemId = parseInt(fields[1]);
        const rating = parseFloat(fields[2]);
        const timestamp = parseInt(fields[3]);
        
        ratings.push({ userId, itemId, rating, timestamp });
    }
}

// Parse recommendation data from u.recommend format (movieId|rec1|rec2|rec3|rec4|rec5)
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
