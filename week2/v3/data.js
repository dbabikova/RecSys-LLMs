// Global variables for storing movie data and genre vectors
let movies = [];
let vectors = new Map(); // movieId -> Float64Array(18) from u.item_vector

// Primary function to load data from the parent week2/ directory
async function loadData() {
    try {
        // Load and parse movie data (u.item is latin-1 encoded)
        const moviesResponse = await fetch('../u.item');
        if (!moviesResponse.ok) {
            throw new Error(`Failed to load movie data: ${moviesResponse.status}`);
        }
        const moviesBuffer = await moviesResponse.arrayBuffer();
        const moviesText = new TextDecoder('iso-8859-1').decode(moviesBuffer);
        parseItemData(moviesText);

        // Load genre vectors (id|title|18 values)
        const vectorResponse = await fetch('../u.item_vector');
        if (!vectorResponse.ok) {
            throw new Error(`Failed to load vector data: ${vectorResponse.status}`);
        }
        const vectorText = await vectorResponse.text();
        parseVectorData(vectorText);
    } catch (error) {
        console.error('Error loading data:', error);
        const resultElement = document.getElementById('result');
        if (resultElement) {
            resultElement.textContent = `Error: ${error.message}. Please make sure ../u.item and ../u.item_vector files are in the correct location.`;
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

// Parse vector data from u.item_vector format (itemId|title|18 values)
function parseVectorData(text) {
    const lines = text.split('\n');

    for (const line of lines) {
        if (line.trim() === '') continue;

        const fields = line.split('|');
        if (fields.length < 20) continue; // id | title | 18 values

        const id = parseInt(fields[0]);
        const values = fields.slice(2, 20).map(value => parseFloat(value));
        vectors.set(id, Float64Array.from(values));
    }
}