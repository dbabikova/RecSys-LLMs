// data.js - loading and parsing of the MovieLens 100K dataset
// Files: u.item (movie catalogue, pipe separated) and u.data (ratings, tab separated)
// Structures:
//   movie  -> { id: number, title: string, year: number|null }
//   rating -> { userId: number, movieId: number, rating: number }

let movies = [];
let ratings = [];
let numUsers = 0;
let numMovies = 0;

const ITEM_URL = 'u.item';
const RATING_URL = 'u.data';

// u.item is ISO-8859-1 (Latin-1), not UTF-8. response.text() always decodes as
// UTF-8, so the nine titles that carry a diacritic come back as U+FFFD instead:
// "Mis?rables, Les", "Double vie de V?ronique, La", "N?nette et Boni",
// "Contempt (M?pris, Le)", "Metisse (Caf? au Lait)", "C?r?monie, La",
// "C'est arriv? pr?s de chez vous", "JLG/JLG - autoportrait de d?cembre" and
// "? k?ldum klaka (Cold Fever)". The question marks stand for U+FFFD; the file
// itself holds 0xC1, 0xE8, 0xE9 nine times and 0xF6. Decoding the bytes
// explicitly is the only fix, because response.text() takes no charset argument.
// The Encoding Standard maps the label "iso-8859-1" onto the windows-1252
// decoder, which agrees with Latin-1 on every byte this file contains.
const ITEM_ENCODING = 'iso-8859-1';

function decodeText(buffer, encoding) {
    return new TextDecoder(encoding).decode(buffer);
}

async function loadData() {
    try {
        const itemResponse = await fetch(ITEM_URL);
        if (!itemResponse.ok) {
            throw new Error(`${ITEM_URL} -> HTTP ${itemResponse.status}`);
        }
        movies = parseItemData(decodeText(await itemResponse.arrayBuffer(), ITEM_ENCODING));

        const ratingResponse = await fetch(RATING_URL);
        if (!ratingResponse.ok) {
            throw new Error(`${RATING_URL} -> HTTP ${ratingResponse.status}`);
        }
        // u.data is pure ASCII (user id, movie id, rating, timestamp), so the default
        // UTF-8 decode of response.text() is already exact and no decoder is needed.
        ratings = parseRatingData(await ratingResponse.text());

        // numUsers / numMovies report the LARGEST id seen, not a count of distinct
        // entities. Ids are 1-based and may contain gaps, so read them as the upper
        // bound of the id space. In this dataset the ids run 1..943 and 1..1682 with
        // no gaps, which is why they coincide with the counts. Nothing is sized from
        // these values: the collaborative filtering indexes are Maps keyed by id.
        let maxUserId = 0;
        let maxMovieId = 0;
        for (const movie of movies) {
            if (movie.id > maxMovieId) maxMovieId = movie.id;
        }
        for (const row of ratings) {
            if (row.userId > maxUserId) maxUserId = row.userId;
            if (row.movieId > maxMovieId) maxMovieId = row.movieId;
        }
        numUsers = maxUserId;
        numMovies = maxMovieId;

        console.log(`Loaded ${ratings.length} ratings, ${numUsers} users, ${numMovies} movies`);

        return { movies, ratings, numUsers, numMovies };
    } catch (error) {
        if (error instanceof TypeError) {
            throw new Error(
                'The dataset could not be read. Serve the folder over HTTP ' +
                '(for example: python -m http.server) - browsers block fetch() on file:// pages.'
            );
        }
        console.error('Error loading data:', error);
        throw error;
    }
}

function parseItemData(text) {
    const movieData = [];

    for (const line of text.split(/\r?\n/)) {
        if (line.trim() === '') continue;

        const parts = line.split('|');
        if (parts.length < 2) continue;

        // u.item title format: "Toy Story (1995)"
        const titleMatch = parts[1].match(/(.*)\s+\((\d{4})\)\s*$/);
        const movie = {
            id: parseInt(parts[0], 10),
            title: parts[1].trim(),
            year: null
        };

        if (titleMatch) {
            movie.title = titleMatch[1].trim();
            movie.year = parseInt(titleMatch[2], 10);
        }

        if (!Number.isNaN(movie.id)) movieData.push(movie);
    }

    return movieData;
}

function parseRatingData(text) {
    const ratingData = [];

    for (const line of text.split(/\r?\n/)) {
        if (line.trim() === '') continue;

        const parts = line.split('\t');
        if (parts.length < 3) continue;

        // u.data columns: user_id | movie_id | rating | timestamp
        const userId = parseInt(parts[0], 10);
        const movieId = parseInt(parts[1], 10);
        const rating = parseFloat(parts[2]);

        if (Number.isNaN(userId) || Number.isNaN(movieId) || Number.isNaN(rating)) continue;
        ratingData.push({ userId, movieId, rating });
    }

    return ratingData;
}
