const express = require('express');
const router = express.Router();
const gameController = require('../controllers/gameController');

// Routes for game logic
router.get('/newGame', gameController.newGame);
router.

module.exports = router;