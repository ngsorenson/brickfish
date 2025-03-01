const express = require('express');
const cors = require('cors');
const app = express();
const port = 3000;

// Use CORS middleware
app.use(cors());

// Middleware to parse JSON bodies
app.use(express.json());

// Basic route
app.get('/', (req, res) => {
  res.send('Hello World!');
});

// Example of a route with a parameter
app.get('/user/:id', (req, res) => {
  const userId = req.params.id;
  res.send(`User ID: ${userId}`);
});

// Example of a POST route
app.post('/user', (req, res) => {
  const newUser = req.body;
  res.status(201).send(newUser);
});

// Start the server
app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});