#About
Brickfish was created during USU Hackathon 2025. A React frontend and a Python backend allow the user to play chess vs 3 different AI models we trained on various partitions of a dataset of roughly 6 million games or vs Catfish (CatBot) which is a more traditional algorithm-based approach.

# Dependencies
- [Python](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/installation/)
- [Node.js](https://nodejs.org/en/download/)
- [React Chessboard](https://www.npmjs.com/package/react-chessboard)
- [python-chess](https://python-chess.readthedocs.io/en/latest/)
- [Vite](https://vite.dev/guide/)
- [React](https://react.dev/)
- Python libraries besides standard as found in /brickfish/backend/requirements.txt
- torch
- pandas
- numpy

# How to Run

1. Clone the repository:

    ```bash
    git clone https://github.com/ngsorenson/brickfish
    ```

2. Navigate to the `backend` folder:

    ```bash
    cd brickfish/backend
    ```

3. Install the required Python packages:

    ```bash
    python3 -m pip install -r requirements.txt
    ```

4. Run the backend:

    ```bash
    python3 backEnd.py
    ```

5. In another terminal window, navigate back to the main folder:

    ```bash
    cd ../brickfish
    ```

6. Install the necessary Node.js packages:

    ```bash
    npm i
    ```

7. Run the development server:

    ```bash
    npm run dev
    ```

8. Access the server in your browser, default is http://localhost:5173/, or you can just type o + enter in the terminal you ran the development server and Vite will open it in your default browser.

