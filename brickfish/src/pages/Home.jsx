import '../App.css';
import MainLayout from '../layout/MainLayout';
import { Chessboard } from 'react-chessboard';
import { useEffect, useState } from 'react';

const Home = () => {
    const [gameId, setGameId] = useState(null);
    const [boardState, setBoardState] = useState('start');
    const [legalMoves, setLegalMoves] = useState([]);
    const [toGo, setToGo] = useState('w');

    useEffect(() => {
        let id = localStorage.getItem('ID');
        let boardState = localStorage.getItem('boardState');

        if (!(id && boardState)) {
            newGame();
        } else {
            setGameId(id);
            setBoardState(boardState);
        }
    }, []);

    useEffect(() => {
        getLegalMoves();
    }, [boardState]);

    async function newGame() {
        try {
            const response = await fetch('http://localhost:5000/newGame', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            const data = await response.json();
            console.log(data.game_id);
            console.log(data.board_state);
            setGameId(data.game_id);
            setBoardState(data.board_state);
            localStorage.setItem('ID', data.game_id);
            localStorage.setItem('boardState', data.board_state);
            getLegalMoves();
            setToGo('w');

        } catch (error) {
            console.error('Error starting new game:', error);
        }
    }

    async function getLegalMoves() {
        try {
            const response = await fetch(`http://localhost:5000/legalMoves/${gameId}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                }
            });
            const data = await response.json();
            if (data.legal_moves) {
                setLegalMoves(data.legal_moves);
                localStorage.setItem('legalMoves', data.legal_moves);
                console.log(data.legal_moves);
                if (legalMoves.length === 0) {
                    alert('Checkmate');
                }
            } else {
                console.log('Error: No legal moves returned');
            }
        } catch (error) {
            console.error('Error getting legal moves:', error);
        }
    }

    async function makeMove(move) {
        try {
            const response = await fetch(`http://localhost:5000/movePiece/${gameId}/${move}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            const data = await response.json();
            if (data.board_state) {
                setBoardState(data.board_state);
                localStorage.setItem('boardState', data.board_state);
                getLegalMoves();
            } else {
                console.log('Error: No board state returned');
            }
        } catch (error) {
            console.error('Error making move:', error);
        }
    }

    function onDrop(sourceSquare, targetSquare, piece) {
        console.log('Piece clicked', sourceSquare, targetSquare, piece);
        let move = sourceSquare + targetSquare;
        console.log('Move:', move);
        console.log('Legal Moves:', legalMoves);
        if (legalMoves.includes(move)) {
            makeMove(move);
            setToGo(toGo === 'w' ? 'b' : 'w');
            if (legalMoves.length === 0) {
                alert('Checkmate');
            }
        }
    }

    const whoToMove = toGo === 'w' ? 'White' : 'Black';

    return (
        <>
            <MainLayout title="Brickfish Home">
                <div className="container-fluid">
                    <div className="row">
                        <div className="col-1 sidebar p-1" style={{ backgroundColor: 'grey' }}>
                            <button onClick={newGame}>New Game</button>
                        </div>
                        <div className="col-2">To move: {whoToMove}</div>
                        <div className="col-6 chessboard-container">
                            <Chessboard position={boardState}
                                onPieceDrop={onDrop} />
                        </div>
                        <div className="col-3">GOOGLE ADS GO HERE</div>
                    </div>
                </div>
            </MainLayout>
        </>
    )
}

export default Home;