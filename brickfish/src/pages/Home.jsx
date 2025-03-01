import '../App.css';
import MainLayout from '../layout/MainLayout';
import { Chessboard } from 'react-chessboard';
import { useEffect, useState } from 'react';
import { Chess } from 'chess.js';

const Home = () => {
    const [gameId, setGameId] = useState(null);
    const [boardState, setBoardState] = useState('start');
    const [legalMoves, setLegalMoves] = useState([]);
    const [chess] = useState(new Chess());

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
            localStorage.setItem('legalMoves', data.legal_moves);
            console.log(data.legal_moves);
        } catch (error) {
            console.error('Error getting legal moves:', error);
        }
    }

    function onClickPiece() {
        console.log('Piece clicked');
    }

    return (
        <>
            <MainLayout title="Brickfish Home">
                <div className="container-fluid">
                    <div className="row">
                        <div className="col-1 sidebar">
                            Sidebar Content
                            <button onClick={newGame}>New Game</button>
                        </div>
                        <div className="col-2"></div>
                        <div className="col-6 chessboard-container">
                            <Chessboard position={boardState} />
                        </div>
                        <div className="col-3"></div>
                    </div>
                </div>
            </MainLayout>
        </>
    )
}

export default Home;