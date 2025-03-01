import '../App.css';
import MainLayout from '../layout/MainLayout';
import { Chessboard } from 'react-chessboard';
import { useEffect } from 'react';
import { Chess } from 'chess.js';

const ID = 'placeholder'

const Home = () => {

    return (
        <MainLayout title="Brickfish Home">
            <h1>Home</h1>

        </MainLayout>
    )

    // function onPieceClick

    function startGame() {
        alert('startGame')
    }
}

export default Home