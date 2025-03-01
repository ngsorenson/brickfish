import '../App.css';
import MainLayout from '../layout/MainLayout';
import { Chessboard } from 'react-chessboard';
import { useEffect } from 'react';
import { Chess } from 'chess.js';

const ID = 'placeholder'

const Home = () => {
    return (
        <>
            <MainLayout title="Brickfish Home">
                <div className="container-fluid">
                    <div className="row">
                        <div className="col-1 sidebar">
                            Sidebar Content
                        </div>
                        <div className="col-3"></div>
                        <div className="col-6 chessboard-container">
                            <Chessboard />
                        </div>
                        <div className="col-2"></div>
                    </div>
                </div>
            </MainLayout>
        </>
    )

    // function onPieceClick

    function startGame() {
        alert('startGame')
    }
}

export default Home