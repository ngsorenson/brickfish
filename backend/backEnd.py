from flask import Flask, request, jsonify
from flask_cors import CORS
import chess
import json
import uuid
from chess_ai import load_chess_model, predict_move
from algorithm_bot import Bot

app = Flask(__name__)
CORS(app)

games = {}
model = load_chess_model('models/chessnet_model_2.pth')
cat_fish = Bot()

class BrickFish:
    def __init__(self):
        self.board = chess.Board()
        # print(self.board)  # Display initial board

    def move_piece(self, move_uci):
        """Executes a move if legal."""
        move = chess.Move.from_uci(move_uci)
        if move in self.board.legal_moves:
            self.board.push(move)
            return True
        else:
            return False

    def bot_move(self):
        move_uci = predict_move(self.board.fen(), model)
        move = chess.Move.from_uci(move_uci)
        self.board.push(move)
        return self.board_state()
    
    def cat_bot_move(self):
        move = cat_fish.bestMove(self.board)
        self.board.push(move)
        return self.board_state()
    
    def legal_moves(self, pos_start=None):
        if not pos_start:
            moves = [move.uci() for move in self.board.legal_moves]
        else:
            pos_square = chess.parse_square(pos_start)
            moves = [move.uci() for move in self.board.legal_moves if move.from_square == pos_square]

        return json.dumps({"legal_moves": moves}, indent=4)
    
    def board_state(self):
        return repr(self.board).split("'").pop(1).split(' ').pop(0)
    
# def main():
#     game = BrickFish()
    
#     print(game.board_state())
#     print(game.legal_moves("e2"))
#     game.move_piece("e2e4")
#     print(game.board_state())
#     print(game.legal_moves("e2"))

@app.route('/newGame', methods=['GET'])
def new_game():
    game_id = str(uuid.uuid4())
    game = BrickFish()
    games[game_id] = game
    return ({"game_id": game_id, "board_state": game.board_state()})

@app.route('/legalMoves/<id>', methods=['GET'])
def legal_moves(id):
    return games[id].legal_moves()

@app.route('/legalMoves/<id>/<piece>', methods=['GET'])
def legal_moves_with_piece(id, piece):
    return games[id].legal_moves(piece)

@app.route('/boardState/<id>', methods=['GET'])
def game_state(id):
    return games[id].board_state()

@app.route('/movePiece/<id>/<move>', methods=['POST'])
def move_piece(id, move):
    # data = request.get_json()
    # move = data[move]
    if games[id].move_piece(move):
        return {"board_state": games[id].board_state()}
    else:
        return {"board_state": games[id].board_state()}

@app.route('/botMove/<id>', methods=['GET'])
def bot_move(id):
    print(games[id].bot_move())
    print(games[id].board_state())
    return {'board_state': games[id].board_state()}

@app.route('/catBotMove/<id>', methods=['GET'])
def catbot_move(id):
    return {"board_state" : games[id].cat_bot_move()}


# if __name__ == "__main__":
#     main()

if __name__ == '__main__':
    app.run(debug=True)