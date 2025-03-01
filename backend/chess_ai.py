import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import chess.pgn
import chess.engine
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.activation = nn.SELU()

    def forward(self, x):
        x_input = x  # Keep gradient tracking
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + x_input  # Skip connection
        return self.activation(x)

class ChessNet(nn.Module):
    def __init__(self, hidden_layers=4, hidden_size=200):
        super(ChessNet, self).__init__()
        self.input_layer = nn.Conv2d(6, hidden_size, 3, stride=1, padding=1)  # Revert to 6 input channels
        self.residual_blocks = nn.Sequential(*[ResidualBlock(hidden_size) for _ in range(hidden_layers)])
        self.output_layer = nn.Conv2d(hidden_size, 2, 3, stride=1, padding=1)  # Revert to 2 output channels

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        x = self.residual_blocks(x)
        x = self.output_layer(x)
        return x


def load_chess_model(model_path="models/chessnet_model_2.pth"):
    model = ChessNet()
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except RuntimeError as e:
        print(f"Warning: Loading with strict=False due to mismatch: {e}")
        model.load_state_dict(new_state_dict, strict=False)
    
    model.eval()
    return model

def fen_to_tensor(fen):
    """ Convert a FEN string to a 12-channel tensor representation """
    board = chess.Board(fen)
    piece_map = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5, 'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}
    
    board_matrix = torch.zeros((12, 8, 8), dtype=torch.float32)
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            idx = piece_map[str(piece)]
            board_matrix[idx, row, col] = 1

    return board_matrix.unsqueeze(0)  # Add batch dimension

def predict_move(fen, model):
    board = chess.Board(fen)
    x = fen_to_tensor(fen)
    
    with torch.no_grad():
        move_scores = model(x).numpy()  # Convert to NumPy
    
    legal_moves = list(board.legal_moves)
    
    move_values = {}
    for move in legal_moves:
        move_uci = move.uci()
        from_sq, to_sq = move.from_square, move.to_square
        row_from, col_from = divmod(from_sq, 8)
        row_to, col_to = divmod(to_sq, 8)
        
        move_values[move_uci] = move_scores[row_to, col_to]  # Use predicted move score

    best_move = max(move_values, key=move_values.get)
    return best_move
