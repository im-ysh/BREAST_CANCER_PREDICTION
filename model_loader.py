import torch
import joblib

class BinClr(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.order = torch.nn.Sequential(
            torch.nn.Linear(30, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.order(x)

def load_model_and_scaler():
    model = BinClr()
    model.load_state_dict(torch.load("model/model.pt", map_location=torch.device('cpu')))
    model.eval()
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler
