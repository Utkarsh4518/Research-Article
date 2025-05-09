import torch
import numpy as np
from model.unet import UNet
from model.reinforcement import DQN
from data_preprocessing.noise_removal import remove_noise
from data_preprocessing.normalize import normalize_eeg

class EpilepsyPredictor:
    def __init__(self, model_path='models/u_trgn.pth'):
        """
        Load trained U-TRGN model for epilepsy prediction.
        Args:
            model_path: Path to saved model weights
        """
        self.unet = UNet()
        self.dqn = DQN()
        state_dict = torch.load(model_path)
        self.unet.load_state_dict(state_dict['unet'])
        self.dqn.load_state_dict(state_dict['dqn'])
        self.unet.eval()
        self.dqn.eval()

    def predict(self, eeg_signal):
        """
        Predict epilepsy from raw EEG signal.
        Args:
            eeg_signal: Raw EEG data (shape: [178,])
        Returns:
            dict: {'prediction': 0/1, 'probability': float, 'confidence': float}
        """
        # Preprocess
        X = np.array(eeg_signal).reshape(1, -1)
        X_clean, _ = remove_noise(X, method='wavelet')
        X_norm, _ = normalize_eeg(X_clean)
        
        # Feature extraction
        with torch.no_grad():
            features = self.unet(torch.FloatTensor(X_norm))
            q_values = self.dqn(features)
            prob = torch.softmax(q_values, dim=1)[0, 1].item()
        
        return {
            'prediction': int(prob > 0.5),
            'probability': prob,
            'confidence': abs(prob - 0.5) * 2  # 0-1 scale
        }

if __name__ == "__main__":
    # Example usage
    predictor = EpilepsyPredictor()
    sample_eeg = np.random.randn(178)  # Replace with real EEG
    print(predictor.predict(sample_eeg))