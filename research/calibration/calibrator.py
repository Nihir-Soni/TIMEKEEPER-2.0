import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

class UncertaintyCalibrator:
    def __init__(self, method='isotonic'):
        self.method = method
        if self.method == 'isotonic':
            self.model = IsotonicRegression(out_of_bounds='clip')
        else:
            raise ValueError(f"Calibration method {method} not supported yet.")
            
    def fit(self, raw_uncertainty_1d, actual_error_1d):
        """
        Fit the calibrator. 
        Input arrays should be flattened 1D numpy arrays of all pixels in the validation set.
        """
        self.model.fit(raw_uncertainty_1d, actual_error_1d)
        
    def calibrate(self, raw_uncertainty_map):
        """
        Transform a raw uncertainty map (2D or 3D) into a calibrated map.
        """
        shape = raw_uncertainty_map.shape
        flat = raw_uncertainty_map.flatten()
        calibrated_flat = self.model.predict(flat)
        return calibrated_flat.reshape(shape)
        
    def save(self, filepath):
        joblib.dump(self, filepath)
        
    @classmethod
    def load(self, filepath):
        return joblib.load(filepath)
