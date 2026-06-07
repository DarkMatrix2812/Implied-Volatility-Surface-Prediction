import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython.display import FileLink

# Graceful fallback for PCHIP
try:
    from scipy.interpolate import pchip_interpolate
    HAS_PCHIP = True
except ImportError:
    HAS_PCHIP = False

# ─── CONFIG ───────────────────────────────────────────────────────────────
DATASET_PATH = '/kaggle/input/competitions/finclub-open-project-26/dataset.csv'
SEPARATOR    = '||'

class VolatilityReconstructor:
    """
    Object-oriented IV Surface Imputer.
    Interior: 75% PCHIP / 25% Global Quadratic (in Variance Space).
    Edges: Overdetermined Anchor Extrapolation.
    """
    def __init__(self, filepath):
        self.dataset_path = filepath
        self.fallback_vol = 1e-6
        
        # Blending Logic
        self.quad_weight = 0.25
        self.pchip_weight = 0.75
        
        self.anchor_points = 6        # Test 7 or 8 here if you want to push further!
        self.use_log_moneyness = True # Fits x = ln(K/S)
        self.use_variance = True      # Fits y = IV^2
        # ------------------------------
        
        self._initialize_data()
        
    def _initialize_data(self):
        self.raw_df = pd.read_csv(self.dataset_path).reset_index(drop=True)
        self.g_median = float(self.raw_df.drop(columns=['datetime', 'underlying_price']).stack().median())
        
        self.opt_cols = [c for c in self.raw_df.columns if c.endswith('CE') or c.endswith('PE')]
        self.expiries = sorted(list(set(re.search(r'NIFTY(\w{7})', c).group(1) for c in self.opt_cols)))
        
        self.chain_map = {}
        for exp in self.expiries:
            puts = sorted([c for c in self.opt_cols if exp in c and c.endswith('PE')], key=lambda x: int(x[-7:-2]))
            calls = sorted([c for c in self.opt_cols if exp in c and c.endswith('CE')], key=lambda x: int(x[-7:-2]))
            self.chain_map[exp] = {
                'ce_cols': calls,
                'pe_cols': puts,
                'ce_strikes': np.array([int(c[-7:-2]) for c in calls]),
                'pe_strikes': np.array([int(c[-7:-2]) for c in puts])
            }

    def _clip(self, v):
        return max(float(v), self.fallback_vol) if np.isfinite(v) else np.nan

    def _get_quadratic_prediction(self, x_arr, y_arr, target_x):
        """Standard Polynomial Curve Fitting"""
        if len(y_arr) == 0: return np.nan
        if len(y_arr) == 1: return y_arr[0]
        
        deg = 1 if len(y_arr) == 2 else 2
        try:
            coefs = np.polyfit(x_arr, y_arr, deg)
            return np.polyval(coefs, target_x)
        except Exception:
            return np.nan

    def _get_pchip_prediction(self, x_arr, y_arr, target_x):
        """Standard PCHIP Interpolation"""
        if not HAS_PCHIP or len(y_arr) < 3: return np.nan

        idx_sort = np.argsort(x_arr)
        x_clean, y_clean = x_arr[idx_sort], y_arr[idx_sort]
        
        u_x, indices = np.unique(x_clean, return_inverse=True)
        if len(u_x) != len(x_clean):
            y_mean = np.bincount(indices, weights=y_clean) / np.bincount(indices)
            x_clean, y_clean = u_x, y_mean

        if not (x_clean[0] <= target_x <= x_clean[-1]): return np.nan
        
        try:
            return pchip_interpolate(x_clean, y_clean, target_x)
        except:
            return np.nan

    def process_surface(self):
        self.filled_df = self.raw_df.copy()
        
        for exp, meta in self.chain_map.items():
            
            for r_idx in tqdm(self.raw_df.index, desc=f"Reconstructing {exp}"):
                row_data = self.raw_df.loc[r_idx]
                S = row_data["underlying_price"]
                if S <= 0: continue
                
                for opt_type in ['CE', 'PE']:
                    cols = meta['ce_cols'] if opt_type == 'CE' else meta['pe_cols']
                    strikes = meta['ce_strikes'] if opt_type == 'CE' else meta['pe_strikes']
                    
                    missing_mask = pd.isna(row_data[cols])
                    if not missing_mask.any(): continue
                        
                    known_cols = np.array(cols)[~missing_mask]
                    if len(known_cols) == 0:
                        for c in np.array(cols)[missing_mask]:
                            self.filled_df.at[r_idx, c] = self.g_median
                        continue

                    # --- 1. COORDINATE TRANSFORMATION ---
                    raw_x = np.array([strikes[cols.index(c)] / S for c in known_cols])
                    known_x = np.log(raw_x) if self.use_log_moneyness else raw_x
                    
                    # --- 2. VARIANCE TRANSFORMATION ---
                    raw_y = np.array([row_data[c] for c in known_cols]).astype(float)
                    known_y = raw_y ** 2 if self.use_variance else raw_y
                    
                    sort_idx = np.argsort(known_x)
                    known_x, known_y = known_x[sort_idx], known_y[sort_idx]
                    
                    min_x, max_x = known_x[0], known_x[-1]
                    missing_cols = np.array(cols)[missing_mask]
                    
                    for m_col in missing_cols:
                        raw_t_x = strikes[cols.index(m_col)] / S
                        t_x = np.log(raw_t_x) if self.use_log_moneyness else raw_t_x
                        
                        # --- INTERIOR ---
                        if min_x < t_x < max_x:
                            q_pred = self._get_quadratic_prediction(known_x, known_y, t_x)
                            p_pred = self._get_pchip_prediction(known_x, known_y, t_x)
                            
                            if np.isfinite(q_pred) and np.isfinite(p_pred):
                                val = (self.quad_weight * q_pred) + (self.pchip_weight * p_pred)
                            elif np.isfinite(p_pred): val = p_pred
                            elif np.isfinite(q_pred): val = q_pred
                            else: val = np.nan
                                
                        # --- LEFT EDGE EXTRAPOLATION ---
                        elif t_x <= min_x:
                            anchor_x = known_x[:self.anchor_points]
                            anchor_y = known_y[:self.anchor_points]
                            val = self._get_quadratic_prediction(anchor_x, anchor_y, t_x)
                            
                        # --- RIGHT EDGE EXTRAPOLATION ---
                        else:
                            anchor_x = known_x[-self.anchor_points:]
                            anchor_y = known_y[-self.anchor_points:]
                            val = self._get_quadratic_prediction(anchor_x, anchor_y, t_x)
                            
                        # --- 3. REVERT VARIANCE TO VOLATILITY ---
                        if np.isfinite(val):
                            if self.use_variance:
                                # Ensure we don't sqrt a negative number if the parabola dips
                                val = np.sqrt(max(val, self.fallback_vol**2))
                            self.filled_df.at[r_idx, m_col] = self._clip(val)
                        else:
                            self.filled_df.at[r_idx, m_col] = self.g_median

    def save_and_submit(self):
        self.filled_df.to_csv('filled_dataset.csv', index=False)
        rows = []
        for col in self.opt_cols:
            for idx in self.raw_df.index[self.raw_df[col].isna()]:
                dt = self.raw_df.at[idx, 'datetime']
                rows.append({'id': f"{dt}||{col}", 'value': self.filled_df.at[idx, col]})
                
        sub = pd.DataFrame(rows).sort_values('id').reset_index(drop=True)
        sub.to_csv('submission.csv', index=False)
        print(f"submission.csv: {len(sub)} rows | NaN={sub['value'].isna().sum()} | range=[{sub['value'].min():.5f}, {sub['value'].max():.5f}]")

# ─── EXECUTION ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    imputer = VolatilityReconstructor(DATASET_PATH)
    imputer.process_surface()
    imputer.save_and_submit()

    display(FileLink('submission.csv'))
    display(FileLink('filled_dataset.csv'))