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
    Interior: 75% PCHIP / 25% Global Quadratic.
    Edges: 50/50 Asymptotic Global Blend (Macro Smile + Flat Extrapolation).
    """
    def __init__(self, filepath):
        self.dataset_path = filepath
        self.fallback_vol = 1e-6
        
        # Blending Logic (Kept your baseline weights)
        self.quad_weight = 0.25
        self.pchip_weight = 0.75
        
        self._initialize_data()
        
    def _initialize_data(self):
        self.raw_df = pd.read_csv(self.dataset_path).reset_index(drop=True)
        # Reverted back to your proven Global Median fallback
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
        if len(y_arr) == 1: return self._clip(y_arr[0])
        
        deg = 1 if len(y_arr) == 2 else 2
        try:
            coefs = np.polyfit(x_arr, y_arr, deg)
            return self._clip(np.polyval(coefs, target_x))
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
            val = pchip_interpolate(x_clean, y_clean, target_x)
            return self._clip(val) if np.isfinite(val) else np.nan
        except:
            return np.nan

    def process_surface(self):
        self.filled_df = self.raw_df.copy()
        
        for exp, meta in self.chain_map.items():
            
            for r_idx in tqdm(self.raw_df.index, desc=f"Reconstructing {exp}"):
                row_data = self.raw_df.loc[r_idx]
                S = row_data["underlying_price"]
                
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

                    known_x = np.array([strikes[cols.index(c)] / S for c in known_cols])
                    known_y = np.array([row_data[c] for c in known_cols])
                    
                    sort_idx = np.argsort(known_x)
                    known_x, known_y = known_x[sort_idx], known_y[sort_idx]
                    
                    min_x, max_x = known_x[0], known_x[-1]
                    missing_cols = np.array(cols)[missing_mask]
                    
                    for m_col in missing_cols:
                        t_x = strikes[cols.index(m_col)] / S if S > 0 else np.nan
                        
                        # --- INTERIOR ---
                        if min_x < t_x < max_x:
                            q_pred = self._get_quadratic_prediction(known_x, known_y, t_x)
                            p_pred = self._get_pchip_prediction(known_x, known_y, t_x)
                            
                            if np.isfinite(q_pred) and np.isfinite(p_pred):
                                val = (self.quad_weight * q_pred) + (self.pchip_weight * p_pred)
                            elif np.isfinite(p_pred): val = p_pred
                            elif np.isfinite(q_pred): val = q_pred
                            else: val = self.g_median
                                
                        # --- LEFT EDGE EXTRAPOLATION (Asymptotic Blend) ---
                        elif t_x <= min_x:
                            # 1. Global Parabola (Uses entire liquid chain, very stable)
                            g_quad = self._get_quadratic_prediction(known_x, known_y, t_x)
                            # 2. Flat Asymptote (Locks to the nearest valid point)
                            flat_wing = known_y[0]
                            
                            if np.isfinite(g_quad):
                                # Blends the curve with the flat line to create a safe, flattening wing
                                val = (0.50 * g_quad) + (0.50 * flat_wing)
                            else:
                                val = flat_wing
                            
                        # --- RIGHT EDGE EXTRAPOLATION (Asymptotic Blend) ---
                        else:
                            # 1. Global Parabola (Uses entire liquid chain, very stable)
                            g_quad = self._get_quadratic_prediction(known_x, known_y, t_x)
                            # 2. Flat Asymptote (Locks to the nearest valid point)
                            flat_wing = known_y[-1]
                            
                            if np.isfinite(g_quad):
                                # Blends the curve with the flat line to create a safe, flattening wing
                                val = (0.50 * g_quad) + (0.50 * flat_wing)
                            else:
                                val = flat_wing
                            
                        self.filled_df.at[r_idx, m_col] = self._clip(val) if np.isfinite(val) else self.g_median

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