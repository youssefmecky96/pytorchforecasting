## Changes to Original Package

This document outlines the modifications made to the original package, focusing on the `temporal_fusion_transformer` and `timeseries` modules, as well as the `utils.py` file. The changes address issues related to data types, device allocation, and data handling specific to PyTorch forecasting.

### Changes in `temporal_fusion_transformer` Module

**File:** `models/temporal_fusion_transformer/__init__.py`

1. **Line 391**
   - **Reason:** `decoder_mask` was on CPU for an unknown reason.
   - **Change:**
     ```python
     decoder_mask = decoder_mask.to(encoder_mask.device)
     ```

2. **Line 371**
   - **Reason:** `decoder_length` was either an integer or a tensor.
   - **Change:**
     ```python
     if not (isinstance(decoder_length, int)):
         decoder_length = int(decoder_length.item())
     ```

### Changes in `utils.py`

**File:** `utils.py`

1. **Line 135**
   - **Reason:** `size` was either a tensor or an integer.
   - **Change:**
     ```python
     if not (isinstance(size, int)):
         size = int(size.item())
     ```

### Changes in `timeseries` Module

**File:** `data/timeseries.py`

1. **Line 814**
   - **Reason:** Avoid redundant scaling of data, which is done automatically by PyTorch forecasting.
   - **Change:**
     ```python
     for name in self.reals:
         if name in self.target_names or name in our lagged_variables or len(self.scalers) == 0:
     ```

2. **Line 1268**
   - **Reason:** To adapt the dataset for Length of Stay prediction, avoiding choosing the last time steps for every data point when `predict_mode=True`.
   - **Change:**
     ```python
     df_index = df_index[
         lambda x: (x["time"] - x["time_first"] + 1 <= max_sequence_length)
         & (x["sequence_length"] >= min_sequence_length)
     ]
     ```
