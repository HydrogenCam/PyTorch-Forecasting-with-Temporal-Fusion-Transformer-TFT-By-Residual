import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.utils import get_single_col_by_input_type
from utils.utils import extract_cols_from_data_type
from data_formatters.electricity import ElectricityFormatter
from data_formatters.base import DataTypes, InputTypes

class TFTDataset(Dataset, ElectricityFormatter):
    """Dataset Basic Structure for Temporal Fusion Transformer"""
    def __init__(self,
                 data_df,
                 *,
                 formatter: ElectricityFormatter | None = None,
                 use_tlf: bool = True,
                 tlf_replace_with_seasonal: bool = False,
                 err_target:bool = False):
        """
        Make sure ElectricityFormatter.__init__ runs, so _column_definition exists.
        You can pass an existing formatter to keep switches consistent.
        """
        # Call both bases' initializers
        Dataset.__init__(self)
        if formatter is not None:
            ElectricityFormatter.__init__(self,
                use_tlf=formatter.use_tlf,
                tlf_replace_with_seasonal=formatter.tlf_replace_with_seasonal,
                err_target=formatter.err_target,
            )
        else:
            ElectricityFormatter.__init__(self,
                use_tlf=use_tlf,
                tlf_replace_with_seasonal=tlf_replace_with_seasonal,
                err_target=err_target,
            )

        # materialize data
        self.data = data_df.copy().reset_index(drop=True)

        # (Optional) build seasonal baseline in E2 if not already present
        if self.tlf_replace_with_seasonal and "seasonal_baseline" not in self.data.columns:
            # add_seasonal_baseline should create the column in-place
            self.data = add_seasonal_baseline(self.data)

        self.id_col = get_single_col_by_input_type(InputTypes.ID, self._column_definition)
        self.time_col = get_single_col_by_input_type(InputTypes.TIME, self._column_definition)
        self.target_col = get_single_col_by_input_type(InputTypes.TARGET, self._column_definition)
    
        self.input_cols = [
                            tup[0]
                            for tup in self._column_definition
                            if tup[2] not in {InputTypes.ID, InputTypes.TIME}
                          ]
        self.col_mappings = {
                              'identifier': [self.id_col],
                              'time': [self.time_col],
                              'outputs': [self.target_col],
                              'inputs': self.input_cols
                          }
        self.lookback = self.get_time_steps()
        self.num_encoder_steps = self.get_num_encoder_steps()
        
        self.data_index = self.get_index_filtering()
        self.group_size = self.data.groupby([self.id_col]).apply(lambda x: x.shape[0]).mean()
        self.data_index = self.data_index[self.data_index.end_rel < self.group_size].reset_index()
        
    def get_index_filtering(self):
        
        g = self.data.groupby([self.id_col])
        
        df_index_abs = g[[self.target_col]].transform(lambda x: x.index+self.lookback) \
                        .reset_index() \
                        .rename(columns={'index':'init_abs',
                                         self.target_col:'end_abs'})
        df_index_rel_init = g[[self.target_col]].transform(lambda x: x.reset_index(drop=True).index) \
                        .rename(columns={self.target_col:'init_rel'})
        df_index_rel_end = g[[self.target_col]].transform(lambda x: x.reset_index(drop=True).index+self.lookback) \
                        .rename(columns={self.target_col:'end_rel'})
        df_total_count = g[[self.target_col]].transform(lambda x: x.shape[0] - self.lookback + 1) \
                        .rename(columns = {self.target_col:'group_count'})
        
        return pd.concat([df_index_abs, 
                          df_index_rel_init,
                          df_index_rel_end,
                          self.data[[self.id_col]], 
                          df_total_count], axis = 1).reset_index(drop = True)

    def __len__(self):
        # In this case, the length of the dataset is not the length of the training data, 
        # rather the ammount of unique sequences in the data
        return self.data_index.shape[0]

    def __getitem__(self, idx):
    # 1) Slice the time window for this sample
        data_index = self.data.iloc[
        self.data_index.init_abs.iloc[idx] : self.data_index.end_abs.iloc[idx]
       ]

    # 2) Build buckets per your col_mappings: {'identifier','time','inputs','outputs'}
        data_map = {}
        for k in self.col_mappings:
            cols = self.col_mappings[k]
            if k not in data_map:
                data_map[k] = [data_index[cols].values]
            else:
                data_map[k].append(data_index[cols].values)

    # 3) Concatenate along time axis
        for k in data_map:
            data_map[k] = np.concatenate(data_map[k], axis=0)

    # 4) Keep only decoder steps for target
        data_map['outputs'] = data_map['outputs'][self.num_encoder_steps:, :]

    # 5) Optional: active mask for decoder targets
        active_entries = np.ones_like(data_map['outputs'])
        data_map['active_entries'] = active_entries

    # 6) Full input matrix: (time, total_feature_count) excluding ID/TIME
        inputs = data_map['inputs']

    # 7) Collect names by role & dtype from your column_definition
        known_real_names = [
        name for name, dtype, role in self._column_definition
        if role == InputTypes.KNOWN_INPUT and dtype == DataTypes.REAL_VALUED
         ]
        obs_real_names = [
        name for name, dtype, role in self._column_definition
        if role == InputTypes.OBSERVED_INPUT and dtype == DataTypes.REAL_VALUED
         ]
        obs_cate_names = [
        name for name, dtype, role in self._column_definition
        if role == InputTypes.OBSERVED_INPUT and dtype == DataTypes.CATEGORICAL
        ]

    # 8) Map names -> indices according to self.input_cols ordering
        known_real_idx = [self.input_cols.index(n) for n in known_real_names] if known_real_names else []
        obs_real_idx   = [self.input_cols.index(n) for n in obs_real_names] if obs_real_names else []
        obs_cate_idx   = [self.input_cols.index(n) for n in obs_cate_names] if obs_cate_names else []

    # 9) Slice each group and set dtypes
        x_known_real     = inputs[:, known_real_idx].astype(np.float32) if known_real_idx else np.zeros((inputs.shape[0], 0), 
       dtype=np.float32)
        x_observed_real  = inputs[:, obs_real_idx].astype(np.float32)   if obs_real_idx else np.zeros((inputs.shape[0], 0), 
        dtype=np.float32)
    # categorical must be int64 for embedding
        x_observed_cate  = inputs[:, obs_cate_idx].astype(np.int64)     if obs_cate_idx else np.zeros((inputs.shape[0], 0), dtype=np.int64)

    # 10) Return tuple expected by your model
        return (x_known_real, x_observed_real, x_observed_cate), data_map['outputs'].astype(np.float32)
