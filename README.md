## Changes to original package 
file: models/temporal_fusion_transformer/__init__.py line 391 
reason:decoder_mask=decoder_mask.to(encoder_mask.device) because decoder_mask for some reason on cpu
change:decoder_mask=decoder_mask.to(encoder_mask.device) 

file: models/temporal_fusion_transformer/__init__.py line 371 
reason: decoder_length was either an int or a tensor
change: if(not (isinstance(decoder_length,int))):
            decoder_length=int(decoder_length.item())
file: utils.py line 135
reason:size was either tensor or int 
change: if(not (isinstance(size,int))):
            size=int(size.item())
file: data/timeseries.py line 814
reason: no need to scale data again which is done automatically by pytorch forecatsing 
change: for name in self.reals:
            if name in self.target_names or name in self.lagged_variables or len(self.scalers)==0:
file data/timeseries.py line 1268 
reason: for the dataset if you set predict_mode=True it chooses the last time steps for every data point which in our case for Length of stay prediction would be zero so we 
made the change to get the datapoint from the begining of the series
change: df_index = df_index[
                lambda x: (x["time"]-x["time_first"]  + 1 <= max_sequence_length)
                & (x["sequence_length"] >= min_sequence_length)
            ]
