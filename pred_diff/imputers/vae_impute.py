from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .imputer_base import ImputerBase, ImputerBaseTabular
from .tabular_vae import TabularVAE


class VAEImputer(ImputerBaseTabular):
    '''
    imputes using a VAE and pseudo Gibbs sampling
    '''
    
    def __init__(self, df_train: pd.DataFrame, **kwargs):
        super().__init__(df_train=df_train)
        kwargs["bs"] = kwargs["bs"] if "bs" in kwargs.keys() else 64
        kwargs["bs"] = 64
        kwargs["bs_eval"] = kwargs["bs_eval"] if "bs_eval" in kwargs.keys() else 128
        kwargs["n_hidden"] = kwargs["n_hidden"] if "n_hidden" in kwargs.keys() else [1000,250]
        kwargs["lr"] = kwargs["lr"] if "lr" in kwargs.keys() else 1e-3
        kwargs["epochs"] = kwargs["epochs"] if "epochs" in kwargs.keys() else 20
        kwargs["gibbs_iterations"] = kwargs["gibbs_iterations"] if "gibbs_iterations" in kwargs.keys() else 10
        kwargs["gpus"] = kwargs["gpus"] if "gpus" in kwargs.keys() else 0
        kwargs["n_jobs"] = kwargs["n_jobs"] if "n_jobs" in kwargs.keys() else 8
        self.clip_min = 0
        self.clip_max = 1
        self.kwargs = kwargs
        self.imputer_name = 'VAEImputer'

    def _impute(self, df_test, impute_cols, n_imputations=100):
        res=[]    
        train_ds = TensorDataset(torch.from_numpy(np.array(self.df_train)).float(), torch.zeros(len(self.df_train)))
        train_dl = DataLoader(train_ds, batch_size=self.kwargs["bs"], drop_last=False, num_workers=self.kwargs["n_jobs"])
        test_ds = TensorDataset(torch.from_numpy(np.array(df_test)).float(), torch.zeros(len(df_test)))
        test_dl = DataLoader(test_ds, batch_size=self.kwargs["bs_eval"], drop_last=False, num_workers=self.kwargs["n_jobs"])
        
        #fit the VAE
        # if(self.imputer is None or retrain is True):
        if self.imputer is None:        # always retrain
            print('Train TabularvVAE')
            self.imputer = TabularVAE(n_input=28**2, learning_rate=self.kwargs["lr"])
            trainer = Trainer(gpus=self.kwargs["gpus"], progress_bar_refresh_rate=20, max_epochs=self.kwargs["epochs"])#,checkpoint_callback=False)
            trainer.fit(self.imputer,train_dl,test_dl)
        else:
            print("Info: using trained imputer; pass retrain=True to retrain") 

        col2i = {s:i for i,s in enumerate(self.df_train.columns)}
        
        with torch.no_grad():
            # for impute_col in tqdm(impute_cols):
            impute_col = impute_cols
            res_data = []
            res_ids = []
            res_imputation_ids = []
            impute_col_num = [col2i[xx] for xx in impute_col]
            for batch_id, (xinput,_) in enumerate(test_dl):
                for imp in range(n_imputations):
                    x=xinput.clone().view(xinput.size(0),-1)
                    for i in range(self.kwargs["gibbs_iterations"]):
                        if(i==0):
                            for ic,icn in zip(impute_col,impute_col_num):
                                x[:,icn].uniform_(min(self.df_train[ic].min(), df_test[ic].min()), max(self.df_train[ic].max(), df_test[ic].max()))
                        else:
                            x[:,impute_col_num] = xhat[:,impute_col_num]
                        xhat = self.imputer(x).detach()
                    #xret = x.clone()
                    #xret[:,impute_col_num] = xhat[:,impute_col_num]
                    res_data.append(xhat[:,impute_col_num].cpu().numpy())
                    res_ids+=list(range(batch_id*self.kwargs["bs_eval"],batch_id*self.kwargs["bs_eval"]+len(xinput)))
                    res_imputation_ids+=[imp]*len(xinput)
            res_data=np.concatenate(res_data,axis=0)
            df_test_tmp= pd.DataFrame(data=res_data,columns=impute_col)
            df_test_tmp["id"] = res_ids
            # df_test_tmp["imputation_id"] = res_imputation_ids
            return_reduced = False
            if(return_reduced is False):
                non_tmp_cols = [a for a in df_test.columns if not a in df_test_tmp.columns]
                df_test_tmp=df_test_tmp.join(df_test[non_tmp_cols], how='left', on="id")
            # res.append(df_test_tmp)
            # convert back in correct format, [imputation1, 2, 3,..]
            grouped = df_test_tmp.groupby('id')
            list_test_tmp = [np.array(temp[1][self.df_train.columns]) for temp in grouped]
            all_in_np = np.array([tmp.reshape(-1, 28, 28) for tmp in list_test_tmp]).swapaxes(0, 1)          # n_imputations, n_samples, n_features
            # list_df = [pd.DataFrame(all_in_np[i], columns=self.columns) for i in range(all_in_np.shape[0])]

            # all_in_np[all_in_np < self.clip_min] = self.clip_min
            # all_in_np[all_in_np > self.clip_max] = self.clip_max
        return all_in_np, None
