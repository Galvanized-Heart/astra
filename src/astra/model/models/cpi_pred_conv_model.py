import torch
import torch.nn as nn


# Orignial class name from CPI-Pred's Z05X_All_Models.py: SQembConv_CPenc_Model_v2
# TODO: Include github link once published

class CpiPredConvModel(nn.Module):
    def __init__(self,
                 in_dim   : int, 
                 hid_dim  : int, 
                 kernal_1 : int, 
                 out_dim  : int, 
                 kernal_2 : int, 
                 max_len  : int, 
                 cmpd_dim : int, 
                 last_hid : int, 
                 dropout  : float = 0. ,
                 ffn_out  : int   =  3 , 
                 ):


        super().__init__()
        self.norm       = nn.BatchNorm1d(in_dim)
        self.conv1      = nn.Conv1d(in_dim, hid_dim, kernal_1, padding = int((kernal_1-1)/2))
        self.dropout1   = nn.Dropout(dropout, inplace = False)
        
        self.conv2_1    = nn.Conv1d(hid_dim, out_dim, kernal_2, padding = int((kernal_2-1)/2))
        self.dropout2_1 = nn.Dropout(dropout, inplace = False)
        
        self.conv2_2    = nn.Conv1d(hid_dim, hid_dim, kernal_2, padding = int((kernal_2-1)/2))
        self.dropout2_2 = nn.Dropout(dropout, inplace = False)
        
        self.cmpd_enc   = nn.Linear(cmpd_dim, in_dim)
        self.cmpd_drop  = nn.Dropout(dropout, inplace = False)

        self.fc_early   = nn.Linear(max_len * hid_dim + in_dim, 1)
        
        self.conv3      = nn.Conv1d(hid_dim, out_dim, kernal_2, padding = int((kernal_2-1)/2))
        self.dropout3   = nn.Dropout(dropout, inplace = False)
        #self.pooling   = nn.MaxPool1d(3, stride = 3,padding = 1)
        
        self.fc_1 = nn.Linear(int(2 * max_len * out_dim + in_dim), last_hid)
        self.fc_2 = nn.Linear(last_hid, last_hid)
        self.fc_3 = nn.Linear(last_hid, ffn_out) 
        self.cls  = nn.Sigmoid()


    def forward(self, protein_embedding=None, ligand_embedding=None, **kwargs):
        
        # Adapt inputs to CPI-Pred's code
        emb_inputs = protein_embedding
        compound = ligand_embedding

        output = emb_inputs.transpose(1, 2)
        output = nn.functional.relu(self.conv1(self.norm(output)))
        output = self.dropout1(output)
        
        output_1 = nn.functional.relu(self.conv2_1(output))
        output_1 = self.dropout2_1(output_1)
        
        output_2 = nn.functional.relu(self.conv2_2(output)) + output
        output_2 = self.dropout2_2(output_2)
        
        compound = self.cmpd_drop(nn.functional.relu(self.cmpd_enc(compound)))

        single_conv = torch.cat( (torch.flatten(output_2,1), compound) ,1)
        single_conv = self.cls(self.fc_early(single_conv))
        
        output_2 = nn.functional.relu(self.conv3(output_2))
        output_2 = self.dropout3(output_2)
        
        output = torch.cat((output_1, output_2), 1)
        
        #output = self.pooling(output)
        
        output = torch.cat( (torch.flatten(output,1), compound) ,1)
        
        output = self.fc_1(output)
        output = nn.functional.relu(output)
        output = self.fc_2(output)
        output = nn.functional.relu(output)
        output = self.fc_3(output)
        
        return output#, single_conv
    




    