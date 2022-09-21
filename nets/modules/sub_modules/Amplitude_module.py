import torch.nn as nn


class Amplitude_module(nn.Module):
    def __init__(self):
        super(Amplitude_module, self).__init__()

        # self.dbp = nn.Sequential(
        #
        # )
        self.dbp = nn.AdaptiveAvgPool1d(1)  # dbp
        self.sbp = nn.AdaptiveAvgPool1d(1)  # sbp
        self.amp = nn.AdaptiveAvgPool1d(1)  # amplitude of abp
        # self.mbp = nn.Linear(360, 1)  # mbp


    def forward(self, feature):
        dbp = self.dbp(feature)
        sbp = self.sbp(feature)
        '''
        to reduce load of model 
        amp is set dependent variable of sbp
        '''
        # amp = self.amp(sbp)

        return dbp, sbp
