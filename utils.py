import pandas as pd
import numpy as np

from tqdm import tqdm

dtypes = {
        'MachineIdentifier':                                    'category',
        'ProductName':                                          'category',
        'EngineVersion':                                        'category',
        'AppVersion':                                           'category',
        'AvSigVersion':                                         'category',
        'IsBeta':                                               'int8',
        'RtpStateBitfield':                                     'float16',
        'IsSxsPassiveMode':                                     'int8',
        'DefaultBrowsersIdentifier':                            'float16',
        'AVProductStatesIdentifier':                            'float32',
        'AVProductsInstalled':                                  'float16',
        'AVProductsEnabled':                                    'float16',
        'HasTpm':                                               'int8',
        'CountryIdentifier':                                    'int16',
        'CityIdentifier':                                       'float32',
        'OrganizationIdentifier':                               'float16',
        'GeoNameIdentifier':                                    'float16',
        'LocaleEnglishNameIdentifier':                          'int8',
        'Platform':                                             'category',
        'Processor':                                            'category',
        'OsVer':                                                'category',
        'OsBuild':                                              'int16',
        'OsSuite':                                              'int16',
        'OsPlatformSubRelease':                                 'category',
        'OsBuildLab':                                           'category',
        'SkuEdition':                                           'category',
        'IsProtected':                                          'float16',
        'AutoSampleOptIn':                                      'int8',
        'PuaMode':                                              'category',
        'SMode':                                                'float16',
        'IeVerIdentifier':                                      'float16',
        'SmartScreen':                                          'category',
        'Firewall':                                             'float16',
        'UacLuaenable':                                         'float32',
        'Census_MDC2FormFactor':                                'category',
        'Census_DeviceFamily':                                  'category',
        'Census_OEMNameIdentifier':                             'float16',
        'Census_OEMModelIdentifier':                            'float32',
        'Census_ProcessorCoreCount':                            'float16',
        'Census_ProcessorManufacturerIdentifier':               'float16',
        'Census_ProcessorModelIdentifier':                      'float16',
        'Census_ProcessorClass':                                'category',
        'Census_PrimaryDiskTotalCapacity':                      'float32',
        'Census_PrimaryDiskTypeName':                           'category',
        'Census_SystemVolumeTotalCapacity':                     'float32',
        'Census_HasOpticalDiskDrive':                           'int8',
        'Census_TotalPhysicalRAM':                              'float32',
        'Census_ChassisTypeName':                               'category',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float16',
        'Census_InternalPrimaryDisplayResolutionHorizontal':    'float16',
        'Census_InternalPrimaryDisplayResolutionVertical':      'float16',
        'Census_PowerPlatformRoleName':                         'category',
        'Census_InternalBatteryType':                           'category',
        'Census_InternalBatteryNumberOfCharges':                'float32',
        'Census_OSVersion':                                     'category',
        'Census_OSArchitecture':                                'category',
        'Census_OSBranch':                                      'category',
        'Census_OSBuildNumber':                                 'int16',
        'Census_OSBuildRevision':                               'int32',
        'Census_OSEdition':                                     'category',
        'Census_OSSkuName':                                     'category',
        'Census_OSInstallTypeName':                             'category',
        'Census_OSInstallLanguageIdentifier':                   'float16',
        'Census_OSUILocaleIdentifier':                          'int16',
        'Census_OSWUAutoUpdateOptionsName':                     'category',
        'Census_IsPortableOperatingSystem':                     'int8',
        'Census_GenuineStateName':                              'category',
        'Census_ActivationChannel':                             'category',
        'Census_IsFlightingInternal':                           'float16',
        'Census_IsFlightsDisabled':                             'float16',
        'Census_FlightRing':                                    'category',
        'Census_ThresholdOptIn':                                'float16',
        'Census_FirmwareManufacturerIdentifier':                'float16',
        'Census_FirmwareVersionIdentifier':                     'float32',
        'Census_IsSecureBootEnabled':                           'int8',
        'Census_IsWIMBootEnabled':                              'float16',
        'Census_IsVirtualDevice':                               'float16',
        'Census_IsTouchEnabled':                                'int8',
        'Census_IsPenCapable':                                  'int8',
        'Census_IsAlwaysOnAlwaysConnectedCapable':              'float16',
        'Wdft_IsGamer':                                         'float16',
        'Wdft_RegionIdentifier':                                'float16',
        'HasDetections':                                        'int8'
        }

numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_columns = [c for c,v in dtypes.items() if v in numerics]
categorical_columns = [c for c,v in dtypes.items() if v not in numerics]
retained_columns = numerical_columns + categorical_columns

true_numerical_columns = [
    'Census_ProcessorCoreCount',
    'Census_PrimaryDiskTotalCapacity',
    'Census_SystemVolumeTotalCapacity',
    'Census_TotalPhysicalRAM',
    'Census_InternalPrimaryDiagonalDisplaySizeInInches',
    'Census_InternalPrimaryDisplayResolutionHorizontal',
    'Census_InternalPrimaryDisplayResolutionVertical',
    'Census_InternalBatteryNumberOfCharges'
]

def get_cat_vars(train, numeric_flag='TargetEncoded'):
    binary_variables = [c for c in train.columns if train[c].nunique() == 2]
    flagged_numerics = [c for c in train.columns if numeric_flag in c]

    categorical_columns = [c for c in train.columns
                           if (c not in true_numerical_columns) \
                           & (c not in binary_variables) \
                           & (c not in flagged_numerics)]
    variables = {
        'categorical_columns': categorical_columns,
        'binary_variables': binary_variables,
        'true_numerical_columns': true_numerical_columns + flagged_numerics
    }
    return variables

class Indexer():
    def __init__(self, categorical_columns=None, leave_out=['MachineIdentifier']):
        self.categorical_columns = categorical_columns
        self.leave_out = leave_out

    def fit(self, train, y=None):
        self.indexer_ = {}
        for col in tqdm(self.categorical_columns):
            if col in self.leave_out: continue
            _, self.indexer_[col] = pd.factorize(train[col])
        return self

    def transform(self, df, y=None):
        for col in tqdm(self.categorical_columns):
            if col in self.leave_out: continue
            df[col] = self.indexer_[col].get_indexer(df[col])
        return df

class HierarchySplitter():
    """ Assumes the number of splits is consistent
        between rows (does not work for OSBuildLab)
    """
    def __init__(self, colnames, splitter='.', drop=True):
        self.colnames = colnames
        self.splitter = splitter
        self.drop = drop

    def fit(self, X, y=None):
        self._col_params = {}
        for col in self.colnames:
            col_params = {}
            nunq = len(X.loc[0, col].split(self.splitter))
            col_params["nunq"] = nunq
            self._col_params[col] = col_params
        return self

    def transform(self, X, y=None):
        for col in self.colnames:
            #print('Transforming column {}'.format(col))
            nunq = self._col_params[col]["nunq"]
            X = self._split_col(X, col, nunq)
        return X

    def _split_col(self, X, col, nunq):
        for i in range(nunq):
            new_name = col + str(i)
            X[new_name] = X[col] \
                .apply(lambda x: str(x).split(self.splitter)[i])
            try:
                X[new_name] = X[new_name].astype(int)
            except:
                X[new_name] = X[new_name].astype(str)
            # if X[new_name].nunique() == 1:
            #     X = X.drop(new_name, axis=1)
        if self.drop:
            X = X.drop(col, axis=1)
        return X


class TargetEncoder():
    def __init__(self, columns, label='HasDetections', subsample=1,
            add_noise=False, noise_sd=0.15, drop=True):
        self.columns = columns
        self.label = label
        self.subsample = subsample
        self.add_noise = add_noise
        self.noise_sd = noise_sd
        self.drop = drop

    def fit(self, X, y):
        dat = pd.concat([X, y], axis=1)
        self.col_mappings_ = {}
        for col in tqdm(self.columns):
            d = dat[[col, self.label]]
            col_dict = {}
            if self.subsample < 1:
                d = d.sample(frac=self.subsample, replace=True, random_state=123)
            prior = d[self.label].mean()
            col_dict['prior'] = prior
            grpd = d.groupby(col) \
                .agg({col: 'size', self.label: 'mean'})
            grpd['weight'] = weighter(grpd[col])

            if self.add_noise:
                noise = np.random.uniform(0, self.noise_sd, size=len(grpd['weight']))
            else:
                noise = 0
            grpd['TargetEncoded'+col] = (grpd['weight']*grpd[self.label] + (1-grpd['weight'])*prior) + noise
            grpd = grpd.drop(col, axis=1)
            col_dict['encoded_table'] = grpd[['TargetEncoded'+col]].reset_index()
            self.col_mappings_[col] = col_dict
        return self

    def transform(self, X, y=None):
        for col in tqdm(self.columns):
            prior = self.col_mappings_[col]['prior']
            grpd = self.col_mappings_[col]['encoded_table']
            X = X.merge(grpd, on=col, how='left')
            if self.add_noise:
                na_cats = np.sum(X['TargetEncoded'+col].isna())
                if na_cats > 0:
                    noise = np.random.uniform(0, self.noise_sd, size=na_cats)
                else:
                    noise = 0
            else:
                noise = 0

            X.loc[X['TargetEncoded'+col].isna(), 'TargetEncoded'+col] = prior + noise
            if self.drop:
                X = X.drop(col, axis=1)
        return X




class FrequencyEncoder():
    pass

class ProbabilityEncoder():
    pass



def effect_encoder(dat, col, label='HasDetections'):
    uniq = dat[col].unique()
    prior = dat[label].mean()
    d = dat[[col, label]]
    grpd = d.groupby(col) \
        .agg({col: 'size', label: 'mean'})
    grpd['weight'] = grpd[col].apply(lambda x: weighter(x))
    grpd['Encoded'+col] = grpd['weight']*grpd[label] + (1-grpd['weight'])*prior + np.random.normal(0, 0.1, 1)
    #grpd[col] = grpd.index
    #effect_table = grpd[['Encoded'+col, col]]
    return grpd['Encoded'+col]

def weighter(size, k=20, f=2000):
    denom = (1 + np.exp((size-k)/f))
    return 1/denom
