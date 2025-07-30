# %% 載入套件
import pandas as pd
import numpy as np
import xgboost
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
import talib
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import scatterplotmatrix
from mlxtend.plotting import heatmap
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

# %% 資料建置2330

stock = pd.read_csv('C:\\Users\\88698\\Desktop\\新尖兵\\2330_0711.csv', encoding='big5')



stock = stock.set_index('date')

print(stock)



# %% 資料建置台股大盤

stockTW = pd.read_csv('C:\\Users\\88698\\Desktop\\新尖兵\\Y9999_0711.csv', encoding='utf-8')

stockTW = stockTW.set_index('date')

stock = pd.merge(stock, stockTW, on='date', how='left')


print(stockTW)

print(stock)

# %% 資料處理Y9999

stock['Y9999_MA5']   = talib.MA(stockTW['Y9999_close'], timeperiod=5)
stock['Y9999_MA10']  = talib.MA(stockTW['Y9999_close'], timeperiod=10)
stock['Y9999_MA20']  = talib.MA(stockTW['Y9999_close'], timeperiod=20)
stock['Y9999_MA60']  = talib.MA(stockTW['Y9999_close'], timeperiod=60)
stock['Y9999_MA120'] = talib.MA(stockTW['Y9999_close'], timeperiod=120)
stock['Y9999_MA240'] = talib.MA(stockTW['Y9999_close'], timeperiod=240)

stock['Y9999_EMA5']   = talib.EMA(stockTW['Y9999_close'], timeperiod=5)
stock['Y9999_EMA10']  = talib.EMA(stockTW['Y9999_close'], timeperiod=10)
stock['Y9999_EMA20']  = talib.EMA(stockTW['Y9999_close'], timeperiod=20)
stock['Y9999_EMA60']  = talib.EMA(stockTW['Y9999_close'], timeperiod=60)
stock['Y9999_EMA120'] = talib.EMA(stockTW['Y9999_close'], timeperiod=120)
stock['Y9999_EMA240'] = talib.EMA(stockTW['Y9999_close'], timeperiod=240)

#動能/變動率
stock['Y9999_MOM10'] = talib.MOM(stockTW['Y9999_close'], timeperiod=10)
stock['Y9999_ROC5']  = talib.ROC(stockTW['Y9999_close'], timeperiod=5)
stock['Y9999_ROC10'] = talib.ROC(stockTW['Y9999_close'], timeperiod=10)

Y9999_bb_upper, Y9999_bb_middle, Y9999_bb_lower = talib.BBANDS(stock['Y9999_close'],timeperiod=20,    nbdevup=2,     nbdevdn=2, )
stock['Y9999_Bollinger_upper']  = Y9999_bb_upper.round(2)     
stock['Y9999_Bollinger_middle'] = Y9999_bb_middle.round(2)    
stock['Y9999_Bollinger_lower']  = Y9999_bb_lower.round(2)  
stock['Y9999_Bollinger_bandwidth'] = (Y9999_bb_upper - Y9999_bb_lower).round(2)

#報酬率（含長期）
stock['Y9999_return_5']   = stockTW['Y9999_close'].pct_change(5)
stock['Y9999_return_10']  = stockTW['Y9999_close'].pct_change(10)
stock['Y9999_return_20']  = stockTW['Y9999_close'].pct_change(20)
stock['Y9999_return_60']  = stockTW['Y9999_close'].pct_change(60)
stock['Y9999_return_120'] = stockTW['Y9999_close'].pct_change(120)
stock['Y9999_return_240'] = stockTW['Y9999_close'].pct_change(240)

print(stock)

# %% 資料處理2330



# === 短中長期均線特徵 ===

stock['MA5']   = talib.MA(stock['close'], timeperiod=5).round(2)        # 5日簡單移動平均（短線）
stock['MA10']  = talib.MA(stock['close'], timeperiod=10).round(2)       # 10日簡單移動平均（短線）
stock['MA20']  = talib.MA(stock['close'], timeperiod=20).round(2)       # 一個月均線（中線）
stock['MA60']  = talib.MA(stock['close'], timeperiod=60).round(2)       # 一季均線
stock['MA120'] = talib.MA(stock['close'], timeperiod=120).round(2)      # 半年均線（長線）
stock['MA240'] = talib.MA(stock['close'], timeperiod=240).round(2)      # 一年均線（長線）

stock['WMA5']  = talib.WMA(stock['close'], timeperiod=5).round(2)       # 5日加權平均（短線）
stock['WMA20']  = talib.WMA(stock['close'], timeperiod=20).round(2)       # 5日加權平均（短線）
stock['WMA60']  = talib.WMA(stock['close'], timeperiod=60).round(2)       # 5日加權平均（短線）
stock['WMA120']  = talib.WMA(stock['close'], timeperiod=120).round(2)       # 5日加權平均（短線）

stock['EMA5']  = talib.EMA(stock['close'], timeperiod=5).round(2)       # 5日指數平均（短線）
stock['EMA10'] = talib.EMA(stock['close'], timeperiod=10).round(2)      # 10日指數平均（短線）
stock['EMA20']  = talib.EMA(stock['close'], timeperiod=20).round(2)     # 一個月EMA（中線）
stock['EMA60']  = talib.EMA(stock['close'], timeperiod=60).round(2)     # 一季EMA
stock['EMA120'] = talib.EMA(stock['close'], timeperiod=120).round(2)    # 半年EMA（長線）
stock['EMA240'] = talib.EMA(stock['close'], timeperiod=240).round(2)    # 一年EMA（長線）


# === 動能/變動率特徵 ===

stock['MOM10'] = talib.MOM(stock['close'], timeperiod=10).round(2)    # 10日動能
stock['ROC5'] = talib.ROC(stock['close'], timeperiod=5)               # 5日變動率
stock['ROC10'] = talib.ROC(stock['close'], timeperiod=10)             # 10日變動率

# 長期報酬
stock['return']      = stock['close'].pct_change()                # 日報酬率
stock['return_20']   = stock['close'].pct_change(20)              # 月報酬率
stock['return_60']   = stock['close'].pct_change(60)              # 季報酬率
stock['return_120']  = stock['close'].pct_change(120)             # 半年報酬率
stock['return_240']  = stock['close'].pct_change(240)             # 年報酬率


# === 隨機、強弱、動能類 ===

slowk, slowd = talib.STOCH(stock['high'], stock['low'], stock['close'], slowk_period=3, slowd_period=3)
stock['K'] = slowk       # KD指標K值
stock['D'] = slowd       # KD指標D值

stock['RSI5']  = talib.RSI(stock['close'], timeperiod=5)        # 5日RSI（短線強弱）
stock['RSI14'] = talib.RSI(stock['close'], timeperiod=14)       # 14日RSI（短線強弱）


# 威廉
stock['willan'] = talib.WILLR(stock['high'], stock['low'], stock['close'], timeperiod=14)   # 威廉指標

# 商品通道指標
stock['CCI'] = talib.CCI(stock['high'], stock['low'], stock['close'], timeperiod=14)        # CCI


# === MACD 指標相關    ===

macd, signal, hist = talib.MACD(stock['close'], fastperiod=12, slowperiod=26, signalperiod=9)
stock['macd'] = macd         # MACD值
stock['signal'] = signal     # 訊號線
stock['hist'] = hist         # 柱狀圖


# === 成交量/能量/趨勢 ===

stock['OBV'] = talib.OBV(stock['close'], stock['volume'])         # OBV能量潮指標
stock['AD']  = talib.AD(stock['high'], stock['low'], stock['close'], stock['volume']) # 累積/分配線
stock['ATR'] = talib.ATR(stock['high'], stock['low'], stock['close'], timeperiod=14)  # 波動度（平均真實區間）
stock['ADX14'] = talib.ADX(stock['high'], stock['low'], stock['close'], timeperiod=14) # 趨勢強度

# === 均量/波動/價差類 ===

bb_upper, bb_middle, bb_lower = talib.BBANDS(stock['close'],timeperiod=20,    nbdevup=2,     nbdevdn=2, )
stock['Bollinger_upper']  = bb_upper.round(2)     
stock['Bollinger_middle'] = bb_middle.round(2)    
stock['Bollinger_lower']  = bb_lower.round(2)  
stock['Bollinger_bandwidth'] = (bb_upper - bb_lower).round(2)


stock['VOL_MA5']   = stock['volume'].rolling(5).mean()           # 5日均量
stock['VOL_MA10']  = stock['volume'].rolling(10).mean()          # 10日均量
stock['VOL_MA20']  = stock['volume'].rolling(20).mean()          # 月均量
stock['VOL_MA60']  = stock['volume'].rolling(60).mean()          # 季均量
stock['VOL_MA120'] = stock['volume'].rolling(120).mean()         # 半年均量
stock['VOL_MA240'] = stock['volume'].rolling(240).mean()         # 年均量

stock['volatility_5']    = stock['close'].rolling(5).std()        # 5日收盤價波動度
stock['volatility_20']   = stock['close'].rolling(20).std()       # 月波動度
stock['volatility_60']   = stock['close'].rolling(60).std()       # 季波動度
stock['volatility_120']  = stock['close'].rolling(120).std()      # 半年波動度
stock['volatility_240']  = stock['close'].rolling(240).std()      # 年波動度

stock['high_low_spread']   = stock['high'] - stock['low']         # 日內高低差
stock['close_open_spread'] = stock['close'] - stock['open']       # 日內開收差

stock['gap_open'] =(stock['open'] - stock['close'].shift(1)) / stock['close'].shift(1)
stock['is_gap_up'] = ((stock['open'] > stock['close'].shift(1)*1.03)).astype(int)    # 跳空上漲3%
stock['is_gap_down'] = ((stock['open'] < stock['close'].shift(1)*0.97)).astype(int)  # 跳空下跌3%

stock['is_crash_day_3'] = (stock['return'] < -0.03).astype(int)    # 單日下跌超過3%
stock['is_crash_day_5'] = (stock['return'] < -0.05).astype(int)    # 單日下跌超過5%
stock['is_surge_day_3'] = (stock['return'] > 0.03).astype(int)     # 單日上漲超過3%
stock['is_surge_day_5'] = (stock['return'] > 0.05).astype(int)


stock['volume_change'] = stock['volume'] / stock['volume'].shift(1)  # 價量變化
stock['rel_return'] = stock['return'] - stock['Y9999_return_5']  # 個股vs大盤強弱

# --- 連續大跌/大漲日 ---
stock['consecutive_down3'] = (stock['return'] < 0).rolling(3).sum().eq(3).astype(int)   # 連續下跌3天
stock['consecutive_up3'] = (stock['return'] > 0).rolling(3).sum().eq(3).astype(int)     # 連續上漲3天

# --- 異常成交量 ---
stock['is_volume_burst'] = (stock['volume'] > stock['volume'].rolling(20).mean()*2).astype(int)    # 爆量
stock['is_volume_shrink'] = (stock['volume'] < stock['volume'].rolling(20).mean()*0.5).astype(int) # 急縮

# --- 放量下跌/放量上漲 ---
stock['is_heavy_volume_down'] = ((stock['return'] < -0.02) & (stock['volume'] > stock['volume'].rolling(20).mean()*2)).astype(int)
stock['is_heavy_volume_up'] = ((stock['return'] > 0.02) & (stock['volume'] > stock['volume'].rolling(20).mean()*2)).astype(int)

# --- 波動率異常 ---
stock['is_volatility_burst'] = (stock['volatility_20'] > stock['volatility_20'].rolling(100).mean()*2).astype(int)

# --- OBV連跌/連升 ---
stock['obv_down_trend5'] = (stock['OBV'].diff().rolling(5).sum() < 0).astype(int)
stock['obv_up_trend5']   = (stock['OBV'].diff().rolling(5).sum() > 0).astype(int)

# --- 長黑K/大陰線 ---
stock['is_long_black'] = ((stock['open']-stock['close'])/stock['close'] > 0.05).astype(int)   # 開高收低5%
# --- 鎚頭K棒 ---
stock['is_hammer'] = ((stock['low'] < np.minimum(stock['open'], stock['close'])*0.98) & (stock['close'] > stock['open'])).astype(int)

# --- 低量突破 ---
stock['is_low_vol_breakout'] = ((stock['close'] > stock['close'].rolling(20).max()) & (stock['volume'] < stock['volume'].rolling(20).mean()*0.7)).astype(int)

# --- 成交值／市值比異常 ---
stock['turnover_ratio'] = stock['volume'] / stock['Y9999_shares_outstanding']
stock['is_turnover_burst'] = (stock['turnover_ratio'] > stock['turnover_ratio'].rolling(60).mean()*2).astype(int)

# === 指標交叉 ===

stock['golden_cross_5_20'] = ((stock['MA5'].shift(1) < stock['MA20'].shift(1)) &(stock['MA5'] > stock['MA20'])).astype(int)
stock['death_cross_5_20'] = ((stock['MA5'].shift(1) > stock['MA20'].shift(1)) &(stock['MA5'] < stock['MA20'])).astype(int)

stock['macd_golden_cross'] = ((stock['macd'].shift(1) < stock['signal'].shift(1)) & (stock['macd'] > stock['signal'])).astype(int)
stock['macd_hist_zero_cross'] = ((stock['hist'].shift(1) < 0) & (stock['hist'] > 0)).astype(int)

stock['kd_golden_cross'] = ((stock['K'].shift(1) < stock['D'].shift(1)) & (stock['K'] > stock['D'])).astype(int)
stock['kd_death_cross'] = ((stock['K'].shift(1) > stock['D'].shift(1)) & (stock['K'] < stock['D'])).astype(int)

stock['rsi5_cross_50_up'] = ((stock['RSI5'].shift(1) < 50) & (stock['RSI5'] >= 50)).astype(int)
stock['rsi5_cross_70_up'] = ((stock['RSI5'].shift(1) < 70) & (stock['RSI5'] >= 70)).astype(int)

stock['close_cross_boll_upper'] = ((stock['close'].shift(1) < stock['Bollinger_upper'].shift(1)) & (stock['close'] > stock['Bollinger_upper'])).astype(int)
stock['close_cross_boll_lower'] = ((stock['close'].shift(1) > stock['Bollinger_lower'].shift(1)) & (stock['close'] < stock['Bollinger_lower'])).astype(int)

stock['adx_cross_20_up'] = ((stock['ADX14'].shift(1) < 20) & (stock['ADX14'] >= 20)).astype(int)

stock['strong_cross'] = ((stock['golden_cross_5_20'] == 1) &(stock['macd_golden_cross'] == 1) &(stock['kd_golden_cross'] == 1)).astype(int)

print(stock.columns)


# %% 資料篩選

# region 建立y數據

stock['Close_1D'] = stock['close'].shift(-1)  # 往上移一格，明日Close
stock['Close_2D'] = stock['close'].shift(-2)  # 往上移一格，明日Close
stock['Close_3D'] = stock['close'].shift(-3)  # 往上移一格，明日Close
stock0711 = stock.loc['20250711':]
stock['1Dspread'] = stock['Close_1D'] - stock['close']
stock['2Dspread'] = stock['Close_2D'] - stock['close']
stock['3Dspread'] = stock['Close_3D'] - stock['close']

stock['1Dspread_label'] = np.where(stock['1Dspread'] > 25 , 1 ,np.where(stock['1Dspread'] < -25 , -1 ,0))
stock['2Dspread_label'] = np.where(stock['2Dspread'] > 25 , 1 ,np.where(stock['2Dspread'] < -25 , -1 ,0))
stock['3Dspread_label'] = np.where(stock['3Dspread'] > 25 , 1 ,np.where(stock['3Dspread'] < -25 , -1 ,0))


train = stock.loc['20000101':'20250101'].dropna()



labelD1 = ('1Dspread')
labelD2 = ('2Dspread')
labelD3 = ('3Dspread')

print(stock0711)
# endregion

# region 建立特徵

feature = (
    # ==== 價量基本面 ====
    'open', 'high', 'low', 'close', 'volume',
    'Y9999_open', 'Y9999_high', 'Y9999_low', 'Y9999_close', 'Y9999_volume', 'Y9999_shares_outstanding',

    # ==== 均線類 ====
    'MA5',  'MA20', 'MA60',
    'WMA5', 'WMA20', 'WMA60', 
    'EMA5',  'EMA20', 'EMA60', 
    'Y9999_MA5',  'Y9999_MA20', 'Y9999_MA60',
    'Y9999_EMA5',  'Y9999_EMA20', 'Y9999_EMA60',

    # ==== 動能/強弱/隨機 ====
    'K', 'D', 'RSI5','CCI', 'willan',
    'MOM10', 'Y9999_MOM10',
    # ==== 變動率 ====
    'ROC5', 'Y9999_ROC5', 

    # ==== MACD相關 ====
    'macd',

    # ==== 趨勢/能量 ====
    'ADX14',  'AD', 'ATR',

    # ==== 布林通道/波動度 ====
    'Bollinger_bandwidth',
    'Y9999_Bollinger_bandwidth',
    'volatility_5', 'volatility_20', 'volatility_60',

    # ==== 價差/缺口 ====
    'high_low_spread', 'close_open_spread',

    # ==== 價量異常事件/極端行情 ====
    'is_gap_up',           # 跳空上漲3%
    'is_gap_down',         # 跳空下跌3%
    'is_crash_day_3',      # 單日下跌超過3%
    'is_crash_day_5',      # 單日下跌超過5%
    'is_surge_day_3',      # 單日上漲超過3%
    'is_surge_day_5',      # 單日上漲超過5%
    'consecutive_down3',   # 連續下跌3天
    'consecutive_up3',     # 連續上漲3天
    'is_volume_burst',     # 爆量
    'is_volume_shrink',    # 急縮
    'is_heavy_volume_down',# 放量下跌
    'is_heavy_volume_up',  # 放量上漲
    'is_volatility_burst', # 波動率異常
    'obv_down_trend5',     # OBV連跌5日
    'obv_up_trend5',       # OBV連升5日
    'is_long_black',       # 長黑K
    'is_hammer',           # 鎚頭K
    'is_low_vol_breakout', # 低量突破
    'turnover_ratio',      # 成交值/市值
    'is_turnover_burst',   # 成交值異常

    # ==== 成交量均量 ====
    'VOL_MA5',  'VOL_MA20', 'VOL_MA60', 
    'volume_change',

    # ==== 報酬率 ====
    'return', 
    'Y9999_return_5', 
    'rel_return',

    # ==== 指標交叉 ====
    'golden_cross_5_20', 'death_cross_5_20',
    'macd_golden_cross', 'macd_hist_zero_cross',
    'kd_golden_cross', 'kd_death_cross',
    'rsi5_cross_50_up', 'rsi5_cross_70_up',
    'close_cross_boll_upper', 'close_cross_boll_lower',
    'adx_cross_20_up',
    'strong_cross'
)

# endregion
"""
# region part1 y資料建置
p1labelD1=('1Dspread_label')
p1labelD2=('2Dspread_label')
p1labelD3=('3Dspread_label')

p1_y_trainD1 = train.loc[:, p1labelD1].values
p1_y_trainD2 = train.loc[:, p1labelD1].values
p1_y_trainD3 = train.loc[:, p1labelD1].values

p1_y_testD1 = stock.loc['20250101':'20250630', p1labelD1].values
p1_y_testD2 = stock.loc['20250101':'20250630', p1labelD1].values
p1_y_testD3 = stock.loc['20250101':'20250630', p1labelD1].values
# endregion
"""
# region y資料建置
y_trainD1 = train.loc[:, labelD1].values
y_trainD2 = train.loc[:, labelD2].values
y_trainD3 = train.loc[:, labelD3].values

y_testD1 = stock.loc['20250101':'20250630', labelD1].values
y_testD2 = stock.loc['20250101':'20250630', labelD2].values
y_testD3 = stock.loc['20250101':'20250630', labelD3].values

sc_yD1 = StandardScaler()
sc_yD2 = StandardScaler()
sc_yD3 = StandardScaler()

y_trainD1_std = sc_yD1.fit_transform(y_trainD1.reshape(-1,1)).flatten()
y_testD1_std = sc_yD1.transform(y_testD1.reshape(-1,1)).flatten()


y_trainD2_std = sc_yD2.fit_transform(y_trainD2.reshape(-1,1)).flatten()
y_testD2_std = sc_yD2.transform(y_testD2.reshape(-1,1)).flatten()

y_trainD3_std = sc_yD3.fit_transform(y_trainD3.reshape(-1,1)).flatten()
y_testD3_std = sc_yD3.transform(y_testD3.reshape(-1,1)).flatten()

# endregion

# region x特徵資料建置
pred_D1 = (stock0711.loc[:,feature]).values
pred_D2 = (stock0711.loc[:,feature]).values
pred_D3 = (stock0711.loc[:,feature]).values

x_trainD1 = (train.loc[:,feature]).values
x_trainD2 = (train.loc[:,feature]).values
x_trainD3 = (train.loc[:,feature]).values

x_testD1 = (stock.loc['20250101':'20250630', feature].dropna()).values
x_testD2 = (stock.loc['20250101':'20250630', feature].dropna()).values
x_testD3 = (stock.loc['20250101':'20250630', feature].dropna()).values

sc_xD1 = StandardScaler()
sc_xD2 = StandardScaler()
sc_xD3 = StandardScaler()

x_trainD1_std = sc_xD1.fit_transform(x_trainD1)
x_testD1_std = sc_xD1.transform(x_testD1)
predD1_std = sc_xD1.transform(pred_D1)


x_trainD2_std = sc_xD2.fit_transform(x_trainD2)
x_testD2_std = sc_xD2.transform(x_testD2)
predD2_std = sc_xD2.transform(pred_D2)


x_trainD3_std = sc_xD3.fit_transform(x_trainD3)
x_testD3_std = sc_xD3.transform(x_testD3)
predD3_std = sc_xD3.transform(pred_D3)
# endregion



# %% 模型建置D1

xgbD1_STD = XGBRegressor(

    n_estimators=1000,          # 樹數量
    learning_rate=0.02,        # 學習率
    max_depth=14,               # 樹深
    subsample=0.8,             # 資料取樣比例
    colsample_bytree=0.8,      # 特徵取樣比例
    random_state=1,            # 隨機種子
    tree_method='gpu_hist',  
    n_jobs=-6
    
)


print("STD")



xgbD1_STD.fit(x_trainD1_std,y_trainD1_std)
y_xgbD1_test_pred = xgbD1_STD.predict(x_testD1_std)
print(f'MSE:{mean_squared_error(y_testD1_std,y_xgbD1_test_pred)}')
print(f'R2:{r2_score(y_testD1_std,y_xgbD1_test_pred)}')

print('-'*50)

xgbD1 = XGBRegressor(

    n_estimators=1000,          # 樹數量
    learning_rate=0.02,        # 學習率
    max_depth=7,               # 樹深
    subsample=0.8,             # 資料取樣比例
    colsample_bytree=0.8,      # 特徵取樣比例
    random_state=1,            # 隨機種子
    tree_method='gpu_hist',  
    n_jobs=-6
    
)

print('NORMAL')


xgbD1.fit(x_trainD1,y_trainD1)
y0_xgbD1_test_pred = xgbD1.predict(x_testD1)
print(f'MSE:{mean_squared_error(y_testD1,y0_xgbD1_test_pred)}')
print(f'R2:{r2_score(y_testD1,y0_xgbD1_test_pred)}')


# %% 模型建置D2

xgbD2_STD = XGBRegressor(

    n_estimators=1000,          # 樹數量
    learning_rate=0.02,        # 學習率
    max_depth=3,               # 樹深
    subsample=0.8,             # 資料取樣比例
    colsample_bytree=0.8,      # 特徵取樣比例
    reg_alpha=0.3,             # L1正則化
    reg_lambda=0.1,            # L2正則化
    random_state=1,            # 隨機種子
    tree_method='gpu_hist',  
    n_jobs=-6
    
)


print("STD")



xgbD2_STD.fit(x_trainD2_std,y_trainD2_std)
y_xgbD2_test_pred = xgbD2_STD.predict(x_testD2_std)
print(f'MSE:{mean_squared_error(y_testD2_std,y_xgbD2_test_pred)}')
print(f'R2:{r2_score(y_testD2_std,y_xgbD2_test_pred)}')

print('-'*50)

xgbD2 = XGBRegressor(

    n_estimators=1000,          # 樹數量
    learning_rate=0.02,        # 學習率
    max_depth=3,               # 樹深
    subsample=0.8,             # 資料取樣比例
    colsample_bytree=0.8,      # 特徵取樣比例
    reg_alpha=0.3,             # L1正則化
    reg_lambda=0.1,            # L2正則化
    random_state=1,            # 隨機種子
    tree_method='gpu_hist',  
    n_jobs=-6
    
)

print('NORMAL')


xgbD2.fit(x_trainD2,y_trainD2)
y0_xgbD2_test_pred = xgbD2.predict(x_testD2)
print(f'MSE:{mean_squared_error(y_testD2,y0_xgbD2_test_pred)}')
print(f'R2:{r2_score(y_testD2,y0_xgbD2_test_pred)}')





# %% 模型建置D3


xgbD3_STD = XGBRegressor(

    n_estimators=1000,          # 樹數量
    learning_rate=0.02,        # 學習率
    max_depth=3,               # 樹深
    subsample=0.8,             # 資料取樣比例
    colsample_bytree=0.8,      # 特徵取樣比例
    reg_alpha=0.3,             # L1正則化
    reg_lambda=0.1,            # L2正則化
    random_state=1,            # 隨機種子
    tree_method='gpu_hist',  
    n_jobs=-6
    
)


print("STD")



xgbD3_STD.fit(x_trainD3_std,y_trainD3_std)
y_xgbD3_test_pred = xgbD3_STD.predict(x_testD3_std)
print(f'MSE:{mean_squared_error(y_testD3_std,y_xgbD3_test_pred)}')
print(f'R2:{r2_score(y_testD3_std,y_xgbD3_test_pred)}')

print('-'*50)

xgbD3 = XGBRegressor(

    n_estimators=1000,          # 樹數量
    learning_rate=0.02,        # 學習率
    max_depth=3,               # 樹深
    subsample=0.8,             # 資料取樣比例
    colsample_bytree=0.8,      # 特徵取樣比例
    reg_alpha=0.3,             # L1正則化
    reg_lambda=0.1,            # L2正則化
    random_state=1,            # 隨機種子
    tree_method='gpu_hist',  
    n_jobs=-6
    
)

print('NORMAL')


xgbD3.fit(x_trainD3,y_trainD3)
y0_xgbD3_test_pred = xgbD3.predict(x_testD3)
print(f'MSE:{mean_squared_error(y_testD3,y0_xgbD3_test_pred)}')
print(f'R2:{r2_score(y_testD3,y0_xgbD3_test_pred)}')



# %% 數據權重

importances = xgbD1_STD.feature_importances_
indices = np.argsort(importances)[::-1]
importances_list = []
for i in range(x_trainD1.shape[1]):
  print(f"{i+1:>2}.{feature[indices[i]]:<10} Importance：{importances[indices[i]]:.4f}")
  importances_list.append(feature[indices[i]])

print(importances_list)
plt.figure(figsize=(10,6))
plt.title("Feature Importance by xgb")
plt.bar(range(x_trainD1.shape[1]), importances[indices], align='center')
plt.xticks(range(x_trainD1.shape[1]), [feature[i] for i in indices], rotation=45)
plt.ylabel("Importance")
plt.tight_layout()
plt.show()




# %%  參數最佳化
model_dict = {
    'D1': (xgbD1_STD, x_trainD1_std, y_trainD1_std),
    #'D2': (xgbD2_STD, x_trainD2_std, y_trainD2_std),
    #'D3': (xgbD3_STD, x_trainD3_std, y_trainD3_std)
}


param_grid = {
    'n_estimators': [300, 600, 1000],
    'learning_rate': [0.03, 0.02, 0.01],
    'max_depth': [3, 5, 7],
    'reg_alpha': [0,0.2, 0.5],
    'reg_lambda': [0.1,0.5, 1.0],

}

for name, (xgb, x_train, y_train) in model_dict.items():
    print(f"\nGridSearch for {name}")
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=1,
        n_jobs=3
    )
    grid_search.fit(x_train, y_train)
    print(f"Best params for {name}:", grid_search.best_params_)
    print(f"Best MSE for {name}: {-grid_search.best_score_:.6f}")

# %%  單模型跑時間
import time
start = time.time()
xgbD1_STD.fit(x_trainD1_std, y_trainD1_std)
print(f"單次訓練耗時：{time.time()-start:.2f} 秒")

# %% 圖表


model = {
      'xgb' :y_xgbD1_test_pred,
      }

for name , pred in model.items():
  plt.figure(figsize=(10,6))
  plt.plot(y_testD1_std, label='Actual', marker='o')
  plt.plot(pred, label='Predicted', marker='x')
  plt.title(f'({name})Actual vs Predicted Values')
  plt.xlabel('Sample Index')
  plt.ylabel('Target Value')
  plt.legend()
  plt.grid(True)
  plt.show()

# %% 輸出測試結果
test_yD1_std = xgbD1_STD.predict(x_testD1_std)
test_yD2_std = xgbD2_STD.predict(x_testD1_std)
test_yD3_std = xgbD3_STD.predict(x_testD1_std)


test_yD1_std = sc_yD1.inverse_transform(test_yD1_std.reshape(-1, 1)).flatten()
test_yD2_std = sc_yD2.inverse_transform(test_yD2_std.reshape(-1, 1)).flatten()
test_yD3_std = sc_yD3.inverse_transform(test_yD3_std.reshape(-1, 1)).flatten()

test_yD1 = xgbD1.predict(x_testD1)
test_yD2 = xgbD1.predict(x_testD1)
test_yD3 = xgbD1.predict(x_testD1)

test_close = stock.loc['20250101':'20250630','close'].values 
test_1Dclose = stock.loc['20250101':'20250630','Close_1D'].values

pred_Close_1DSTD = test_close+test_yD1_std

pred_Close_1D:  test_close+test_yD1

plt.figure(figsize=(10, 6))
plt.plot(test_1Dclose, label="Actual",   marker="o")
plt.plot(pred_Close_1DSTD,   label="Predicted", marker="x")
plt.title(f"D1 Actual vs Predicted")
plt.xlabel("Sample Index")
plt.ylabel("Target Value")
plt.legend()
plt.grid(True)
plt.show()



print(f'誤差:{abs(test_1Dclose-pred_Close_1DSTD).sum()}')

# %% 輸出預測結果
pred_yD1_std = xgbD1_STD.predict(predD1_std)
pred_yD2_std = xgbD2_STD.predict(predD2_std)
pred_yD3_std = xgbD3_STD.predict(predD3_std)


pred_yD1_std = sc_yD1.inverse_transform(pred_yD1_std.reshape(-1, 1)).flatten()
pred_yD2_std = sc_yD2.inverse_transform(pred_yD2_std.reshape(-1, 1)).flatten()
pred_yD3_std = sc_yD3.inverse_transform(pred_yD3_std.reshape(-1, 1)).flatten()

pred_yD1 = xgbD1.predict(pred_D1)
pred_yD2 = xgbD1.predict(pred_D2)
pred_yD3 = xgbD1.predict(pred_D3)

true_close = stock0711['close']
result_STD = pd.DataFrame({
    'date': stock0711.index,       
    'pred_Close_1D': true_close+pred_yD1_std,
    'pred_Close_2D': true_close+pred_yD2_std,
    'pred_Close_3D': true_close+pred_yD3_std,
})
print("STD")
print(result_STD)


result = pd.DataFrame({
    'date': stock0711.index,       
    'pred_Close_1D': true_close+pred_yD1,
    'pred_Close_2D': true_close+pred_yD2,
    'pred_Close_3D': true_close+pred_yD3,
})
print("normal")
print(result)

# %% 暫存參數
D1feature = (
    # ==== 價量基本面 ====
    'open', 'high', 'low', 'close', 'volume',
    'Y9999_open', 'Y9999_high', 'Y9999_low', 'Y9999_close', 'Y9999_volume', 'Y9999_shares_outstanding',

    # ==== 均線類 ====
    'MA5', 'MA10', 'MA20', 'MA60', 'MA120', 'MA240',
    'WMA5', 'WMA20', 'WMA60', 'WMA120',
    'EMA5', 'EMA10', 'EMA20', 'EMA60', 'EMA120', 'EMA240',
    'Y9999_MA5', 'Y9999_MA10', 'Y9999_MA20', 'Y9999_MA60', 'Y9999_MA120', 'Y9999_MA240',
    'Y9999_EMA5', 'Y9999_EMA10', 'Y9999_EMA20', 'Y9999_EMA60', 'Y9999_EMA120', 'Y9999_EMA240',

    # ==== 動能/強弱/隨機 ====
    'K', 'D', 'RSI5', 'RSI14', 'CCI', 'willan',
    'MOM10', 'Y9999_MOM10',

    # ==== 變動率 ====
    'ROC5', 'ROC10', 'Y9999_ROC5', 'Y9999_ROC10',

    # ==== MACD相關 ====
    'macd', 'signal', 'hist',

    # ==== 趨勢/能量 ====
    'ADX14', 'OBV', 'AD', 'ATR',

    # ==== 布林通道/波動度 ====
    'Bollinger_upper', 'Bollinger_middle', 'Bollinger_lower', 'Bollinger_bandwidth',
    'Y9999_Bollinger_upper', 'Y9999_Bollinger_middle', 'Y9999_Bollinger_lower', 'Y9999_Bollinger_bandwidth',
    'volatility_5', 'volatility_20', 'volatility_60', 'volatility_120', 'volatility_240',

    # ==== 價差/缺口 ====
    'high_low_spread', 'close_open_spread', 'gap_open',

    # ==== 價量異常事件/極端行情 ====
    'is_gap_up',           # 跳空上漲3%
    'is_gap_down',         # 跳空下跌3%
    'is_crash_day_3',      # 單日下跌超過3%
    'is_crash_day_5',      # 單日下跌超過5%
    'is_surge_day_3',      # 單日上漲超過3%
    'is_surge_day_5',      # 單日上漲超過5%
    'consecutive_down3',   # 連續下跌3天
    'consecutive_up3',     # 連續上漲3天
    'is_volume_burst',     # 爆量
    'is_volume_shrink',    # 急縮
    'is_heavy_volume_down',# 放量下跌
    'is_heavy_volume_up',  # 放量上漲
    'is_volatility_burst', # 波動率異常
    'obv_down_trend5',     # OBV連跌5日
    'obv_up_trend5',       # OBV連升5日
    'is_long_black',       # 長黑K
    'is_hammer',           # 鎚頭K
    'is_low_vol_breakout', # 低量突破
    'turnover_ratio',      # 成交值/市值
    'is_turnover_burst',   # 成交值異常

    # ==== 成交量均量 ====
    'VOL_MA5', 'VOL_MA10', 'VOL_MA20', 'VOL_MA60', 'VOL_MA120', 'VOL_MA240',
    'volume_change',

    # ==== 報酬率 ====
    'return', 'return_20', 'return_60', 'return_120', 'return_240',
    'Y9999_return_5', 'Y9999_return_10', 'Y9999_return_20', 'Y9999_return_60', 'Y9999_return_120', 'Y9999_return_240',
    'rel_return',

    # ==== 指標交叉 ====
    'golden_cross_5_20', 'death_cross_5_20',
    'macd_golden_cross', 'macd_hist_zero_cross',
    'kd_golden_cross', 'kd_death_cross',
    'rsi5_cross_50_up', 'rsi5_cross_70_up',
    'close_cross_boll_upper', 'close_cross_boll_lower',
    'adx_cross_20_up',
    'strong_cross'
)
