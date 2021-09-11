import pandas as pd
import numpy as np
df = pd.read_csv("atropine_test_100.csv")
i = 0
while True:
    if 60*i >= len(df):
        break
    df[60*i:60*(i+1)].to_csv(f"public_dat/{i}.csv")
    i += 1