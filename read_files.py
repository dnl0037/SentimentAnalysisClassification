import pandas as pd

file_path = "reviews_train.tsv"
try:
    df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv(file_path, sep='\t', encoding='latin1')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, sep='\t', encoding='iso-8859-1')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, sep='\t', encoding='cp1252')

# Muestra las primeras filas del DataFrame
print(df.head())
