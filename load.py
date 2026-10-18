import sys, pandas as pd
src, out = sys.argv[1], sys.argv[2]
sheets = pd.read_excel(src, sheet_name=None, dtype={'Invoice': str, 'InvoiceNo': str, 'StockCode': str})
frames = []
for name, df in sheets.items():
    print("SHEET", repr(name), df.shape, list(df.columns))
    print(df.dtypes.to_string())
    df = df.copy()
    df['_sheet'] = name
    df['_row_in_sheet'] = range(2, len(df) + 2)  # Excel row number
    frames.append(df)
all_ = pd.concat(frames, ignore_index=True)
all_.to_pickle(out)
print("TOTAL", all_.shape)
