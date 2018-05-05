import pandas as pd
import numpy as np


df = pd.read_csv('csv/in.csv')

print df.shape
print df.describe()


df_new = pd.DataFrame()

for col in df.columns[2:12]:
	print col

	df_col = df[col]
	df_new[col] = (df_col-df_col.min())/(df_col.max()-df_col.min())


print '**data_channel'

for col in df.columns[13:19]:
	print col
	df_col = df[col]

print '**'

for col in df.columns[19:31]:
	print col

	df_col = df[col]
	df_new[col] = (df_col-df_col.min())/(df_col.max()-df_col.min())


print '**days_of_week'

for col in df.columns[31:39]:
	print col

	df_col = df[col]

print '**'

for col in df.columns[39:60]:
	print col

	df_col = df[col]
	df_new[col] = (df_col-df_col.min())/(df_col.max()-df_col.min())


print 'target'

for col in df.columns[60:]:
	print col

	df_col = df[col]
	df_new[col] = np.log10(df_col)
	# df_new[col] = (df_col-df_col.min())/(df_col.max()-df_col.min())


df_new.to_csv('csv/prep_data_log.csv',index=False)
# df_new.to_csv('csv/prep_data_norm.csv',index=False)