import pandas as pd
CELLULAR_LOCATION_FILE = r'C:\Users\user4\PycharmProjects\fog-computing-based-collision-warning-system\visualization\cellular_location.csv'
SORTED_CELLULAR_LOCATION_FILE = r'C:\Users\user4\PycharmProjects\fog-computing-based-collision-warning-system\visualization\sorted_cellular_location.csv'

df = pd.read_csv(CELLULAR_LOCATION_FILE, error_bad_lines=False, sep=' ')
new_df = df.sort_values(by=['x'], ascending=[1])
print(new_df.head(5))
ID = new_df['ID']
with open(SORTED_CELLULAR_LOCATION_FILE, 'a+', encoding='utf-8') as file:
    file.writelines('x,y\n')
    for id in ID:
        file.writelines(str(df['x'][id]) + ',' + str(df['y'][id]) + '\n')
