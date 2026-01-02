# -*- coding: utf-8 -*-
import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Load the data
df = pd.read_excel('Translations.xlsx.ods', engine='odf')

print("="*60)
print("UNDERSTANDING THE DATA STRUCTURE")
print("="*60)

# Find all idiom header rows
idiom_rows = df[df['legenda'] == 'idiom']
print(f"\nNumber of idioms: {len(idiom_rows)}")
print(f"\nFirst 5 idiom examples from 'Unnamed: 1' column:")
for idx, row in idiom_rows.head(5).iterrows():
    print(f"  Row {idx}: {row['Unnamed: 1']}")

# Look at types
print("\n" + "="*60)
print("ANALYZING TYPES (1-4)")
print("="*60)

# Get rows with values 1, 2, 3, or 4 in DeepL column
type_values = df[df['DeepL'].isin([1, 2, 3, 4, '1', '2', '3', '4'])]
print(f"Number of type rows: {len(type_values)}")
print(f"\nValue counts for DeepL types:")
print(type_values['DeepL'].value_counts())
print(f"\nValue counts for Google Translate types:")
print(type_values['Google Translate'].value_counts())
print(f"\nValue counts for Gemini types:")
print(type_values['Gemini'].value_counts())
print(f"\nValue counts for ChatGPT types:")
print(type_values['ChatGPT'].value_counts())

# Get yes/no rows
print("\n" + "="*60)
print("ANALYZING YES/NO (Conveys meaning)")
print("="*60)
yes_no_rows = df[df['DeepL'].isin(['yes', 'no', 'Yes', 'No'])]
print(f"Number of yes/no rows: {len(yes_no_rows)}")
print(f"\nValue counts for DeepL yes/no:")
print(yes_no_rows['DeepL'].value_counts())
print(f"\nValue counts for Google Translate yes/no:")
print(yes_no_rows['Google Translate'].value_counts())
print(f"\nValue counts for Gemini yes/no:")
print(yes_no_rows['Gemini'].value_counts())
print(f"\nValue counts for ChatGPT yes/no:")
print(yes_no_rows['ChatGPT'].value_counts())

# Look for "red" cells - the notation says "red - the idiom is correct, the whole sentence is not"
print("\n" + "="*60)
print("LOOKING FOR SPECIAL NOTATION (Red cells)")
print("="*60)
# Red cells might be indicated somehow - let's check
# First let's look at the legend row
print("First row (legend):")
print(df.iloc[0].to_string())

# Let's also check what unique values there are
print("\n" + "="*60)
print("ALL UNIQUE VALUES IN EACH COLUMN")
print("="*60)
for col in df.columns:
    unique_vals = df[col].dropna().unique()
    if len(unique_vals) < 30:
        print(f"\n{col}:")
        print(unique_vals)
