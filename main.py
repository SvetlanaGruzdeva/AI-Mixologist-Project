#!/usr/bin/env python
# coding: utf-8

# ## TOC:
# * [Step 1: prepare dataset to be used for identifying ingredients and measures](#1)
# * [Step 2: ingredients](#2)
# * [Step 3: measures](#3)
# * [Step 4: garnish](#4)
# * [Step 5: generate cocktail](#5)
# * [Step 6: select alcoholic ingredients from presented variaty based on it's type](#6)

# In[1]:


import warnings
import pandas as pd
import numpy as np
import nltk
import collections
import random
import ipywidgets as widgets
from IPython.display import clear_output, display
from tkinter import *
from tkinter import messagebox as mb


# In[2]:


pd.set_option('display.max_columns', 100)
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv('./clean_data/clean_data.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)


# ### Step 1: prepare dataset to be used for identifying ingredients and measures <a class="anchor" id="1"></a>
# - each cocktail = one row
# - punches to be excluded due to scaling issues
# - rows related to garnish to be excluded
# - all ingredients generalized and combined in one column

# In[4]:


df.head()


# In[5]:


df_processed = df.copy()     # This way we will always have access to our original dataset


# In[6]:


# Drop rows with related to punches to avoid scaling issue in next steps

df_processed.drop(df_processed.loc[df_processed['strCategory'].str.contains('Punch')].index, axis=0, inplace=True)
df_processed.drop(df_processed.loc[df_processed['strGlass'].str.contains('Pitcher')].index, axis=0, inplace=True)


# In[7]:


# Drop nulls in values to avoid ingredients without measures in generated cocktail

cond1 = (df_processed['Value_ml'].isnull())
cond2 = (df_processed['Value_gr'].isnull())
cond3 = (df_processed['Garnish_type'].isnull())

df_processed.drop(df_processed.loc[cond1 & cond2 & cond3].index,
                 axis=0, inplace=True)


# In[8]:


# Combine non-alcoholic ingredients as they are and generalised alcoholic ingredients in one column

df_processed['Ingredients'] = np.where(~df_processed['Basic_taste'].isnull(), df_processed['strIngredients'], np.nan)
df_processed['Ingredients'] = np.where(df_processed['Ingredients'].isnull(),
                                       df_processed['Alc_type'], df_processed['Ingredients'])


# In[9]:


# Replace ' ' with '_' to keep adjectives with nouns

df_processed['Ingredients'] = df_processed['Ingredients'].apply(lambda x: x.replace(' ', '_'))


# In[10]:


# Drop rows with garnish and save as a separate dataset, this part of the cocktail will be processed on the later stages

df_processed_no_garnish = df_processed.drop(df_processed.loc[~df_processed['Garnish_type'].isnull()].index, axis=0)


# In[11]:


# Define reduced datasets containing only the columns required for particular step

df_ingredients = df_processed_no_garnish[['strDrink', 'Ingredients']]
df_measures = df_processed_no_garnish[['strDrink', 'Ingredients', 'Value_ml', 'Value_gr']]


# In[12]:


df_ingredients.head()


# ### Step 2: ingredients:<a class="anchor" id="2"></a>
# - combine all ingredients per cocktail in one string
# - split ingerients by pairs
# - compute most common (25%) pairs

# In[13]:


# Reallocate ingredients as columns so each column takes only one row

df_ingredients = df_ingredients[df_ingredients['Ingredients'] != 'Ingredients']
s =  df_ingredients.groupby('strDrink').cumcount().add(1)
df_ingredients = (df_ingredients.set_index(['strDrink',s])
        .unstack()
        .sort_index(axis=1, level=1)
       )
df_ingredients.columns = ['{}_{}'.format(a, b) for a,b in df_ingredients.columns]

df_ingredients = df_ingredients.reset_index()
df_ingredients.head()


# In[14]:


# Combine all ingredients per cocktial in one column

df_ingredients['Ingredients'] = df_ingredients.drop(['strDrink'], axis=1).fillna('').apply(lambda x: ' '.join(x), axis=1)
df_ingredients = df_ingredients[['strDrink', 'Ingredients']]
df_ingredients.head()


# In[15]:


# Generate pairs from ingredients of each cocktail and combine them in one list

bigram = [list(nltk.bigrams(nltk.word_tokenize(i))) for i in df_ingredients['Ingredients']]
pairs_list = [j for i in bigram for j in i]
print(len(pairs_list))
pairs_list[:10]


# There are some tuples where elements are swopped but essential taste of such combination is not unique. Such tuples need to be alighned.

# In[16]:


# First, define a list of tuples to be amended

to_aligh = []
for a in pairs_list:
    for b in pairs_list:
        if a != b:
            if b[1] == a[0]:
                if b[0] == a[1]:            # If both elements of tuple are equal to the swopped tuple under the check
                    if a not in to_aligh and b not in to_aligh:
                        to_aligh.append(b)


# In[17]:


# Next, amend them

pairs_list_aligned = []
for i in pairs_list:
    if i in to_aligh:
        pairs_list_aligned.append((i[1], i[0]))
    else:
        pairs_list_aligned.append(i)


# In[18]:


# Check that total number of tuples hasn't changed, only content should be amended

len(pairs_list_aligned) == len(pairs_list)


# In[19]:


# Define 25% of the most common pairs as a separate list

counter=collections.Counter(pairs_list_aligned)
print(len(counter))
common_pairs = counter.most_common(int(len(counter)*0.25))
common_pairs


# In[20]:


common_ingredients = []

for n in common_pairs:
    common_ingredients.append(n[0][0])
    common_ingredients.append(n[0][1])
common_ingredients = list(set(common_ingredients))
common_ingredients


# ### Step 3: measures<a class="anchor" id="3"></a>

# In[21]:


# Combine values in one column and define measure for each in a separate column

df_measures['Measure'] = np.where(df_measures['Value_ml'].isnull(), 'gr', 'ml')
df_measures['Value'] = df_measures['Value_ml'].fillna(0) + df_measures['Value_gr'].fillna(0)
df_measures.head()


# In[22]:


# Combine value and measure in one string, this way it will be easier to pick up random value together with correct measure

df_measures['Value'] = df_measures['Value'].astype('object').apply(lambda x: str(x))
df_measures['Value_Measure'] = df_measures[['Value', 'Measure']].apply(lambda x: ' '.join(x), axis=1)
df_measures.head()


# ### Step 4: garnish<a class="anchor" id="4"></a>

# ***Prepare list of main ingredients with garnish for each cocktail in dataset***

# In[23]:


# Define reduced dataframe which contains only relevant fields

df_garnish = df_processed[['strDrink', 'Ingredients', 'Value_ml', 'Garnish_amount', 'Garnish_type']]
df_garnish.head()


# In[24]:


# Combine value and measure of garnish in one column so it will be easier to pick it up later in the code

df_garnish['Garnish_ingr'] = np.where(~df_garnish['Garnish_amount'].isnull(), df_garnish['Ingredients'], np.nan)
df_garnish['Garnish'] = df_garnish[['Garnish_ingr', 'Garnish_amount', 'Garnish_type']].fillna('').apply(lambda x:
                                                                                                       ' '.join(x), axis=1)
df_garnish['Garnish'] = df_garnish['Garnish'].apply(lambda x: x.replace('0 top', 'top'))
df_garnish['Garnish'] = np.where(df_garnish['Garnish'] == '  ', np.nan, df_garnish['Garnish'])
df_garnish.head()


# In[25]:


# Rearrange dataframe that way that it's possible to identify garnish per cocktail per dominant ingredient

drink_name_list = []
garnish_list = []
ingredient_list = []

for drink in df_garnish['strDrink'].unique():      # For each cocktial
    df_selected = df_garnish.loc[df_garnish['strDrink'] == drink]
    max_value = df_selected['Value_ml'].max()

    for ingr in df_selected.loc[df_selected['Value_ml'] == max_value]['Ingredients']:
        for garnish in df_selected['Garnish'].unique():
            drink_name_list.append(drink)
            garnish_list.append(garnish)           # Include all garnishes for the drink
            ingredient_list.append(ingr)           # Include an ingredient taking the biggest part of the drink

df_garnish_final = pd.DataFrame({'Drink':drink_name_list, 'Ingredient':ingredient_list, 'Garnish':garnish_list})
# df_garnish_final.drop(df_garnish_final.loc[df_garnish_final['Garnish'].isnull()].index, axis=0, inplace = True)


# ### Step 5: generate cocktail <a class="anchor" id="5"></a>

# **Generate a frame for new combinations:**
# - define total number of ingerients (random choice from a range 3-6))
# - from ingredients included in top pairs pick one randomly
# - find suitable pair for this ingredient (from all pairs, but give top pairs bigger weight)
# - do the same for the next ingredient but check that it's not included already
# - etc until limit is reached

# In[26]:


def first_ingredient():
    '''Picks the first ingredient randomly from the top popular'''

    return random.choice(common_ingredients)


# In[27]:


def next_ingredient(new_cocktail):
    '''Picks the next ingredient based on matching pairs with previous ingredient'''
    temp_list = []
    for i in set(pairs_list_aligned):
        if new_cocktail[-1] in i:
            temp_list.append(i)
    random_pair = random.choice(temp_list)
    if random_pair[0] == new_cocktail[-1]:
        next_ingr = random_pair[1]
    else:
        next_ingr = random_pair[0]
  
    return next_ingr


# **Define volume of each ingredient defined above**

# In[28]:


def volume(new_cocktail):
    '''Picks random volume of each ingredient'''
    
    for i in new_cocktail:
        volume = [random.choice(df_measures.loc[df_measures['Ingredients'] == i]['Value_Measure'].tolist()) for i in new_cocktail]
        new_cocktail_final = pd.DataFrame({'Ingredient': new_cocktail, 'Volume': volume})
    
    return new_cocktail_final


# **Select garnish**

# In[29]:


def garnish(new_cocktail_final):
    '''Picks garnish based on the main ingredient in generated cocktail and original dataset'''
    
    # Identify the main ingredient of a generated cocktail
    new_cocktail_final = new_cocktail_final.join(new_cocktail_final['Volume'].str.rsplit(n=1, expand=True).rename(columns={
                                                                                                    0: 'Value', 1: 'Measure'}))
    new_cocktail_final['Value'] = new_cocktail_final['Value'].astype('float')
    main_ingr = random.choice(new_cocktail_final.loc[new_cocktail_final['Value'] ==
                                                                     new_cocktail_final['Value'].max()]['Ingredient'].tolist())
    new_cocktail_final.drop(['Value', 'Measure'], axis=1, inplace=True)
    
    # Find suitable garnish and add to the recipe
    lst = df_garnish_final.loc[df_garnish_final['Ingredient'] == main_ingr]['Garnish'].tolist()

    if lst:
        garnish_to_add = random.choice(lst)
        new_cocktail_final.loc[len(new_cocktail_final)] = ['Garnish', garnish_to_add]    
    
    return new_cocktail_final


# **Replace alcoholic ingredients with specific ingredients**

# In[30]:


def liqueurs(new_cocktail_final):
    '''Picks alcoholic ingredients from presented variaty based on it's type'''
    
    for i in new_cocktail_final['Ingredient']:
        lst = df.loc[df['Alc_type'] == i.replace('_', ' ')]['strIngredients'].tolist()
        if lst:
            new_cocktail_final['Ingredient'] = new_cocktail_final['Ingredient'].apply(lambda x: x.replace(i, random.choice(lst)))
            
    return new_cocktail_final