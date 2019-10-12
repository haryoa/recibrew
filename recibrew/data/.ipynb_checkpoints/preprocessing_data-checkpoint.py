def cleaning_data(df):
    df['Ingredients'] = df.Ingredients.str.replace('--',' ** ')
    from nltk import sent_tokenize, word_tokenize
    import nltk
    nltk.download('punkt')
    df['Ingredients_Custom'] = '<START>' + df['Ingredients'] + '<END>'
    df['Ingredients_Custom'] = df.Ingredients.apply(lambda x : '<START> ' + prepare_recipe(x) + ' <END>')
    df['Title_Custom'] = df.Title.apply(lambda x : '<START> ' + prepare_recipe(x) + ' <END>')
    return df


def prepare_recipe(ingredient):
    from nltk import word_tokenize
    ingredient = ' '.join(word_tokenize(ingredient))

    return ingredient
