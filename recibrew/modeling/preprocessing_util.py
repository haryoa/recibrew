import tensorflow as tf


def create_tokenize(lang, data_fit_for_tokenize, lang_tokenizer=None):
    if lang_tokenizer is None:
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
            filters='')
        lang_tokenizer.fit_on_texts(data_fit_for_tokenize)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post')
    return tensor, lang_tokenizer


def get_seq_max_len(dataset):
    example_input_batch, example_target_batch = next(iter(dataset))
    return example_input_batch.shape, example_target_batch.shape


def prepare_to_dataset_tf(batch_size, input_tensor_train, target_tensor_train, buffer_size):
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    steps_per_epoch = len(input_tensor_train)//batch_size
    return dataset, steps_per_epoch


def convert_to_text(lang, tensor):
    string_mix = ''
    for t in tensor:
        if t != 0:
            string_mix += lang.index_word[t] + ' '
    return string_mix


def get_list_fit_for_tokenizer(df):
    list_fit_for_tokenizer = df['Ingredients_Custom'].append(df['Title_Custom'])
    list_fit_for_tokenizer = list_fit_for_tokenizer.tolist()
    return list_fit_for_tokenizer
