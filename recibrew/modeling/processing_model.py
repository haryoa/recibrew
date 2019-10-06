from recibrew.modeling.nn_model import Encoder, Decoder, DecoderAttention
import tensorflow as tf


class Seq2SeqBaseline:

    def __init__(self,  embedding_dim, units, vocab_size, checkpoint_dir, lang_tokenizer):
        self.embedding_dim = embedding_dim
        self.lang_tokenizer = lang_tokenizer
        self.units = units
        self.vocab_size = vocab_size
        self.encoder = Encoder(vocab_size, embedding_dim, units)
        self.decoder = Decoder(self.encoder.embedding, units, vocab_size)

        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.__prepare_checkpoint(checkpoint_dir, self.optimizer, self.encoder, self.decoder)

    @tf.function
    def train_step(self, inp, targ, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder(inp, enc_hidden)

            dec_hidden = enc_hidden

            dec_input = tf.expand_dims([self.lang_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden = self.decoder(dec_input, dec_hidden, enc_output)

                loss += self.loss_function(targ[:, t], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    def train(self, epochs, steps_per_epoch, dataset):
        import time

        for epoch in range(epochs):
            start = time.time()

            enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0

            for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
                batch_loss = self.train_step(inp, targ, enc_hidden)
                total_loss += batch_loss

                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                                 batch,
                                                                 batch_loss.numpy()))
            # saving (checkpoint) the model every 2 epochs
            if (epoch + 1) % 2 == 0:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)

                print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                                    total_loss / steps_per_epoch))
                print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    def __prepare_checkpoint(self, checkpoint_dir, optimizer, encoder, decoder):
        import os
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                              encoder=encoder,
                                              decoder=decoder)


class Seq2SeqAttention:

    def __init__(self,  embedding_dim, units, vocab_size, checkpoint_dir, lang_tokenizer, batch_size):
        self.embedding_dim = embedding_dim
        self.lang_tokenizer = lang_tokenizer
        self.units = units
        self.vocab_size = vocab_size
        self.encoder = Encoder(vocab_size, embedding_dim, units)
        self.decoder = DecoderAttention(self.encoder.embedding, units, vocab_size)
        self.batch_size = batch_size
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.__prepare_checkpoint(checkpoint_dir, self.optimizer, self.encoder, self.decoder)


    @tf.function
    def train_step(self, inp, targ, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder(inp, enc_hidden)

            dec_hidden = enc_hidden

            dec_input = tf.expand_dims([self.lang_tokenizer.word_index['<start>']] * self.batch_size, 1)

            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)

                loss += self.loss_function(targ[:, t], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    def train(self, epochs, steps_per_epoch, dataset):
        import time

        for epoch in range(epochs):
            start = time.time()

            enc_hidden = self.encoder.initialize_hidden_state(self.batch_size)
            total_loss = 0

            for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
                batch_loss = self.train_step(inp, targ, enc_hidden)
                total_loss += batch_loss

                if batch % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                                 batch,
                                                                 batch_loss.numpy()))
            # saving (checkpoint) the model every 2 epochs
            if (epoch + 1) % 2 == 0:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)

                print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                                    total_loss / steps_per_epoch))
                print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    def predict(self, sentence, max_seq_input, max_seq_output, use_choice=False):
        import numpy as np
        attention_plot = np.zeros((max_seq_input, max_seq_output))

        inputs = [self.lang_tokenizer.word_index[i] for i in sentence.strip().split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                               maxlen=max_seq_input,
                                                               padding='post')
        inputs = tf.convert_to_tensor(inputs)

        result = ''

        hidden = [tf.zeros((1, self.units))]
        enc_out, enc_hidden = self.encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.lang_tokenizer.word_index['<start>']], 0)

        for t in range(max_seq_output):
            predictions, dec_hidden, attention_weights = self.decoder(dec_input,
                                                                 dec_hidden,
                                                                 enc_out)
            # storing the attention weights to plot later on
            attention_weights = tf.reshape(attention_weights, (-1, ))
            print(attention_weights.shape)
            attention_plot[:,t] = attention_weights.numpy()

            if use_choice:
                softmax = tf.keras.layers.Softmax()
                predicted_id = np.random.choice(np.arange(predictions.shape[1]),1,p=softmax(predictions).numpy()[0])[0]
            else:
                predicted_id = tf.argmax(predictions[0]).numpy()

            result += self.lang_tokenizer.index_word[predicted_id] + ' '

            if self.lang_tokenizer.index_word[predicted_id] == '<end>':
                return result, sentence, attention_plot

            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims([predicted_id], 0)

        return result, sentence, attention_plot

    def __prepare_checkpoint(self, checkpoint_dir, optimizer, encoder, decoder):
        import os
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                              encoder=encoder,
                                              decoder=decoder)

