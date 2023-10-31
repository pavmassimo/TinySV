This folder contains all the needed files to train the tiny speaker embeddings extractor and test its performance by using Mean Cosine Similarity and Best-Match cosine similarity approaches.

----
TRAINING THE EMBEDDINGS EXTRACTOR MODEL

To perform the training of the embeddings extractor model, run the script "train_fe_cnn.py" with the following command:

python train_fe_cnn.py

This will train the convolutional embedding extractor, and will produce in the folder "models/" the files "cnn-librispeech-classifier.h5" and "d-vector-extractor-256.h5". The latter is the model able to extract the embedding from a speech utterance in the shape of a 256 long array of floating point values.

----
TESTING THE EMBEDDINGS EXTRACTOR WITH BEST-MATCH COSSIM AND MEAN COSSIM ALGORITHMS

Once the model for extracting embeddings has been produced, run the following commands to execute the tests. You will find accuracy, f1-score, EER and AUC metrics respectively in the files "test-results-td-bestmatch.txt" and "test-results-td-meancos.py".

python testing-meancos.py 0 1
python testing-meancos.py 1 1
python testing-meancos.py 2 1
python testing-meancos.py 3 1
python testing-meancos.py 0 8
python testing-meancos.py 1 8
python testing-meancos.py 2 8
python testing-meancos.py 3 8
python testing-meancos.py 0 16
python testing-meancos.py 1 16
python testing-meancos.py 2 16
python testing-meancos.py 3 16
python testing-meancos.py 0 64
python testing-meancos.py 1 64
python testing-meancos.py 2 64
python testing-meancos.py 3 64

python testing-bestmatch.py 0 1
python testing-bestmatch.py 1 1
python testing-bestmatch.py 2 1
python testing-bestmatch.py 3 1
python testing-bestmatch.py 0 8
python testing-bestmatch.py 1 8
python testing-bestmatch.py 2 8
python testing-bestmatch.py 3 8
python testing-bestmatch.py 0 16
python testing-bestmatch.py 1 16
python testing-bestmatch.py 2 16
python testing-bestmatch.py 3 16
python testing-bestmatch.py 0 64
python testing-bestmatch.py 1 64
python testing-bestmatch.py 2 64
python testing-bestmatch.py 3 64

In this commands, the first command line argument represents the speaker ID, and the second argument represents the number of enrollment samples used for the test. For example, python testing-bestmatch.py 0 16 performs the test on speaker 0 using 16 enrollment samples.