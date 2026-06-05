import pickle
import konlpy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
form konlpy.tag import Okt, Komoran
from sklearn.preprocessing import LabelEncoder
form keras.util import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import re