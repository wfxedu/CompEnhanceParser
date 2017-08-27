#include "common.h"
unsigned parser_config::LAYERS = 2;
unsigned parser_config::INPUT_DIM = 100;
unsigned parser_config::HIDDEN_DIM = 100;
unsigned parser_config::ACTION_DIM = 20;
unsigned parser_config::PRETRAINED_DIM = 50;
unsigned parser_config::LSTM_INPUT_DIM = 100;
unsigned parser_config::POS_DIM = 10;
unsigned parser_config::REL_DIM = 20;
float parser_config::DROP_OUT = 0.2f;

bool parser_config::USE_POS = true;

char* parser_config::ROOT_SYMBOL = "ROOT";
unsigned parser_config::kROOT_SYMBOL = 0;
unsigned parser_config::ACTION_SIZE = 0;
unsigned parser_config::VOCAB_SIZE = 0;
unsigned parser_config::TRAIN_VOCAB_SIZE = 0;
unsigned parser_config::UNK_WORD_IDX = -1;
unsigned parser_config::POS_SIZE = 0;
unsigned parser_config::KBEST = 10;
unsigned parser_config::NER_SIZE = 0;


