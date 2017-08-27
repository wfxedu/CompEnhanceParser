#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <chrono>
#include <ctime>
#include <unordered_map>
#include <unordered_set>
#ifdef WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#include "execinfo.h"

#include <signal.h>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "cnn/training.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/nodes.h"
#include "cnn/lstm.h"
#include "cnn/rnn.h"
#include "TreeReader.h"
#include "SRParser.h"
#include "util.h"
#include "ParserTrainer.h"
#include "ParserTester.h"
#include "model_transfer.h"

#ifdef WIN32
#include <process.h>
#define getpid _getpid
#endif

/*
float* v1 = new float[3 * 8];
float* v2 = new float[3];
Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> arra1(v1, 8, 3);
Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> arra2(v2, 3, 1);
Eigen::Map<Eigen::MatrixXf, Eigen::Aligned> arra3(v1, 3*8, 1);
for (int i = 0;i < 3 * 8;i++) {
arra3(i) = i;
}
for (int i = 0;i < 3 ;i++) {
arra2(i) = i+1;
}
for (int r = 0;r < 8;r++) {
for (int c = 0;c < 3;c++)
printf("%f,", arra1(r, c));
printf("\n");
}
auto aa = arra1*arra2;
for (int i = 0;i < 8;i++) {
printf("%f,", aa(i));
}
printf("\n");
*/


using namespace cnn::expr;
using namespace cnn;
using namespace std;

EmbedDict								pretrained;
DatasetTB									sr_corpus;
extern bool g_is_punt_en;

void fill_cfg() {
	if (false) { //v0
		parser_config::LAYERS = 1;
		parser_config::INPUT_DIM = 100;
		parser_config::HIDDEN_DIM = 100;
		parser_config::ACTION_DIM = 50;
		parser_config::PRETRAINED_DIM = 100;
		parser_config::LSTM_INPUT_DIM = 100;
		parser_config::POS_DIM = 50;
		parser_config::REL_DIM = 50;
		parser_config::USE_POS = true;

		parser_config::ROOT_SYMBOL = "ROOT";
		parser_config::kROOT_SYMBOL = 0;
		parser_config::ACTION_SIZE = 0;
		parser_config::VOCAB_SIZE = 0;
		parser_config::POS_SIZE = 0;
	}
	else { //for chinese
		parser_config::LAYERS = 1;
		parser_config::INPUT_DIM = 200;
		parser_config::HIDDEN_DIM = 200;
		parser_config::ACTION_DIM = 50;
		parser_config::PRETRAINED_DIM = 50;
		parser_config::LSTM_INPUT_DIM = 200;
		parser_config::POS_DIM = 50;
		parser_config::REL_DIM = 50;
		parser_config::USE_POS = true;

		parser_config::ROOT_SYMBOL = "ROOT";
		parser_config::kROOT_SYMBOL = 0;
		parser_config::ACTION_SIZE = 0;
		parser_config::VOCAB_SIZE = 0;
		parser_config::POS_SIZE = 0;
	}



	
}

int main(int argc, char** argv) {
	cnn::Initialize(argc, argv);
	float unk_prob = 0.2f;
	bool train_mode = false;
	char* training_data = 0;
	bool embedding_type = false;
	char* words_embedding = 0;
	int embedding_size = 0;
	char* model_path = 0;
	char* out_model_path = 0;
	char* dev_data = 0;
	char* test_data = 0;
	char* test_out = 0;
	char* save_name = 0;
	char* load_name = 0;
	char* convert_word = 0;
	char* test_real = 0;
	char* tran_args = 0;

	fill_cfg();
	for (int i = 0; i < argc; i++) {
		if (!strcmp(argv[i], "-unk_p")) {
			unk_prob = atof(argv[i + 1]); i++;
		}
		if (!strcmp(argv[i], "-train")) {
			training_data = argv[i + 1]; i++;
		}
		if (!strcmp(argv[i], "-save_name")) {
			save_name = argv[i + 1]; i++;
		}
		if (!strcmp(argv[i], "-load_name")) {
			load_name = argv[i + 1]; i++;
		}
		if (!strcmp(argv[i], "-dropout")) {
			parser_config::DROP_OUT = atof(argv[i + 1]); i++;
			printf("dropout rate: %f\n", parser_config::DROP_OUT);
		}
		if (!strcmp(argv[i], "-tm")) {
			train_mode = true; 
		}
		if (!strcmp(argv[i], "-em_org")) {
			embedding_type = true;
		}
		if (!strcmp(argv[i], "-punt_en")) {
			g_is_punt_en = true;
		}
		if (!strcmp(argv[i], "-convert_word")) {
			convert_word = argv[i + 1]; i++;
		}
		if (!strcmp(argv[i], "-em_path")) {
			words_embedding = argv[i + 1]; i++;
		}
		if (!strcmp(argv[i], "-em_sz")) {
			embedding_size = atoi(argv[i + 1]); i++;
		}
		if (!strcmp(argv[i], "-kbest")) {
			parser_config::KBEST = atoi(argv[i + 1]); i++;
			printf("setting kbest=%d\n", parser_config::KBEST);
		}
		if (!strcmp(argv[i], "-model")) {
			model_path = argv[i + 1]; i++;
		}
		if (!strcmp(argv[i], "-out_model_path")) {
			out_model_path = argv[i + 1]; i++;
		}
		if (!strcmp(argv[i], "-dev")) {
			dev_data = argv[i + 1]; i++;
		}
		if (!strcmp(argv[i], "-test")) {
			test_data = argv[i + 1]; i++;
		}
		if (!strcmp(argv[i], "-real_test")) {
			test_real = argv[i + 1]; i++;
		}		
		if (!strcmp(argv[i], "-tran_args")) {
			tran_args = argv[i + 1]; i++;
		}		
		if (!strcmp(argv[i], "-test_out")) {
			test_out = argv[i + 1]; i++;
		}
	}
	/////////////////////////////////////////////////////////////////////////////////////////////
	unsigned unk_strategy = 1; //only one way "STOCHASTIC REPLACEMENT"
	assert(unk_prob >= 0.); assert(unk_prob <= 1.);
	//------------------------------------------------------
	if (train_mode) {
		cerr << "loading training data..." << endl;
		sr_corpus.load_train (training_data);
		//sr_corpus.save_config("corpus_cfg.ini");
		cerr << "finished!" << endl;
	}
	else {
		//sr_corpus.load_config("corpus_cfg.ini");
		cerr << "loading training data..." << endl;
		sr_corpus.load_train(training_data);
	}
	//------------------------------------------------------
	cerr << "Words: " << sr_corpus.words2Int.nTokens << endl;
	parser_config::TRAIN_VOCAB_SIZE = sr_corpus.words2Int.nTokens;
	sr_corpus.load_dev(dev_data);
	sr_corpus.load_test(test_data);//error
	parser_config::UNK_WORD_IDX = sr_corpus.words2Int.get(treebank::UNK, 0);
	parser_config::VOCAB_SIZE = sr_corpus.words2Int.nTokens + 1;
	int add_words = parser_config::VOCAB_SIZE - parser_config::TRAIN_VOCAB_SIZE;
	printf("#Addtional words=%d(%f)\n", add_words, add_words/(float)parser_config::TRAIN_VOCAB_SIZE);

	parser_config::ACTION_SIZE = sr_corpus.prel2Int.nTokens*2 + 6;
	parser_config::POS_SIZE = sr_corpus.pos2Int.nTokens + 10; 
	parser_config::NER_SIZE = sr_corpus.ner2Int.nTokens + 10;
	printf("NER Size %d\n", parser_config::NER_SIZE);
	for (auto iter1 = sr_corpus.ner2Int.tokens2Int.begin(); iter1 != sr_corpus.ner2Int.tokens2Int.end(); iter1++) {
		printf("%s(%d), ", iter1->first,iter1->second);
	}
	printf("\n");
	//------------------------------------------------------
	EmbedDict pretrained;
	if (words_embedding) { //error!!!!!!!!
		parser_config::PRETRAINED_DIM = embedding_size;
		cerr << "loading EmbedDict..." << endl;
		util::load_EmbedDict(sr_corpus, pretrained, words_embedding, embedding_type);
		cerr << "finished!" << endl;
	}
	parser_config nn_cfg;
	if (test_real) {
		sr_corpus.testing.sentences.clear();
		sr_corpus.testing.sentences1.clear();
		sr_corpus.load_test(test_real);
	}
    //------------------------------------------------------
	if (train_mode) {
		const string oname = out_model_path;
		cerr << "out model name: " << oname << endl;

		ParserTrainer trainer(nn_cfg, sr_corpus, pretrained);
		if (model_path) {
			cerr << "load model: " << model_path << endl;
			trainer.LoadModel(model_path);
			cerr << "finished loading model" << endl;
		}
		if (save_name) {
			save_cnn_model_byname(save_name, &trainer.model);
			string w2i_path = ((string)save_name + ".word2idx");
			save_string2idx(w2i_path.c_str(), sr_corpus.words2Int.tokens2Int);
		} 
		if (load_name) {
			parameters_store ps;
			load_cnn_model_byname(load_name,ps, &trainer.model);
			if (convert_word) {
				std::map<std::string, unsigned> tokens2Int;;
				string w2i_path = ((string)load_name + ".word2idx");
				load_string2idx(w2i_path.c_str(), tokens2Int);
				convert_args(ps, trainer.model, convert_word, tokens2Int, sr_corpus.words2Int.tokens2Int);
			}
			if (tran_args) {
				transfer_args(ps, trainer.model, tran_args);
			}
		}
		trainer.InitParser();
		//sr_corpus.load_dev(dev_data);
		trainer.train(unk_prob, unk_strategy, oname);
	}
	else 
	{
		ParserTester tester(nn_cfg, sr_corpus, pretrained);
		if (model_path) {
			cerr << "load model: " << model_path << endl;
			tester.LoadModel(model_path);
			cerr << "finished loading model" << endl;
		}
		if (save_name) {
			save_cnn_model_byname(save_name, &tester.model);
			string w2i_path = ((string)save_name + ".word2idx");
			save_string2idx(w2i_path.c_str(), sr_corpus.words2Int.tokens2Int);
		}
		if (load_name) {
			parameters_store ps;
			load_cnn_model_byname(load_name, ps, &tester.model);
			if (convert_word) {
				std::map<std::string, unsigned> tokens2Int;;
				string w2i_path = ((string)load_name + ".word2idx");
				load_string2idx(w2i_path.c_str(), tokens2Int);
				convert_args(ps, tester.model, convert_word, tokens2Int, sr_corpus.words2Int.tokens2Int);
			}
			if (tran_args) {
				transfer_args(ps, tester.model, tran_args);
			}
		}
		tester.InitParser();

		string tout = test_out;
		tester.test(unk_prob, unk_strategy, tout);
	}
}
