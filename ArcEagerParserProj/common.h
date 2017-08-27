#pragma once
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#ifdef WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
//#include "execinfo.h"
#include <signal.h>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>

#include "cnn/training.h"
#include "cnn/cnn.h"
#include "cnn/expr.h"
#include "cnn/nodes.h"
#include "cnn/lstm.h"
#include "cnn/rnn.h"
#include "Corpus.h"

using namespace cnn::expr;
using namespace cnn;
using namespace std;


typedef unordered_map<unsigned, vector<float>> EmbedDict;
typedef  vector<int> t_unvec;

struct parser_config{
	Model* model;
	const EmbedDict* pretrained;

	static unsigned LAYERS;
	static unsigned INPUT_DIM;
	static unsigned HIDDEN_DIM;
	static unsigned ACTION_DIM;
	static unsigned PRETRAINED_DIM;
	static unsigned LSTM_INPUT_DIM;
	static unsigned POS_DIM;
	static unsigned REL_DIM;
	static float DROP_OUT;

	static bool USE_POS;
	static char* ROOT_SYMBOL;
	static unsigned kROOT_SYMBOL;
	static unsigned ACTION_SIZE;
	static unsigned VOCAB_SIZE;
	static unsigned TRAIN_VOCAB_SIZE;
	static unsigned UNK_WORD_IDX;
	static unsigned POS_SIZE;
	static unsigned NER_SIZE;
	static unsigned KBEST;

};

class seninfo {
public:
	t_unvec raw_sent;
	t_unvec sent;
	t_unvec sentPos;

	seninfo(const t_unvec& rs, const t_unvec& s, const t_unvec& sp) : raw_sent(rs), sent(s), sentPos(sp) {}
};

inline void string_split(char* str, char ch, vector<char*>& out)
{
	int len = strlen(str);
	out.clear();
	char* curitm = str;
	out.push_back(curitm);
	for (int i = 0; i<len; i++)
	{
		if (str[i] == ch) {
			str[i] = 0;
			if (i + 1<len) {
				curitm = str + i + 1;
				out.push_back(curitm);
			}
			else
				curitm = 0;
		}
	}
}

inline char* strtrim(char* source)
{
	while (*source != 0 && (*source == ' ' || *source == '\n' || *source == '\t'))
		source++;
	if (*source == 0)
		return source;
	int len = strlen(source);
	char *cur = source + len - 1;
	while (len>0 && (*cur == ' ' || *cur == '\n' || *cur == '\t')) {
		*cur = 0;
		cur--; len--;
	}
	return source;
};

struct score_pair {
	float score;
	int kbest_idx;
	int actidx;
	bool bdone;
	Expression adiste;
	score_pair() : bdone(false) {}
};

//#define lre_active_function
//#define soft_max_neg_scorer