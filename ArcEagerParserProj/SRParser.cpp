#include "SRParser.h"
#include "TreeReader.h"
//#define use_ner_cfg

void word_layer::Init_layer(parser_config& cfg)
{
	layer::Init_layer(cfg);
	Model* model = cfg.model;
	const unordered_map<unsigned, vector<float>>& pretrained = *cfg.pretrained;
	model->push_scope(model->get_scope() + "/word_layer/");
	//////////////////////////////////////////////////////////////////////////
	p_w = model->add_lookup_parameters(parser_config::TRAIN_VOCAB_SIZE, Dim(parser_config::INPUT_DIM, 1),"words_new");
	p_w2l = model->add_parameters(Dim(parser_config::LSTM_INPUT_DIM, parser_config::INPUT_DIM), "w2l");
#ifdef use_ner_cfg
	p_n = model->add_lookup_parameters(parser_config::NER_SIZE, Dim(parser_config::POS_DIM, 1),"ner");
	p_n2l = model->add_parameters(Dim(parser_config::LSTM_INPUT_DIM, parser_config::POS_DIM), "n2l");
#endif
	p_ib = model->add_parameters(Dim(parser_config::LSTM_INPUT_DIM, 1),"ib");
	if (parser_config::USE_POS) {
		p_p = model->add_lookup_parameters(parser_config::POS_SIZE, Dim(parser_config::POS_DIM, 1),"pos");
		p_p2l = model->add_parameters(Dim(parser_config::LSTM_INPUT_DIM, parser_config::POS_DIM),"p2l");
	}
	if (pretrained.size() > 0) {
		p_t = model->add_lookup_parameters(parser_config::VOCAB_SIZE, Dim(parser_config::PRETRAINED_DIM, 1),"pre_train_all_new");
		for (auto it : pretrained)
			p_t->Initialize(it.first, it.second);
		p_t2l = model->add_parameters(Dim(parser_config::LSTM_INPUT_DIM, parser_config::PRETRAINED_DIM),"t2l");
	}
	else {
		p_t = nullptr;
		p_t2l = nullptr;
	}
	model->pop_scope();
}

void word_layer::Init_Graph(ComputationGraph* hg)
{
	layer::Init_Graph(hg);
	//for word layer
	ib = parameter(*hg, p_ib);
	w2l = parameter(*hg, p_w2l);
#ifdef use_ner_cfg
	n2l = parameter(*hg, p_n2l);
#endif
	if (parser_config::USE_POS)
		p2l = parameter(*hg, p_p2l);
	if (p_t2l)
		t2l = parameter(*hg, p_t2l);
}

Expression word_layer::build(unsigned word_id, unsigned pos_id, unsigned orgword_id, unsigned ner_id)
{
	//assert(sent[i] < VOCAB_SIZE);
	Expression w;
	if (word_id >= parser_config::TRAIN_VOCAB_SIZE)
		w = lookup(*m_hg, p_w, parser_config::UNK_WORD_IDX);
	else
	   w = lookup(*m_hg, p_w, word_id);

#ifdef use_ner_cfg
    Expression n = lookup(*m_hg, p_n, ner_id);
	vector<Expression> args = { ib, w2l, w , n2l, n}; // learn embeddings
#else
	vector<Expression> args = { ib, w2l, w };// , n2l, n}; // learn embeddings
#endif

	if (parser_config::USE_POS) { // learn POS tag?
		Expression p = lookup(*m_hg, p_p, pos_id);
		args.push_back(p2l);
		args.push_back(p);
	}
#ifdef use_ner_cfg
	if (p_t && m_cfg->pretrained->count(orgword_id) && orgword_id <parser_config::TRAIN_VOCAB_SIZE) {  // change 1, improve result
#else
	if (p_t && m_cfg->pretrained->count(orgword_id)) {  // change 1, improve result
#endif
		Expression t = const_lookup(*m_hg, p_t, orgword_id);
		args.push_back(t2l);
		args.push_back(t);
	}
	return leaky_rectify(affine_transform(args),0.01);
}
//----------------------------------------------
void parser_state::Init_layer(parser_config& cfg)
{
	layer::Init_layer(cfg);
	Model* model = cfg.model;
	const EmbedDict& pretrained = *cfg.pretrained;
	//////////////////////////////////////////////////////////////////////////
	model->push_scope(model->get_scope() + "/parser_state/");
	stack_lstm = LSTMBuilder(parser_config::LAYERS*2, parser_config::LSTM_INPUT_DIM, parser_config::HIDDEN_DIM,"stack_lstm", model);
	buffer_lstm = LSTMBuilder(parser_config::LAYERS * 2, parser_config::LSTM_INPUT_DIM, parser_config::HIDDEN_DIM,"buffer_lstm", model);
	action_lstm = LSTMBuilder(parser_config::LAYERS*2, parser_config::ACTION_DIM, parser_config::HIDDEN_DIM, "action_lstm",model);
	stack_lstm.set_dropout(parser_config::DROP_OUT);
	buffer_lstm.set_dropout(parser_config::DROP_OUT);
	action_lstm.set_dropout(parser_config::DROP_OUT);

	p_action_start = model->add_parameters(Dim(parser_config::ACTION_DIM, 1),"action_start");
	p_buffer_guard =model->add_parameters(Dim(parser_config::LSTM_INPUT_DIM, 1),"buffer_guard");
	p_stack_guard = model->add_parameters(Dim(parser_config::LSTM_INPUT_DIM, 1),"stack_guard");

	//------------------------
	fw_lstm = LSTMBuilder(parser_config::LAYERS, parser_config::LSTM_INPUT_DIM, parser_config::LSTM_INPUT_DIM,"fw_lstm", model);
	bw_lstm = LSTMBuilder(parser_config::LAYERS, parser_config::LSTM_INPUT_DIM, parser_config::LSTM_INPUT_DIM,"bw_lstm", model);
	seg_lstm = LSTMBuilder(parser_config::LAYERS, parser_config::LSTM_INPUT_DIM, parser_config::LSTM_INPUT_DIM,"seg_lstm", model);
	fw_lstm.set_dropout(parser_config::DROP_OUT);
	bw_lstm.set_dropout(parser_config::DROP_OUT);
	seg_lstm.set_dropout(parser_config::DROP_OUT);

	p_w2l = model->add_parameters(Dim(parser_config::LSTM_INPUT_DIM, parser_config::LSTM_INPUT_DIM * 2),"state_w2l");
	p_ib = model->add_parameters(Dim(parser_config::LSTM_INPUT_DIM, 1), "state_ib");
	p_tUNK = model->add_parameters(Dim(parser_config::PRETRAINED_DIM, 1), "state_tunk");
	model->pop_scope();
}

void parser_state::Init_Graph(ComputationGraph* hg)
{
	layer::Init_Graph(hg);

	stack_lstm.new_graph(*hg);
	buffer_lstm.new_graph(*hg);
	action_lstm.new_graph(*hg);

	stack_lstm.start_new_sequence();
	buffer_lstm.start_new_sequence();
	action_lstm.start_new_sequence();
	////////////////////////////////////////////////////////////////
	action_start = parameter(*hg, p_action_start);

	fw_lstm.new_graph(*hg);
	bw_lstm.new_graph(*hg);
	seg_lstm.new_graph(*hg);

	fw_lstm.start_new_sequence();
	bw_lstm.start_new_sequence();
	seg_lstm.start_new_sequence();

}

void parser_state::Init_state(const t_unvec& raw_sent, const t_unvec& sent, const t_unvec& sentPos, const t_unvec& sentNER, word_layer& layer)
{
	int sen_sz = (int)sent.size();
	action_lstm.add_input(action_start);
	//------------------------------------------------------------------------
	buffer.resize(sen_sz + 1);  // variables representing word embeddings (possibly including POS info)
	bufferi.resize(sen_sz + 1);
	word2inner_mean.resize(sen_sz + 1);
	vector<Expression> raw_exps(sen_sz);
	for (unsigned i = 0; i < sent.size(); ++i) {
		assert(sent[i] < parser_config::VOCAB_SIZE);
		buffer[sent.size() - i] = layer.build(sent[i], sentPos[i], raw_sent[i],sentNER[i]);
		raw_exps[i] = buffer[sent.size() - i];
		bufferi[sent.size() - i] = i;
		word2inner_mean[i] = buffer[sent.size() - i];
	}
	// dummy symbol to represent the empty buffer
	buffer[0] = parameter(*m_hg, p_buffer_guard);
	bufferi[0] = -999;
	for (auto& b : buffer)
		buffer_lstm.add_input(b);
	//------------------------------------------------------------------------
	stack.push_back(parameter(*m_hg, p_stack_guard));
	stacki.push_back(-999); // not used for anything
	stack_lstm.add_input(stack.back());
	//-----------------------------------------------------------------------
	//
	Expression ib = parameter(*m_hg, p_ib);
	Expression w2l = parameter(*m_hg, p_w2l);
	vector<Expression> raw_fwexps(sen_sz);
	vector<Expression> raw_bwexps(sen_sz);
	for (int i = 0;i < sen_sz;i++) {
		fw_lstm.add_input(raw_exps[i]);
		bw_lstm.add_input(raw_exps[sen_sz - i - 1]);
		raw_fwexps[i] = fw_lstm.back();
		raw_bwexps[sen_sz - i - 1] = bw_lstm.back();
	}
	word2represent.resize(sen_sz);
	for (int i = 0;i < sen_sz;i++) {
		vector<Expression> args = { ib ,w2l ,concatenate({ dropout(raw_fwexps[i], parser_config::DROP_OUT),dropout(raw_bwexps[i], parser_config::DROP_OUT) }) };
		word2represent[i] = rectify(affine_transform(args));
	}
	word2segment.resize(sen_sz);
	for (int i = 0;i < sen_sz;i++) {
		vector<Expression> args = { ib ,w2l ,concatenate({ dropout(raw_fwexps[i], parser_config::DROP_OUT),dropout(raw_bwexps[i], parser_config::DROP_OUT) }) };
		seg_lstm.add_input(affine_transform(args));
		word2segment[i] = seg_lstm.back();
	}
}

void pstate_layer::Init_layer(parser_config& cfg)
{
	layer::Init_layer(cfg);
	Model* model = cfg.model;
	////////////////////////////////////////////////
	model->push_scope(model->get_scope() + "/pstate_layer/");
	p_pbias = model->add_parameters(Dim(parser_config::HIDDEN_DIM, 1),"pbias");
	p_A = model->add_parameters(Dim(parser_config::HIDDEN_DIM, parser_config::HIDDEN_DIM),"A");
	p_B = model->add_parameters(Dim(parser_config::HIDDEN_DIM, parser_config::HIDDEN_DIM),"B");
	p_S = model->add_parameters(Dim(parser_config::HIDDEN_DIM, parser_config::HIDDEN_DIM),"S");
	model->pop_scope();
}

void pstate_layer::Init_Graph(ComputationGraph* hg)
{
	layer::Init_Graph(hg);

	//for parser state
	pbias = parameter(*hg, p_pbias);
	S = parameter(*hg, p_S);
	B = parameter(*hg, p_B);
	A = parameter(*hg, p_A);
}

Expression pstate_layer::build(parser_state& state)
{
	// p_t = pbias + S * slstm + B * blstm + A * almst
	Expression p_t = affine_transform({ pbias, S, state.stack_lstm.back(), B, state.buffer_lstm.back(), A, state.action_lstm.back() });
	return leaky_rectify(p_t,0.01);
}


void action_layer::Init_layer(parser_config& cfg)
{
	layer::Init_layer(cfg);
	Model* model = cfg.model;
	////////////////////////////////////////////////
	model->push_scope(model->get_scope() + "/action_layer/");
	p_p2a = model->add_parameters(Dim(parser_config::ACTION_SIZE, parser_config::HIDDEN_DIM),"p2a");
	p_abias = model->add_parameters(Dim(parser_config::ACTION_SIZE, 1),"abias");
	p_a = model->add_lookup_parameters(parser_config::ACTION_SIZE, Dim(parser_config::ACTION_DIM, 1),"a_embedding");
	model->pop_scope();
}

void action_layer::Init_Graph(ComputationGraph* hg)
{
	layer::Init_Graph(hg);

	p2a = parameter(*hg, p_p2a);
	abias = parameter(*hg, p_abias);
}

Expression action_layer::build(Expression pt, vector<unsigned>& current_valid_actions)
{
	Expression r_t = affine_transform({ abias, p2a, pt });
	// adist = log_softmax(r_t, current_valid_actions)
	return log_softmax(r_t, current_valid_actions);
}
//-------------------------------------
void composition_layer::Init_layer(parser_config& cfg)
{
	layer::Init_layer(cfg);
	Model* model = cfg.model;
	model->push_scope(model->get_scope() + "/composition_layer/");
	////////////////////////////////////////////////
	p_r = (model->add_lookup_parameters(parser_config::ACTION_SIZE, Dim(parser_config::REL_DIM, 1),"r"));
	p_H = (model->add_parameters(Dim(parser_config::LSTM_INPUT_DIM, parser_config::LSTM_INPUT_DIM),"H"));
	p_D = (model->add_parameters(Dim(parser_config::LSTM_INPUT_DIM, parser_config::LSTM_INPUT_DIM),"D"));
	p_R = (model->add_parameters(Dim(parser_config::LSTM_INPUT_DIM, parser_config::REL_DIM),"R"));
	p_cbias = (model->add_parameters(Dim(parser_config::LSTM_INPUT_DIM, 1),"cbias"));

	p_H4pre = (model->add_parameters(Dim(parser_config::LSTM_INPUT_DIM, parser_config::LSTM_INPUT_DIM),"H4pre"));
	p_H4mid = (model->add_parameters(Dim(parser_config::LSTM_INPUT_DIM, parser_config::LSTM_INPUT_DIM),"H4mid"));
	p_H4post = (model->add_parameters(Dim(parser_config::LSTM_INPUT_DIM, parser_config::LSTM_INPUT_DIM),"H4post"));
	p_Horg = (model->add_parameters(Dim(parser_config::LSTM_INPUT_DIM, parser_config::LSTM_INPUT_DIM),"Horg"));
	p_Dprg = (model->add_parameters(Dim(parser_config::LSTM_INPUT_DIM, parser_config::LSTM_INPUT_DIM),"Dprg"));
	p_segBegin = (model->add_parameters(Dim(parser_config::LSTM_INPUT_DIM, 1),"segBegin"));
	p_segEnd = (model->add_parameters(Dim(parser_config::LSTM_INPUT_DIM, 1),"segEnd"));

	//int h2att_dim = 8 * parser_config::LSTM_INPUT_DIM + parser_config::REL_DIM;
	//p_H2atten = (model->add_parameters(Dim(3,h2att_dim),"H2atten"));
	//p_H2atten_bias = (model->add_parameters(Dim(3), "H2atten_bias"));;
	model->pop_scope();
}
void composition_layer::Init_Graph(ComputationGraph* hg)
{
	layer::Init_Graph(hg);
	H = parameter(*hg, p_H);
	D = parameter(*hg, p_D);
	R = parameter(*hg, p_R);
	cbias = parameter(*hg, p_cbias);

	H4pre = parameter(*hg, p_H4pre);
	H4mid = parameter(*hg, p_H4mid);
	H4post = parameter(*hg, p_H4post);
	Horg = parameter(*hg, p_Horg);
	Dorg = parameter(*hg, p_Dprg);

	//H2atten = parameter(*hg, p_H2atten);
	//H2atten_bias = parameter(*hg, p_H2atten_bias);
}
// composed = cbias + H * head + D * dep + R * relation
Expression composition_layer::build(Expression head, int headi, Expression dep, int depi, Expression relation)
{
	Expression segBegin = parameter(*m_hg, p_segBegin);
	Expression segEnd = parameter(*m_hg, p_segEnd);
	Expression horg = (*word2represent)[headi];
	Expression dorg = (*word2represent)[depi];
	Expression hpre, hmid, hpost;
	int minI = headi, maxI = depi;
	if (maxI < minI) {
		int tt = maxI;
		maxI = minI;
		minI = tt;
	}
	hpre = minI > 0 ? (*word2segment)[minI - 1] : segBegin;
	hmid = (*word2segment)[maxI] - hpre;
	hpost = segEnd - (*word2segment)[maxI];


	Expression composed = affine_transform(
	{
		cbias,
		H, head,
		D, dep,
		R, relation ,
		Horg,horg,
		Dorg,dorg,
		H4pre,hpre,
		H4mid,hmid,
		H4post,hpost
	});

	Expression out1 = tanh(composed); //ver1
	return out1;
	//------------------------------
	Expression attn_in = concatenate({ head,dep,horg,dorg,hpre,hmid, hpost ,out1, relation });
	Expression attn = exp( log_softmax( affine_transform({ H2atten_bias ,H2atten,attn_in }) ) );
	Expression out2 =  reshape(concatenate({ head,dep,out1 }), Dim({ (long)parser_config::LSTM_INPUT_DIM,3 }))*attn;
	return out2;
}

void composition_layer::Init_state(const t_unvec& raw_sent,
	const t_unvec& sent,
	const t_unvec& sentPos,
	parser_state& layer)
{
	word2represent = &layer.word2represent;
	word2segment = &layer.word2segment;
}
//-------------------------------------

void ShiftReduceParser::build_setOfActions(dictionary * prel2Int)
{
	//vector<unsigned>	possible_actions;
	//vector<string>		setOfActions;
	int idx = 0;
	setOfActions.push_back("SHIFT");
	possible_actions.push_back(idx++);
	setOfActions.push_back("REDUCE");
	possible_actions.push_back(idx++);

	char buf[512];
	for(int i=0;i<prel2Int->nTokens;i++) {
		sprintf(buf, "LEFT_%d", i);
		setOfActions.push_back(buf);
		possible_actions.push_back(idx++);

		sprintf(buf, "RIGHT_%d", i);
		setOfActions.push_back(buf);
		possible_actions.push_back(idx++);
	}
}

////////////////////////////////////////////////////////////////////////
ShiftReduceParser::ShiftReduceParser(parser_config& cfg) : nn_cfg(cfg)
{
	//nn_cfg.model = model;
	//nn_cfg.pretrained = &pretrained;
	nn_words.Init_layer(nn_cfg);
	nn_parser.Init_layer(nn_cfg);
	nn_pstate.Init_layer(nn_cfg);
	nn_actions.Init_layer(nn_cfg);
	nn_composition.Init_layer(nn_cfg);
}
////////////////////////////////////////////////////////////////////////
vector<pair<int, int>> ShiftReduceParser::log_prob_parser(
	ComputationGraph* hg,
	const vector<int>& raw_sent,  // raw sentence
	const vector<int>& sent,  // sent with oovs replaced
	const vector<int>& sentPos,
	const vector<int>& sentNER, 
	const vector<int>& goldhead,
	const vector<int>& goldrel,
	map<int, vector<int>>&  goldhead2deps,
	bool btrain, double *right,double iter_num) 
{
	nn_words.Init_Graph(hg);
	nn_parser.Init_Graph(hg);
	nn_pstate.Init_Graph(hg);
	nn_actions.Init_Graph(hg);
	nn_composition.Init_Graph(hg);

	LSTMBuilder& stack_lstm = nn_parser.stack_lstm; // (layers, input, hidden, trainer)
	LSTMBuilder& buffer_lstm = nn_parser.buffer_lstm;
	LSTMBuilder& action_lstm = nn_parser.action_lstm;
	nn_parser.clear();
	nn_tree2embedding.clear();
	vector<Expression>& buffer = nn_parser.buffer;
	vector<int>& bufferi = nn_parser.bufferi;
	vector<Expression>& stack = nn_parser.stack;  // variables representing subtree embeddings
	vector<int>& stacki = nn_parser.stacki; //
	nn_parser.Init_state(raw_sent, sent, sentPos,sentNER, nn_words);
	nn_composition.Init_state(raw_sent, sent, sentPos, nn_parser);

	int size_sent = sent.size();
	map<int, vector<int>>  head2deps;
	map<int, int> modify2head;
	vector<pair<int,int>> results(size_sent, pair<int, int>(-1,-1));
	//////////////////////////////////////////////////////////////////////
	const bool build_training_graph = btrain;
	unsigned action_count = 0;  // incremented at each prediction
	while (stack.size() > 1 || buffer.size() > 2) {
		// get list of possible actions for the current parser state
		map<int, bool> legal_transitions;
		Oracle_ArcEager::legal(bufferi, stacki, head2deps, modify2head, goldhead, legal_transitions);

		vector<unsigned> current_valid_actions;
		for (auto a : possible_actions) {
			const string& actionString = setOfActions[a];
			const char ac = actionString[0];
			const char ac2 = actionString[1];
			if (ac == 'S' && ac2 == 'H') {  // SHIFT
				if(legal_transitions[shift_action])
					current_valid_actions.push_back(a);
			}
			else if (ac == 'R' && ac2 == 'E') { // REDUCE
				if (legal_transitions[reduce_action])
					current_valid_actions.push_back(a);
			}
			else if (ac == 'L') { // LEFT 
				if (legal_transitions[left_action])
					current_valid_actions.push_back(a);
			}
			else if (ac == 'R') {// RIGHT
				if (legal_transitions[right_action])
					current_valid_actions.push_back(a);
			}
		}
		//------------------------------
		Expression r_t = nn_pstate.build(nn_parser);
		Expression adiste = nn_actions.build(r_t, current_valid_actions);
		vector<float> adist = as_vector(hg->incremental_forward());
		//------------------------------
		double best_score = adist[current_valid_actions[0]];
		unsigned best_a = current_valid_actions[0];
		for (unsigned i = 1; i < current_valid_actions.size(); ++i) {
			if (adist[current_valid_actions[i]] > best_score) {
				best_score = adist[current_valid_actions[i]];
				best_a = current_valid_actions[i];
			}
		}
		unsigned action = best_a;
		if (build_training_graph) {  // if we have reference actions (for training) use the reference action
			map<int, bool> options;
			int s = stacki.back();
			int b = bufferi.back();
			Oracle_ArcEager::dyn_oracle(bufferi, stacki, goldhead2deps, 
				head2deps, modify2head, goldhead, options);
			//---------------
			if (options.empty()) {
				printf("ERR");
			}
			vector<unsigned> current_oracle_actions;
			for (auto a : possible_actions) {
				const string& actionString = setOfActions[a];
				const char ac = actionString[0];
				const char ac2 = actionString[1];
				if (ac == 'S' && ac2 == 'H') {  // SHIFT
					if (options[shift_action])
						current_oracle_actions.push_back(a);
				}
				else if (ac == 'R' && ac2 == 'E') { // REDUCE
					if (options[reduce_action])
						current_oracle_actions.push_back(a);
				}
				else if (ac == 'L') { // LEFT 
					if (!options[left_action])
						continue;
					int rel = goldrel[s];
					int ifd = actionString.rfind('_');
					string idx = actionString.substr(ifd+1);
					if (atoi(idx.c_str()) != rel)
						continue;
					current_oracle_actions.push_back(a);
				}
				else if (ac == 'R') {// RIGHT
					if (!options[right_action])
						continue;
					int rel = goldrel[b];
					int ifd = actionString.rfind('_');
					string idx = actionString.substr(ifd+1);
					if (atoi(idx.c_str()) != rel)
						continue;
					current_oracle_actions.push_back(a);
				}
			}
			//---------------
			assert(current_oracle_actions.size() > 0);
			int find_action = -1;
			float best_src = -1000;
			bool bfind_best_a = false;
			for (int i = 0;i < current_oracle_actions.size();i++) {
				if (best_a == current_oracle_actions[i])
					bfind_best_a = true;
				if (adist[current_oracle_actions[i]] > best_src) {
					find_action = current_oracle_actions[i];
					best_src = adist[current_oracle_actions[i]];
				}
			}
			/*if (bfind_best_a)
				action = best_a;
			else*/
				action = find_action;

			if (best_a == action) 
				(*right)++; 
			else/* if(!bfind_best_a)*/ {
				if (iter_num >= 0 && rand() / ((float)RAND_MAX) < 0.5f)
					action = best_a;
			}
		}
		++action_count;
		
		//------------------------------
		nn_parser.accum(adiste, action);

		Expression actione = nn_actions.lookup_act(action);
		action_lstm.add_input(actione);
		//------------------------------
		Expression relation = nn_composition.lookup_rel(action);
		// do action
		const string& actionString = setOfActions[action];
		const char ac = actionString[0];
		const char ac2 = actionString[1];
		int s = stacki.back();
		int b = bufferi.back();
		if (ac == 'S' && ac2 == 'H') {  // SHIFT
			assert(buffer.size() > 1); // dummy symbol means > 1 (not >= 1)

			stack.push_back(buffer.back());
			stack_lstm.add_input(buffer.back());
			buffer.pop_back();
			buffer_lstm.rewind_one_step();
			stacki.push_back(bufferi.back());
			bufferi.pop_back();
		}
		else if (ac == 'R' && ac2 == 'E'){ //reduce --- Miguel
			stack.pop_back();
			stacki.pop_back();
			stack_lstm.rewind_one_step();
		}
		else if (ac == 'R') { // LEFT or RIGHT
			assert(stack.size() > 1 && buffer.size() > 1);
			unsigned depi = bufferi.back(), headi = stacki.back();
			Expression dep=buffer.back(), head= stack.back();
			buffer.pop_back();
			bufferi.pop_back();
			buffer_lstm.rewind_one_step();

			Expression nlcomposed = nn_composition.build(head,headi, dep,depi, relation);

			//reflesh head embedding to nlcomposed
			stack.pop_back(); 
			stacki.pop_back();
			stack_lstm.rewind_one_step();
			stack.push_back(nlcomposed);
			stacki.push_back(headi);
			stack_lstm.add_input(nlcomposed);

			//push dep to stack
			stack.push_back(dep);
			stacki.push_back(depi);
			stack_lstm.add_input(dep);

			head2deps[headi].push_back(depi);
			modify2head[depi] = headi;

			int ifd = actionString.rfind('_');
			int rel_idx = atoi(actionString.substr(ifd+1).c_str());
			results[depi].first = headi;
			results[depi].second = rel_idx;
		}
		else if (ac == 'L') {
			assert(stack.size() > 1 && buffer.size() > 1);
			unsigned depi = stacki.back(), headi = bufferi.back();
			Expression dep = stack.back(), head = buffer.back();
			stack.pop_back();
			stacki.pop_back();
			stack_lstm.rewind_one_step();
			buffer.pop_back();
			bufferi.pop_back();
			buffer_lstm.rewind_one_step();

			Expression nlcomposed = nn_composition.build(head,headi, dep,depi, relation);

			//reflesh head embedding to nlcomposed
			buffer.push_back(nlcomposed);
			bufferi.push_back(headi);
			buffer_lstm.add_input(nlcomposed);

			head2deps[headi].push_back(depi);
			modify2head[depi] = headi;

			int ifd = actionString.rfind('_');
			int rel_idx = atoi(actionString.substr(ifd + 1).c_str());
			results[depi].first = headi;
			results[depi].second = rel_idx;
		}
	}
	assert(stack.size() == 1); // guard symbol, root
	assert(stacki.size() == 1);
	assert(buffer.size() == 2); // guard symbol
	assert(bufferi.size() == 2);
	Expression tot_neglogprob = nn_parser.log_p_total();
	assert(tot_neglogprob.pg != nullptr);
	return results;
}
