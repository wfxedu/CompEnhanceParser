#include "cnn/lstm.h"

#include <string>
#include <cassert>
#include <vector>
#include <iostream>

#include "cnn/nodes.h"

using namespace std;
using namespace cnn::expr;

namespace cnn {

enum { X2I, H2I, C2I, BI, X2O, H2O, C2O, BO, X2C, H2C, BC };

LSTMBuilder::LSTMBuilder(unsigned layers,
                         unsigned input_dim,
                         unsigned hidden_dim,
						 std::string ns,
                         Model* model) : layers(layers) {
  unsigned layer_input_dim = input_dim;
  model->push_scope(model->get_scope() + "/LSTM/" + ns);
  char name_buf[128];
  for (unsigned i = 0; i < layers; ++i) {
	  sprintf(name_buf, "_%d", i);
    // i
	  Parameters* p_x2i = model->add_parameters({ (long)hidden_dim, (long)layer_input_dim }, (string)"x2i" + name_buf);
	  Parameters* p_h2i = model->add_parameters({ (long)hidden_dim, (long)hidden_dim }, (string)"h2i" + name_buf);
	  Parameters* p_c2i = model->add_parameters({ (long)hidden_dim, (long)hidden_dim }, (string)"c2i" + name_buf);
	  Parameters* p_bi = model->add_parameters({ (long)hidden_dim }, (string)"bi" + name_buf);

    // o
	  Parameters* p_x2o = model->add_parameters({ (long)hidden_dim, (long)layer_input_dim }, (string)"x2o" + name_buf);
	  Parameters* p_h2o = model->add_parameters({ (long)hidden_dim, (long)hidden_dim }, (string)"h2o" + name_buf);
	  Parameters* p_c2o = model->add_parameters({ (long)hidden_dim, (long)hidden_dim }, (string)"c2o" + name_buf);
	  Parameters* p_bo = model->add_parameters({ (long)hidden_dim }, (string)"bo" + name_buf);

    // c
	  Parameters* p_x2c = model->add_parameters({ (long)hidden_dim, (long)layer_input_dim }, (string)"x2c" + name_buf);
	  Parameters* p_h2c = model->add_parameters({ (long)hidden_dim, (long)hidden_dim }, (string)"h2c" + name_buf);
	  Parameters* p_bc = model->add_parameters({ (long)hidden_dim }, (string)"bc" + name_buf);
    layer_input_dim = hidden_dim;  // output (hidden) from 1st layer is input to next

    vector<Parameters*> ps = {p_x2i, p_h2i, p_c2i, p_bi, p_x2o, p_h2o, p_c2o, p_bo, p_x2c, p_h2c, p_bc};
    params.push_back(ps);
  }  // layers
  dropout_rate = 0.0f;
  model->pop_scope();
}

void LSTMBuilder::new_graph_impl(ComputationGraph& cg){
  param_vars.clear();

  for (unsigned i = 0; i < layers; ++i){
    auto& p = params[i];

    //i
    Expression i_x2i = parameter(cg,p[X2I]);
    Expression i_h2i = parameter(cg,p[H2I]);
    Expression i_c2i = parameter(cg,p[C2I]);
    Expression i_bi = parameter(cg,p[BI]);
    //o
    Expression i_x2o = parameter(cg,p[X2O]);
    Expression i_h2o = parameter(cg,p[H2O]);
    Expression i_c2o = parameter(cg,p[C2O]);
    Expression i_bo = parameter(cg,p[BO]);
    //c
    Expression i_x2c = parameter(cg,p[X2C]);
    Expression i_h2c = parameter(cg,p[H2C]);
    Expression i_bc = parameter(cg,p[BC]);

    vector<Expression> vars = {i_x2i, i_h2i, i_c2i, i_bi, i_x2o, i_h2o, i_c2o, i_bo, i_x2c, i_h2c, i_bc};
    param_vars.push_back(vars);


  }
}

// layout: 0..layers = c
//         layers+1..2*layers = h
void LSTMBuilder::start_new_sequence_impl(const vector<Expression>& hinit) {
  h.clear();
  c.clear();
  if (hinit.size() > 0) {
    assert(layers*2 == hinit.size());
    h0.resize(layers);
    c0.resize(layers);
    for (unsigned i = 0; i < layers; ++i) {
      c0[i] = hinit[i];
      h0[i] = hinit[i + layers];
    }
    has_initial_state = true;
  } else {
    has_initial_state = false;
  }
}

Expression LSTMBuilder::add_input_impl(int prev, const Expression& x) {
  h.push_back(vector<Expression>(layers));
  c.push_back(vector<Expression>(layers));
  vector<Expression>& ht = h.back();
  vector<Expression>& ct = c.back();
  Expression in = x;
  for (unsigned i = 0; i < layers; ++i) {
    const vector<Expression>& vars = param_vars[i];
    Expression i_h_tm1, i_c_tm1;
    bool has_prev_state = (prev >= 0 || has_initial_state);
    if (prev < 0) {
      if (has_initial_state) {
        // intial value for h and c at timestep 0 in layer i
        // defaults to zero matrix input if not set in add_parameter_edges
        i_h_tm1 = h0[i];
        i_c_tm1 = c0[i];
      }
    } else {  // t > 0
      i_h_tm1 = h[prev][i];
      i_c_tm1 = c[prev][i];
    }
	// apply dropout according to http://arxiv.org/pdf/1409.2329v5.pdf
	if (dropout_rate) in = dropout(in, dropout_rate);
    // input
    Expression i_ait;
    if (has_prev_state)
//      i_ait = vars[BI] + vars[X2I] * in + vars[H2I]*i_h_tm1 + vars[C2I] * i_c_tm1;
      i_ait = affine_transform({vars[BI], vars[X2I], in, vars[H2I], i_h_tm1, vars[C2I], i_c_tm1});
    else
//      i_ait = vars[BI] + vars[X2I] * in;
      i_ait = affine_transform({vars[BI], vars[X2I], in});
    Expression i_it = logistic(i_ait);
    // forget
    Expression i_ft = 1.f - i_it;
    // write memory cell
    Expression i_awt;
    if (has_prev_state)
//      i_awt = vars[BC] + vars[X2C] * in + vars[H2C]*i_h_tm1;
      i_awt = affine_transform({vars[BC], vars[X2C], in, vars[H2C], i_h_tm1});
    else
//      i_awt = vars[BC] + vars[X2C] * in;
      i_awt = affine_transform({vars[BC], vars[X2C], in});
    Expression i_wt = tanh(i_awt);
    // output
    if (has_prev_state) {
      Expression i_nwt = cwise_multiply(i_it,i_wt);
      Expression i_crt = cwise_multiply(i_ft,i_c_tm1);
      ct[i] = i_crt + i_nwt;
    } else {
      ct[i] = cwise_multiply(i_it,i_wt);
    }

    Expression i_aot;
    if (has_prev_state)
//      i_aot = vars[BO] + vars[X2O] * in + vars[H2O] * i_h_tm1 + vars[C2O] * ct[i];
      i_aot = affine_transform({vars[BO], vars[X2O], in, vars[H2O], i_h_tm1, vars[C2O], ct[i]});
    else
//      i_aot = vars[BO] + vars[X2O] * in;
      i_aot = affine_transform({vars[BO], vars[X2O], in});
    Expression i_ot = logistic(i_aot);
    Expression ph_t = tanh(ct[i]);
    in = ht[i] = cwise_multiply(i_ot,ph_t);
  }
  if (dropout_rate) return dropout(ht.back(), dropout_rate);
  else return ht.back();
  //return ht.back();
}

void LSTMBuilder::copy(const RNNBuilder & rnn) {
  const LSTMBuilder & rnn_lstm = (const LSTMBuilder&)rnn;
  assert(params.size() == rnn_lstm.params.size());
  for(size_t i = 0; i < params.size(); ++i)
      for(size_t j = 0; j < params[i].size(); ++j)
        params[i][j]->copy(*rnn_lstm.params[i][j]);
}

} // namespace cnn
